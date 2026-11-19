# 02_clean_dataset.py - curate the retrieved records into one analysis table.
# Input : raw/*.json (from 01_download_dbaasp.py).  Output: synergy_clean.csv, antibiotic_smiles.csv
import json, os, re, time
import numpy as np, pandas as pd, pubchempy as pcp

# ---------- 1. load every downloaded peptide card ----------
rows = []
for f in os.listdir("raw"):
    if not f.endswith(".json"):
        continue
    try:
        card = json.load(open(f"raw/{f}"))
    except Exception:
        continue
    for s in (card.get("synergies") or []):
        s = dict(s)
        s["peptide_id"] = card.get("id")
        s["sequence"] = card.get("sequence")
        rows.append(s)
syn = pd.json_normalize(rows)
print("raw synergy rows:", len(syn))

# ---------- 2. keep rows where the partner has a NAME ----------
# (blank name = the partner is another peptide, not an antibiotic)
syn = syn[syn["antibioticName"].fillna("").str.strip() != ""].copy()
syn["antibiotic"] = syn["antibioticName"].str.lower().str.strip()

# polymyxin B has no PubChem structure -> use colistin as a stand-in, but flag it
syn["proxy"] = (syn["antibiotic"] == "polymyxin b").astype(int)
syn["antibiotic"] = syn["antibiotic"].replace(
    {"rifampin": "rifampicin", "polymyxin e": "colistin", "polymyxin b": "colistin"})

# remove partners that are not antibacterial drugs
DROP = {"agno3", "ag nanoparticles (agnp)", "amphotericin b", "terbinafine",
        "caspofungin", "cocamidopropyl betaine", "fluconazole", "nystatin"}
syn = syn[~syn["antibiotic"].isin(DROP)]

# ---------- 3. FICI text -> number + synergy label ----------
def parse_fici(x):
    s = (str(x).strip().replace("–", "-").replace("—", "-")
         .replace("≤", "<=").replace("≥", ">=").replace(" ", ""))
    m = re.fullmatch(r"(\d*\.?\d+)-(\d*\.?\d+)", s)          # a range, e.g. 0.19-0.50
    if m:
        lo, hi = sorted([float(m.group(1)), float(m.group(2))])
        lab = 1.0 if hi <= 0.5 else (0.0 if lo > 0.5 else np.nan)
        return (lo + hi) / 2, lab
    m = re.fullmatch(r"(<=|>=|<|>)?(\d*\.?\d+)", s)          # single or bounded value
    if not m:
        return np.nan, np.nan
    op, v = m.group(1), float(m.group(2))
    if op is None:
        return v, float(v <= 0.5)
    if op in ("<", "<="):
        return v, (1.0 if v <= 0.5 else np.nan)
    if op == ">":
        return v, (0.0 if v >= 0.5 else np.nan)
    return v, (0.0 if v > 0.5 else np.nan)                   # ">="

syn[["fici_val", "label"]] = syn["fici"].apply(lambda x: pd.Series(parse_fici(x)))
print("unparseable FICI:", syn["fici_val"].isna().sum(),
      "| ambiguous label:", syn["label"].isna().sum())
syn = syn.dropna(subset=["fici_val", "label"])

# ---------- 4. peptides: standard amino acids only, length 5-50 ----------
syn["sequence"] = syn["sequence"].str.upper().str.strip()
syn = syn[syn["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+")]
syn = syn[syn["sequence"].str.len().between(5, 50)]

# ---------- 5. bacterium species + Gram stain ----------
def species(name):
    w = re.sub(r"[^A-Za-z ]", " ", str(name)).lower().split()
    return " ".join(w[:2]) if len(w) >= 2 else (w[0] if w else "")

syn["species"] = syn["targetSpecie.name"].apply(species)
syn["genus"] = syn["species"].str.split().str[0]

GRAM_NEG = {"escherichia", "klebsiella", "pseudomonas", "acinetobacter", "enterobacter",
            "salmonella", "proteus", "serratia", "citrobacter", "franconibacter",
            "stenotrophomonas", "burkholderia", "haemophilus", "neisseria", "vibrio",
            "shigella", "helicobacter", "campylobacter", "legionella", "yersinia",
            "morganella", "cronobacter", "aeromonas", "edwardsiella"}
GRAM_POS = {"staphylococcus", "enterococcus", "bacillus", "listeria", "streptococcus",
            "clostridium", "micrococcus", "corynebacterium", "lactobacillus",
            "mycobacterium", "clostridioides", "cutibacterium"}
syn["gram_neg"] = syn["genus"].apply(
    lambda g: 1 if g in GRAM_NEG else (0 if g in GRAM_POS else np.nan))
print("dropped (not a mapped bacterium):",
      syn.loc[syn["gram_neg"].isna(), "genus"].value_counts().to_dict())
syn = syn.dropna(subset=["gram_neg"])

# ---------- 6. antibiotic name -> chemical structure (SMILES), cached ----------
def get_smiles(c):
    for attr in ("connectivity_smiles", "isomeric_smiles", "canonical_smiles"):
        v = getattr(c, attr, None)
        if v:
            return v
    return None

ALT = {"spiramycin": ["spiramycin I", "spiramycin"]}
cache_file = "antibiotic_smiles.csv"
smi = {}
if os.path.exists(cache_file):
    old = pd.read_csv(cache_file)
    smi = {k: v for k, v in old.values if pd.notna(v)}

for name in syn["antibiotic"].unique():
    if name in smi:
        continue
    for q in ALT.get(name, [name]):
        try:
            res = pcp.get_compounds(q, "name")
            if res and get_smiles(res[0]):
                smi[name] = get_smiles(res[0])
                break
        except Exception:
            pass
        time.sleep(0.3)
pd.DataFrame(list(smi.items()), columns=["antibiotic", "smiles"]).to_csv(cache_file, index=False)

syn["smiles"] = syn["antibiotic"].map(smi)
print("antibiotics without SMILES (dropped):",
      sorted(syn.loc[syn["smiles"].isna(), "antibiotic"].unique()))
syn = syn.dropna(subset=["smiles"])

# ---------- 7. merge repeated measurements, drop contradictory ones ----------
g = (syn.groupby(["sequence", "antibiotic", "species"])
        .agg(fici_val=("fici_val", "median"), label=("label", "mean"),
             gram_neg=("gram_neg", "first"), smiles=("smiles", "first"),
             reference=("reference", "first"), proxy=("proxy", "max"),
             n=("label", "size")).reset_index())
print("conflicting replicates dropped:", ((g.label > 0) & (g.label < 1)).sum())
g = g[g["label"].isin([0.0, 1.0])].copy()
g["label"] = g["label"].astype(int)
g["log2_fici"] = np.log2(g["fici_val"])
g.to_csv("synergy_clean.csv", index=False)

print("\nFINAL rows:", len(g), "| synergy rate:", round(g["label"].mean(), 3))
print("peptides:", g["sequence"].nunique(), "| antibiotics:", g["antibiotic"].nunique(),
      "| species:", g["species"].nunique(), "| papers:", g["reference"].nunique())
print("Gram-neg share:", round(g["gram_neg"].mean(), 2),
      "| polymyxin-B proxy rows:", int(g["proxy"].sum()))
print(g["antibiotic"].value_counts().head(15))
