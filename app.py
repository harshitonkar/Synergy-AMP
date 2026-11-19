import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="Synergy-AMP", page_icon="🧬", layout="wide")

# Final reported research results supplied for the project report.
TEST1 = pd.DataFrame([
    ["Antibiotic only", 0.594, 0.541, 0.122],
    ["Bacterium only", 0.506, 0.507, 0.002],
    ["Baseline (antibiotic + bacterium)", 0.643, 0.593, 0.207],
    ["Baseline + charge, hydrophobic moment, length", 0.613, 0.572, 0.167],
    ["Baseline + ESM reduced to 10", 0.592, 0.548, 0.125],
    ["Baseline + all 7 simple peptide features", 0.590, 0.562, 0.110],
    ["Full", 0.586, 0.533, 0.127],
], columns=["Model", "AUROC", "AUPRC", "MCC"])

TEST2 = pd.DataFrame([
    ["Full", 0.758, 0.743, 0.376],
    ["Baseline", 0.590, 0.552, 0.114],
    ["Peptide-only", 0.767, 0.757, 0.384],
], columns=["Model", "AUROC", "AUPRC", "MCC"])

TEST3 = pd.DataFrame([
    ["Baseline (antibiotic + bacterium)", 0.669, 0.645, 0.247],
    ["Full", 0.652, 0.635, 0.236],
    ["Peptide-only", 0.613, 0.588, 0.171],
], columns=["Model", "AUROC", "AUPRC", "MCC"])


@st.cache_resource
def load_artifacts():
    required = [
        MODEL_DIR / "baseline_model.json",
        MODEL_DIR / "full_model.json",
        MODEL_DIR / "catalog.json",
        MODEL_DIR / "model_card.json",
    ]
    if not all(p.exists() for p in required):
        return None
    catalog = json.loads((MODEL_DIR / "catalog.json").read_text())
    card = json.loads((MODEL_DIR / "model_card.json").read_text())
    import xgboost as xgb
    base = xgb.XGBClassifier()
    base.load_model(MODEL_DIR / "baseline_model.json")
    full = xgb.XGBClassifier()
    full.load_model(MODEL_DIR / "full_model.json")
    return base, full, catalog, card


def peptide_features(sequence):
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    eisen = dict(A=.25, R=-1.76, N=-.64, D=-.72, C=.04, Q=-.69, E=-.62,
                 G=.16, H=-.40, I=.73, L=.53, K=-1.10, M=.26, F=.61,
                 P=-.07, S=-.26, T=-.18, W=.37, Y=.02, V=.54)
    angles = np.deg2rad(100) * np.arange(len(sequence))
    h = np.array([eisen[c] for c in sequence])
    hm = np.hypot((h * np.cos(angles)).sum(), (h * np.sin(angles)).sum()) / len(sequence)
    p = ProteinAnalysis(sequence)
    return np.array([len(sequence), p.charge_at_pH(7.4), p.gravy(),
                     p.isoelectric_point(), p.aromaticity(), hm,
                     sum(sequence.count(a) for a in "KR") / len(sequence)], dtype=float)


@st.cache_resource
def load_esm():
    import torch
    from transformers import AutoTokenizer, EsmModel
    model_id = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = EsmModel.from_pretrained(model_id)
    model.eval()
    return tokenizer, model, torch


def embed_sequence(sequence):
    tokenizer, model, torch = load_esm()
    with torch.no_grad():
        batch = tokenizer([sequence], return_tensors="pt", padding=True)
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        emb = ((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy()[0]
    return emb.astype(float)


def antibiotic_features(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("The stored SMILES for this antibiotic could not be parsed.")
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
    desc = [Descriptors.MolLogP(mol), Descriptors.MolWt(mol), Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)]
    return np.concatenate([np.asarray(desc, dtype=float),
                           fpgen.GetFingerprintAsNumPy(mol).astype(float)])


st.title("🧬 Synergy-AMP")
st.caption("A leakage-controlled research benchmark for antimicrobial peptide–antibiotic synergy prediction")
st.warning(
    "Research demonstration only. Predictions are computational outputs from models trained on a "
    "literature-derived dataset. They are not experimentally validated, are not clinical advice, "
    "and must not be used to select treatments."
)

tabs = st.tabs(["Combination tester", "Research dashboard", "Methods & limitations"])

with tabs[0]:
    st.subheader("Test a peptide–antibiotic–bacterium combination")
    artifacts = load_artifacts()
    if artifacts is None:
        st.info(
            "The interactive predictor is not activated yet. The research dashboard is available, "
            "but the trained model files and feature catalog must first be generated with "
            "`python train_models.py --data synergy_clean.csv` and added to `models/`."
        )
    else:
        baseline_model, full_model, catalog, card = artifacts
        c1, c2 = st.columns(2)
        with c1:
            sequence = st.text_area("Peptide sequence (20 standard amino acids)", height=110,
                                    placeholder="KKW...").strip().upper().replace(" ", "").replace("\n", "")
            antibiotic = st.selectbox("Antibiotic", options=sorted(catalog["antibiotics"].keys()))
        with c2:
            species = st.selectbox("Bacterium", options=sorted(catalog["species"].keys()))
            st.caption("Only antibiotics and bacterial species present in the training data are supported.")
        if st.button("Compare baseline and full model", type="primary", use_container_width=True):
            if not sequence or any(ch not in "ACDEFGHIKLMNPQRSTVWY" for ch in sequence):
                st.error("Enter a non-empty sequence containing only the 20 standard amino-acid letters.")
            elif len(sequence) > 500:
                st.error("For this demo, peptide sequences are limited to 500 residues.")
            else:
                try:
                    abx = antibiotic_features(catalog["antibiotics"][antibiotic])
                    species_cols = catalog["species_columns"]
                    bug = np.array([catalog["species"][species]["gram_neg"]] +
                                   [1.0 if s == species else 0.0 for s in species_cols],
                                   dtype=float)
                    x_base = np.concatenate([abx, bug]).reshape(1, -1)
                    esm = embed_sequence(sequence)
                    phys = peptide_features(sequence)
                    x_full = np.concatenate([esm, phys, abx, bug]).reshape(1, -1)
                    p_base = float(baseline_model.predict_proba(x_base)[0, 1])
                    p_full = float(full_model.predict_proba(x_full)[0, 1])
                    st.subheader("Model outputs")
                    a, b = st.columns(2)
                    a.metric("Baseline score", f"{p_base:.1%}")
                    b.metric("Full-model score", f"{p_full:.1%}")
                    st.caption(
                        "These are model scores, not calibrated clinical probabilities. The baseline "
                        "uses antibiotic + bacterium only; it does not evaluate the peptide sequence."
                    )
                    st.info(
                        "Interpretation: compare the outputs as exploratory model behaviour only. "
                        "The benchmark did not show that peptide features improve prediction on unseen "
                        "peptide families."
                    )
                except Exception as exc:
                    st.error(f"Could not calculate this combination: {exc}")

with tabs[1]:
    st.subheader("Benchmark results")
    st.markdown("### Test 1 · Unseen peptide families")
    st.caption("Five-seed mean ± standard deviation. Values below are the reported benchmark results.")
    st.dataframe(TEST1, use_container_width=True, hide_index=True)
    st.bar_chart(TEST1.set_index("Model")[["AUROC"]])
    st.markdown("### Test 2 · Unseen antibiotic")
    st.caption("Single leave-one-antibiotic-out run.")
    st.dataframe(TEST2, use_container_width=True, hide_index=True)
    st.markdown("### Test 3 · Unseen paper")
    st.caption("Single leave-one-paper-out run.")
    st.dataframe(TEST3, use_container_width=True, hide_index=True)
    st.markdown("### Main interpretation")
    st.write(
        "On unseen peptide families, the antibiotic + bacterium baseline (AUROC 0.643 ± 0.005) "
        "outperformed the full model (0.586 ± 0.010). In the unseen-antibiotic test, peptide-only "
        "slightly exceeded the full model, suggesting the apparent transfer result may reflect "
        "peptide or study-level patterns rather than robust peptide–antibiotic interaction learning. "
        "The study-held-out numbers are exploratory: the current curation step can collapse records from multiple references, so they are not definitive evidence of unseen-study generalization."
    )

with tabs[2]:
    st.subheader("Methods")
    st.markdown("""
- **Dataset:** 2,325 peptide–antibiotic–bacterium records; 419 unique peptides; 86 antibiotics; 22 referenced studies.
- **Label:** synergy derived from fractional inhibitory concentration index (FICI) parsing in the curated dataset.
- **Peptide representation:** ESM-2 `esm2_t6_8M_UR50D` embeddings (320 dimensions) plus seven physicochemical descriptors.
- **Antibiotic representation:** five RDKit descriptors and a 256-bit Morgan fingerprint.
- **Bacterial representation:** Gram-negative indicator and one-hot species encoding.
- **Classifier:** XGBoost.
- **Validation:** peptide-similarity grouped cross-validation, leave-one-antibiotic-out, and leave-one-paper-out.
""")
    st.subheader("Limitations")
    st.markdown("""
1. Data are dominated by a small number of studies; the largest study contributes about 60% of rows.
2. The positive (synergy) class represents 46.7% of records, so publication bias is possible.
3. Between-paper effects may influence the baseline.
4. FICI labels are noisy; ambiguous ranges were excluded.
5. Colistin structure is used as a proxy for polymyxin B in 121 rows.
6. Peptide representation omits modifications such as amidation and removes non-standard residues.
7. Leave-one-antibiotic-out and leave-one-paper-out results are single runs.
8. No wet-lab validation has been performed.
""")
    st.caption("This tool is an educational/research prototype, not a drug-discovery product or clinical decision system.")

st.divider()
st.caption("Synergy-AMP · Research prototype · Reproducibility and limitations take priority over predictive claims")
