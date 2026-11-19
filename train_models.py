"""
Train deployable demo models from the same cleaned table and feature definitions
used by the Synergy-AMP benchmark notebook.

Run:
    python train_models.py --data synergy_clean.csv

This fits final demonstration estimators on all available rows. These are for
interactive exploration only; their in-sample fit is NOT a validation metric.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import torch
from transformers import AutoTokenizer, EsmModel
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EISEN = dict(A=.25, R=-1.76, N=-.64, D=-.72, C=.04, Q=-.69, E=-.62, G=.16,
             H=-.40, I=.73, L=.53, K=-1.10, M=.26, F=.61, P=-.07, S=-.26,
             T=-.18, W=.37, Y=.02, V=.54)


def pep_phys(seq):
    angles = np.deg2rad(100) * np.arange(len(seq))
    h = np.array([EISEN[c] for c in seq])
    hm = np.hypot((h * np.cos(angles)).sum(), (h * np.sin(angles)).sum()) / len(seq)
    p = ProteinAnalysis(seq)
    return [len(seq), p.charge_at_pH(7.4), p.gravy(), p.isoelectric_point(),
            p.aromaticity(), hm, sum(seq.count(a) for a in "KR") / len(seq)]


def main(data_path):
    df = pd.read_csv(data_path).reset_index(drop=True)
    required = {"label", "sequence", "smiles", "species", "gram_neg", "antibiotic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Required columns contain missing values. Use the same cleaned dataset as the notebook.")
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("label must contain binary 0/1 values.")

    sequences = df["sequence"].astype(str).str.upper().unique().tolist()
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    invalid = [s for s in sequences if not s or set(s) - allowed]
    if invalid:
        raise ValueError(f"{len(invalid)} peptide sequences contain non-standard residues. "
                         "Resolve them using the same cleaning rule as the notebook before training.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    esm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device).eval()

    embeddings = {}
    with torch.no_grad():
        for start in range(0, len(sequences), 32):
            batch_seqs = sequences[start:start + 32]
            batch = tokenizer(batch_seqs, return_tensors="pt", padding=True).to(device)
            hidden = esm(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            vectors = ((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy()
            embeddings.update(zip(batch_seqs, vectors))

    phys = {s: pep_phys(s) for s in sequences}
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
    antibiotic_smiles = {}
    for antibiotic, group in df.groupby("antibiotic"):
        smiles_values = group["smiles"].astype(str).unique().tolist()
        if len(smiles_values) != 1:
            raise ValueError(f"{antibiotic!r} maps to multiple SMILES strings; resolve before deployment.")
        antibiotic_smiles[str(antibiotic)] = smiles_values[0]

    abx_cache = {}
    for smi in df["smiles"].astype(str).unique():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES encountered: {smi}")
        desc = [Descriptors.MolLogP(mol), Descriptors.MolWt(mol), Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)]
        abx_cache[smi] = np.concatenate([desc, fpgen.GetFingerprintAsNumPy(mol).astype(float)])

    species_cols = sorted(df["species"].astype(str).unique().tolist())
    species_catalog = {}
    for species in species_cols:
        vals = df.loc[df["species"].astype(str) == species, "gram_neg"].unique()
        if len(vals) != 1:
            raise ValueError(f"Species {species!r} has inconsistent gram_neg values.")
        species_catalog[species] = {"gram_neg": float(vals[0])}

    X_esm = np.vstack([embeddings[s] for s in df["sequence"].astype(str)])
    X_pep = np.vstack([phys[s] for s in df["sequence"].astype(str)])
    X_abx = np.vstack([abx_cache[s] for s in df["smiles"].astype(str)])
    X_bug = np.column_stack([
        df["gram_neg"].astype(float).to_numpy(),
        pd.get_dummies(df["species"].astype(str)).reindex(columns=species_cols, fill_value=0).to_numpy(dtype=float)
    ])
    X_base = np.hstack([X_abx, X_bug])
    X_full = np.hstack([X_esm, X_pep, X_abx, X_bug])
    y = df["label"].astype(int).to_numpy()

    def make():
        return xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, eval_metric="logloss",
            n_jobs=-1, random_state=0
        )

    baseline = make().fit(X_base, y)
    full = make().fit(X_full, y)
    baseline.save_model(MODEL_DIR / "baseline_model.json")
    full.save_model(MODEL_DIR / "full_model.json")

    catalog = {
        "antibiotics": antibiotic_smiles,
        "species": species_catalog,
        "species_columns": species_cols,
    }
    (MODEL_DIR / "catalog.json").write_text(json.dumps(catalog, indent=2))
    card = {
        "dataset_rows": int(len(df)),
        "unique_peptides": int(df["sequence"].nunique()),
        "unique_antibiotics": int(df["antibiotic"].nunique()),
        "unique_species": int(df["species"].nunique()),
        "positive_fraction": float(y.mean()),
        "training_note": "Both models are fitted on all cleaned rows for interactive demonstration. "
                         "These training artifacts are not independently validated predictions.",
        "esm_model": "facebook/esm2_t6_8M_UR50D",
        "feature_dimensions": {"baseline": int(X_base.shape[1]), "full": int(X_full.shape[1])},
    }
    (MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2))
    print("Saved models and catalog to ./models")
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "synergy_clean.csv")
    args = parser.parse_args()
    main(args.data if args.data.is_absolute() else ROOT / args.data)
