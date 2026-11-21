# Standalone export for the Hugging Face Space. Needs no GPU, no ESM-2 and none of the notebook cells.
# In Colab a CPU runtime is enough. Paste this whole block into ONE new cell of a fresh notebook.
!pip install -q rdkit xgboost==3.4.1

import json, os, shutil
import numpy as np, pandas as pd, xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from google.colab import files

uploaded = files.upload()                      # choose synergy_clean.csv
df = pd.read_csv("synergy_clean.csv").reset_index(drop=True)
y = df["label"].values

# --- antibiotic features: identical to the notebook ---
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
def abx_feat(smi):
    m = Chem.MolFromSmiles(smi)
    d = [Descriptors.MolLogP(m), Descriptors.MolWt(m), Descriptors.TPSA(m),
         Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m)]
    return np.concatenate([d, fpgen.GetFingerprintAsNumPy(m).astype(float)])

abx = {s: abx_feat(s) for s in df["smiles"].unique()}
species_cols = list(pd.get_dummies(df["species"]).columns)
X_abx = np.vstack([abx[s] for s in df["smiles"]])
X_bug = np.hstack([df[["gram_neg"]].values, pd.get_dummies(df["species"]).values.astype(float)])
X_base = np.hstack([X_abx, X_bug])             # antibiotic + bacterium only (no peptide information)
print("feature matrix:", X_base.shape)         # expect (2325, 300)

# --- the same XGBoost settings and seed as the notebook ---
model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.5, eval_metric="logloss", n_jobs=-1, random_state=0)
model.fit(X_base, y)

os.makedirs("space", exist_ok=True)
model.save_model("space/baseline_xgb.json")
abx_by_name = {name: abx[smi].tolist()
               for name, smi in df[["antibiotic", "smiles"]].drop_duplicates().values}
species_gram = {sp: int(g) for sp, g in df.groupby("species")["gram_neg"].first().items()}
json.dump({"abx": abx_by_name, "species_cols": species_cols, "species_gram": species_gram},
          open("space/baseline_meta.json", "w"))
known = (df.groupby(["antibiotic", "species"])
           .agg(experiments=("label", "size"), studies=("reference", "nunique"),
                synergy_rate=("label", "mean")).round(3).reset_index())
known.to_csv("space/known_combinations.csv", index=False)
print(len(abx_by_name), "antibiotics |", len(species_cols), "species |", len(known), "pairs")

shutil.make_archive("space_files", "zip", "space")
files.download("space_files.zip")
