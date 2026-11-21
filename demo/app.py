import json
import numpy as np, pandas as pd, xgboost as xgb, gradio as gr

# ZeroGPU Spaces refuse to start unless at least one function is marked @spaces.GPU.
# This model runs on CPU, so the function below is a placeholder that is never called.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        return None
except ImportError:          # running locally without the Hugging Face 'spaces' package
    pass

meta = json.load(open("baseline_meta.json"))
ABX, SPECIES_COLS, GRAM = meta["abx"], meta["species_cols"], meta["species_gram"]
KNOWN = pd.read_csv("known_combinations.csv")
model = xgb.XGBClassifier()
model.load_model("baseline_xgb.json")

DISCLAIMER = (
    "**How to read this.** This is a context-only baseline: it uses the antibiotic and the bacterium, "
    "not the peptide. Its validated AUROC on unseen peptide families is **0.643** (chance = 0.50), "
    "so it is a weak predictor. Peptide sequence is deliberately not an input, because peptide features "
    "did not improve prediction in this study. Research demonstration only; not for clinical or "
    "experimental decisions.")

def features(antibiotic, species):
    onehot = np.zeros(len(SPECIES_COLS))
    onehot[SPECIES_COLS.index(species)] = 1
    return np.concatenate([ABX[antibiotic], [GRAM[species]], onehot]).reshape(1, -1)

def band(p):
    return "below average" if p < 0.35 else ("above average" if p > 0.65 else "close to average")

def predict(antibiotic, species):
    p = float(model.predict_proba(features(antibiotic, species))[0, 1])
    row = KNOWN[(KNOWN.antibiotic == antibiotic) & (KNOWN.species == species)]
    if len(row):
        r = row.iloc[0]
        head = f"### Estimated probability of synergy: {p:.0%}"
        evidence = (f"Published experiments in the dataset: **{int(r.experiments)}** from "
                    f"**{int(r.studies)}** study(ies); observed synergy rate **{r.synergy_rate:.0%}**. "
                    "For pairs with published data the estimate mostly reflects those observations.")
    else:
        head = f"### No published data for this pair. The model leans: {band(p)}"
        evidence = ("There is **no published experiment for this antibiotic and species** in the dataset, so "
                    "the model is extrapolating from similar antibiotics and bacteria. The overall synergy rate "
                    "in the data is 47%. Treat this as a hint, not an estimate.")
    return f"{head}\n\n{evidence}\n\n{DISCLAIMER}"

examples = KNOWN.sort_values("experiments", ascending=False).head(3)[["antibiotic", "species"]].values.tolist()

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Dropdown(sorted(ABX), label="Antibiotic"),
            gr.Dropdown(sorted(SPECIES_COLS), label="Bacterium (species)")],
    outputs=gr.Markdown(),
    examples=examples,
    title="Synergy-AMP: context baseline for AMP-antibiotic synergy",
    description=("Estimates how likely an antimicrobial peptide combined with the chosen antibiotic is to be "
                 "synergistic against the chosen bacterium, from 2,325 published experiments (DBAASP)."))

if __name__ == "__main__":
    demo.launch()
