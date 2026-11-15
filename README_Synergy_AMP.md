# Synergy-AMP
## A leakage-controlled benchmark for antimicrobial peptide–antibiotic synergy prediction

Synergy-AMP investigates whether antimicrobial peptide (AMP) sequence information improves prediction of AMP–antibiotic synergy against bacteria. It benchmarks peptide, antibiotic-chemistry and bacterial-context features under validation schemes designed to reduce peptide-family, antibiotic and study-level leakage.

> **Main finding:** On the similarity-grouped test of unseen peptide families, the antibiotic-plus-bacterium baseline outperformed the full multimodal model. In this dataset and evaluation setup, the tested peptide features did not provide incremental predictive value over that baseline.

This is a benchmarking result, not a drug-discovery claim. No candidate peptide was experimentally validated, and no wet-lab validation was performed.

---

## Project at a glance

| Item | Description |
|---|---|
| Data source | DBAASP (Antimicrobial Peptide Database), accessed through its API |
| Cleaned records | 2,325 peptide–antibiotic–bacterium records |
| Unique peptides | 419 |
| Antibiotics | 86 |
| Bacterial species | 38 |
| Referenced studies | 22 papers with a reference number; 7 rows have no reference |
| Label | Synergy = 1 when FICI ≤ 0.5; otherwise 0 |
| Class balance | 46.7% synergistic rows |
| Main estimator | XGBoost classifier |
| Peptide representation | ESM-2 `esm2_t6_8M_UR50D` embeddings and physicochemical descriptors |
| Antibiotic representation | RDKit molecular descriptors and Morgan fingerprints |
| Validation | Similarity-grouped peptide split, leave-one-antibiotic-out, leave-one-paper-out |

## Why this benchmark matters

Random row-level splits can put near-identical peptide variants in both training and test sets. A model can then appear to generalize while recognizing related sequences or study-specific patterns. Synergy-AMP groups similar peptide sequences before evaluating unseen-family performance and adds two complementary tests: holding out antibiotics and holding out papers.

The key comparison is against a baseline that receives antibiotic and bacterium information but no peptide features. A peptide-enhanced model is useful only if it improves on this baseline under a relevant held-out evaluation.

## Data processing

The project collected peptide records from DBAASP, parsed reported fractional inhibitory concentration index (FICI) values, mapped bacterial Gram status, and looked up antibiotic structures for chemical feature generation.

The final modelling table contains 2,325 records, 419 unique peptides, 86 antibiotics and 38 bacterial species. Records with FICI ranges straddling the 0.5 threshold were excluded from binary labelling. Polymyxin B could not be resolved to a PubChem structure and was represented using colistin as a flagged structural stand-in for 121 rows.

**Label definition**

- `1`: FICI ≤ 0.5 (synergy)
- `0`: FICI > 0.5 (not labelled synergistic; includes additive/indifferent and rare antagonistic records)

The observed 46.7% positive rate is a property of this published dataset, not an estimate of the natural prevalence of synergy.

## Feature groups and models

- **Antibiotic + bacterium baseline:** antibiotic descriptors/fingerprint and bacterial identity/context.
- **Peptide-only control:** peptide ESM-2 and simple peptide features, used in particular to diagnose peptide-family memorization in the new-antibiotic test.
- **Full model:** peptide features combined with antibiotic chemistry and bacterial context.
- **Feature ablations:** antibiotic-only, bacterium-only, baseline plus selected simple peptide features, baseline plus all seven simple peptide features, and baseline plus ESM reduced to 10 dimensions.

The peptide representation used the 8-million-parameter ESM-2 model `esm2_t6_8M_UR50D`. Simple peptide descriptors included length, charge, hydrophobicity, isoelectric point, aromaticity, hydrophobic moment and fraction of lysine/arginine. Antibiotic features were computed from available structures using RDKit.

## Validation design

### Test 1 — New peptide families

Similar peptide sequences were clustered using global alignment with gap penalties and average-linkage clustering. The selected distance threshold was 0.8, producing 71 clusters; the largest cluster contained 10% of rows. A stratified grouped 5-fold split kept each peptide cluster wholly in either training or testing.

Results are reported as mean ± standard deviation across five seeds.

| Model | AUROC | AUPRC | MCC |
|---|---:|---:|---:|
| Antibiotic only | 0.594 ± 0.009 | 0.541 ± 0.011 | 0.122 ± 0.016 |
| Bacterium only | 0.506 ± 0.025 | 0.507 ± 0.016 | 0.002 ± 0.064 |
| **Baseline (antibiotic + bacterium)** | **0.643 ± 0.005** | **0.593 ± 0.009** | **0.207 ± 0.014** |
| Baseline + charge, hydrophobic moment, length | 0.613 ± 0.016 | 0.572 ± 0.015 | 0.167 ± 0.028 |
| Baseline + ESM reduced to 10 | 0.597 ± 0.018 | 0.549 ± 0.017 | 0.135 ± 0.029 |
| Baseline + all 7 simple peptide features | 0.590 ± 0.024 | 0.562 ± 0.019 | 0.110 ± 0.036 |
| Full | 0.585 ± 0.011 | 0.533 ± 0.018 | 0.126 ± 0.015 |

The baseline is best on all three reported metrics. None of the tested peptide-feature additions improves on it. The differences are descriptive benchmark results; they should not be presented as statistically significant without an appropriate paired uncertainty analysis.

### Test 2 — New antibiotic

Leave-one-antibiotic-out validation holds out all records for one antibiotic at a time. The following results are from a **single run**, not a multi-seed estimate.

| Model | AUROC | AUPRC | MCC |
|---|---:|---:|---:|
| Full | 0.758 | 0.743 | 0.376 |
| Baseline | 0.590 | 0.552 | 0.114 |
| Peptide-only | 0.767 | 0.757 | 0.384 |

Although the full model scores well on this test, the peptide-only control scores slightly higher. This is consistent with peptide-family or paper-level patterns contributing strongly to the apparent new-antibiotic result. It does **not** support a claim that the multimodal model has learned robust new-antibiotic synergy biology.

### Test 3 — New paper

Leave-one-paper-out validation holds out all records associated with one referenced paper at a time. These are **single-run** results.

| Model | AUROC | AUPRC | MCC |
|---|---:|---:|---:|
| Baseline (antibiotic + bacterium) | **0.669** | **0.645** | **0.247** |
| Full | 0.651 | 0.634 | 0.239 |
| Peptide-only | 0.612 | 0.587 | 0.161 |

The baseline is again slightly ahead of the full model. With only 22 referenced papers and one run, the small gap should not be described as statistically significant. The peptide-only model is above chance but does not add value on top of the baseline.

### Chance references

For this dataset, AUROC chance is 0.50. The AUPRC reference for a non-informative ranking is approximately the positive-class share, 0.467. MCC chance reference is 0.00.

## Robustness and diagnostic checks

- **Polymyxin-B stand-in:** removing the 121 rows represented with colistin changes baseline Test 1 AUROC from 0.643 to 0.642. This suggests the baseline result is not materially driven by the stand-in.
- **Paper concentration:** paper 1 contributes 1,395 rows (about 60%); papers 1–3 contribute about 84%. Seven rows have no reference number.
- **Within-paper comparison:** baseline AUROC is 0.588 within paper 1 and 0.595 across the other referenced papers. The full model is below the baseline in both subsets.
- **SHAP diagnostic:** the baseline model's leading mean absolute SHAP features include antibiotic molecular weight, bacterial-species indicators and antibiotic fingerprint bits. SHAP describes model reliance; it does not establish causal biological mechanisms.

## Interpretation

Across the primary unseen-peptide-family test, the antibiotic-plus-bacterium baseline performs better than the full model. In the new-antibiotic test, peptide-only performance is comparable to or slightly above the full model, suggesting that the apparent score may be driven by peptide-related patterns rather than transferable peptide–antibiotic interactions. In the new-paper test, the baseline again performs slightly better.

The most defensible conclusion is therefore narrow: **with the available DBAASP records, current feature representations and validation design, peptide features did not demonstrate incremental predictive value over antibiotic and bacterial context.** This does not prove that AMP sequence biology is irrelevant; it shows that this dataset and modelling setup do not establish that added value.

## Limitations

1. **Study concentration:** the dataset is dominated by a few studies. Paper 1 contributes about 60% of rows; papers 1–3 contribute about 84%. Only 22 papers have a reference number, and 7 rows have none.
2. **Publication bias:** 46.7% of rows are labelled synergistic, much higher than would be expected from arbitrary combinations.
3. **Between-paper effects:** part of the baseline signal may reflect which antibiotics and species each paper tested, rather than generalizable biology.
4. **Noisy labels:** FICI can vary by laboratory and method. FICI ranges that straddled 0.5 were dropped.
5. **Polymyxin-B proxy:** polymyxin B is represented by the colistin structure for 121 rows because no PubChem record was found.
6. **Limited peptide representation:** the 8M ESM-2 model was used; terminal modifications such as amidation were not represented, and non-standard residues were removed.
7. **Single-run external tests:** leave-one-antibiotic-out and leave-one-paper-out results are single runs without seed-spread estimates.
8. **No wet-lab validation:** all results are computational. This is not a drug-discovery or therapeutic-efficacy claim.

## Reproducibility and repository layout

Recommended repository structure:

```text
synergy-amp/
├── README.md
├── notebooks/
│   └── Synergy_AMP.ipynb
├── scripts/
│   ├── 00_download.py
│   ├── 01_clean.py
│   ├── 02_colab_main.py
│   └── 03_colab_day2.py
├── results/
│   ├── results_summary.csv
│   └── figures/
├── data/                 # only if redistribution is permitted
└── requirements.txt
```

Run the notebook in Google Colab. The notebook installs the required packages, loads `synergy_clean.csv`, computes peptide and antibiotic features, constructs peptide similarity groups, evaluates the three validation tests, runs the seed and robustness checks, and computes SHAP diagnostics.

For exact reproduction, preserve the notebook, its random seeds, package versions, the cleaned input table and the scripts used to generate it. Colab package versions can change, so a future reproduction should record the environment and dependency versions explicitly.

**Data note:** Do not commit raw DBAASP downloads or the cleaned dataset to a public repository unless the database's current terms permit redistribution. If redistribution is not permitted, publish the download/cleaning scripts and instructions for obtaining the source data independently. Never commit API keys, credentials or personal tokens.

## Resume-ready summary

- Curated and cleaned 2,325 peptide–antibiotic–bacterium synergy records (419 peptides, 86 antibiotics and 22 referenced studies) from DBAASP using its API, with reproducible FICI parsing, Gram-stain mapping and PubChem structure lookup.
- Designed leakage-controlled validation using peptide-similarity-grouped splits, leave-one-antibiotic-out and leave-one-paper-out tests; found that the peptide-only model matched or exceeded the full model on the new-antibiotic test (AUROC 0.767 vs 0.758).
- Benchmarked ESM-2 embeddings and physicochemical peptide features against an antibiotic-plus-bacterium baseline over five seeds; peptide features did not improve unseen-peptide-family prediction (AUROC 0.585 vs 0.643).

## Scope and disclaimer

Synergy-AMP is a computational benchmark of published records. Its outputs are not clinical recommendations, antimicrobial susceptibility results, or experimentally validated peptide candidates. Any biological hypothesis requires independent laboratory validation.
