# Synergy-AMP

**A leakage-controlled benchmark: do peptide sequence features improve prediction of antimicrobial-peptide (AMP) + antibiotic synergy?**

> **Main finding.** Under validation that withholds similar peptides, whole antibiotics, or whole studies from training, a model that knows only
> *which antibiotic* is used *against which bacterium* outperforms every model that also receives peptide features (ESM-2 embeddings or simple
> physicochemical descriptors). An apparent ability to generalize to unseen antibiotics was matched by a model that sees **only the peptide**,
> indicating memorization of peptides rather than learned chemistry.

This is an in-silico analysis of published data. Nothing here is experimentally validated, and no new drugs or peptides are claimed.

---

## 1. Background

Antibiotic resistance is increasing. Gram-negative bacteria are difficult to treat partly because their outer membrane blocks large drugs such as vancomycin.
Antimicrobial peptides can damage that membrane, so a peptide combined with an antibiotic can be far more effective than either alone (**synergy**).
Synergy is measured with a checkerboard assay and summarized as the **fractional inhibitory concentration index (FICI)**; FICI ≤ 0.5 is the conventional threshold for synergy.
Because testing every peptide/antibiotic/bacterium combination is impractical, a model that ranks combinations would be useful, provided it works on combinations it has not seen.
This project tests whether that is currently achievable with public data.

## 2. Data

| | |
|---|---|
| Source | [DBAASP](https://dbaasp.org) peptide records (synergy section), retrieved through its public web service in November 2026 |
| Raw | 843 peptides, 5,617 synergy records |
| Curated | **2,325 rows**, 419 peptides, 86 antibiotics, 38 bacterial species, 22 referenced studies (7 rows carry no reference) |
| Label | `1` if FICI ≤ 0.5, otherwise `0`; ranges that straddle 0.5 are excluded. 46.7 % of rows are positive |
| Targets | 74 % Gram-negative |

Curation (`scripts/02_clean_dataset.py`) retains records whose partner is a named antibacterial drug (a blank partner name denotes another peptide), parses FICI text
such as `0.19-0.50` or `<0.19`, keeps peptides composed of the 20 standard amino acids with length 5 to 50, maps targets to species and Gram stain (non-bacterial targets are
excluded), resolves each antibiotic's structure through PubChem, and merges repeated measurements (contradictory repeats are excluded).

**The dataset is not redistributed in this repository.** It can be rebuilt with the scripts below. Consult the DBAASP terms of use before sharing derived data.

## 3. Methods

* **Features (627 per row).**
  Peptide: 320-dimensional mean-pooled ESM-2 embedding (`esm2_t6_8M_UR50D`) and 7 descriptors (length, net charge at pH 7.4, GRAVY, isoelectric point, aromaticity, hydrophobic moment, K+R fraction).
  Antibiotic: 5 RDKit descriptors and a 256-bit Morgan fingerprint (radius 2).
  Bacterium: Gram-negative indicator and one-hot species.
* **Model.** XGBoost (300 trees, depth 4, learning rate 0.05). The model is deliberately simple; the contribution is the validation design.
* **Baseline.** The same model without any peptide features (antibiotic and bacterium only).
* **Control.** A peptide-only model (no antibiotic, no bacterium) to detect memorization.
* **Validation (three questions, three splits).**
  1. *Unseen peptide families:* peptides are clustered by global-alignment similarity (average linkage, distance cut-off 0.8; 71 groups, largest = 10 % of rows) and whole groups are withheld (5-fold, 5 seeds).
  2. *Unseen antibiotic:* leave-one-antibiotic-out.
  3. *Unseen study:* leave-one-study-out.
* **Metrics.** AUROC, AUPRC, MCC. Chance level: AUROC 0.50, AUPRC 0.467 (the positive rate), MCC 0.

A first grouping based on longest-common-subsequence similarity with single-linkage clustering placed 82 % of the rows in one group, which made the grouped split invalid;
results from that grouping were discarded and are not reported. The notebook documents the corrected procedure.

## 4. Results

### 4.1 Unseen peptide families (5 seeds, mean ± SD)

| Model | AUROC | AUPRC | MCC |
|---|---|---|---|
| Antibiotic only | 0.594 ± 0.009 | 0.541 ± 0.011 | 0.122 ± 0.016 |
| Bacterium only | 0.506 ± 0.025 | 0.507 ± 0.016 | 0.002 ± 0.064 |
| **Baseline (antibiotic + bacterium)** | **0.643 ± 0.005** | **0.593 ± 0.009** | **0.207 ± 0.014** |
| Baseline + charge, hydrophobic moment, length | 0.613 ± 0.016 | 0.572 ± 0.015 | 0.167 ± 0.028 |
| Baseline + ESM-2 reduced to 10 dimensions (PCA) | 0.592 ± 0.017 | 0.548 ± 0.014 | 0.125 ± 0.023 |
| Baseline + all 7 simple peptide descriptors | 0.590 ± 0.024 | 0.562 ± 0.019 | 0.110 ± 0.036 |
| Full model (all features) | 0.586 ± 0.010 | 0.533 ± 0.016 | 0.127 ± 0.014 |

Every variant that adds peptide features scores below the baseline, by more than the seed-to-seed spread.

### 4.2 Unseen antibiotic (leave-one-antibiotic-out, single run)

| Model | AUROC | AUPRC | MCC |
|---|---|---|---|
| Full model | 0.758 | 0.743 | 0.376 |
| Baseline | 0.590 | 0.552 | 0.114 |
| **Peptide-only** | **0.767** | 0.757 | 0.384 |

The full model scores well here, but a model that sees only the peptide scores as well. The same peptides occur with other antibiotics in the training data,
so the score reflects peptide memorization (some peptides synergize frequently), not generalization to new antibiotics.

### 4.3 Unseen study (leave-one-study-out, single run)

96 % of peptides occur in only one study, so this split is close to a test on unseen peptides from unseen laboratories.

| Model | AUROC | AUPRC | MCC |
|---|---|---|---|
| Baseline | 0.669 | 0.645 | 0.247 |
| Full model | 0.652 | 0.635 | 0.236 |
| Peptide-only | 0.613 | 0.588 | 0.171 |

### 4.4 Robustness checks

* **Polymyxin B stand-in.** No PubChem structure was found for polymyxin B, so its 121 rows use the colistin structure (the two differ by one amino acid) and are flagged `proxy = 1`.
  Excluding them leaves the baseline unchanged (AUROC 0.643 → 0.642); the full model decreases (0.586 → 0.552). The conclusion is unchanged.
* **Study composition.** Study 1 supplies 1,395 of the 2,325 rows (60 %) and 300 of 419 peptides; studies 1 to 3 supply about 84 %.
  Within study 1 alone the baseline scores AUROC 0.588 ± 0.014 and the full model 0.565 ± 0.016; within all other studies 0.595 ± 0.015 versus 0.542 ± 0.048.
  The baseline's overall 0.64 exceeds both subsets, which suggests part of its signal arises from differences between studies (which antibiotics and species each tested).
* **Baseline drivers (SHAP).** Leading features: antibiotic molecular weight, specific species (*K. pneumoniae*, *P. aeruginosa*), antibiotic logP, and fingerprint bits that largely identify particular drugs.

## 5. Interpretation and scope

**Shown:** with leakage-controlled validation on this public dataset, ESM-2 embeddings and simple physicochemical peptide descriptors did not improve synergy prediction beyond antibiotic and bacterial context,
and a favorable-looking score under a weaker split reflected memorization.

**Not shown:** that peptide properties are biologically irrelevant. The result concerns these features, this model, and roughly 400 peptides of noisy, study-dominated data.
Consequently, a genetic-algorithm peptide designer was not pursued: it would optimize a model that cannot score unseen peptides.

## 6. Limitations

1. The data are dominated by a few studies (study 1 = 60 % of rows); only 22 studies carry a reference number.
2. Publication bias: a 46.7 % positive rate is far above what random pairs would yield, so the model learns what is tested and published.
3. FICI is noisy (strains, media, laboratories). Ranges straddling 0.5 were excluded rather than assigned.
4. Between-study effects inflate the baseline (about 0.64 overall versus about 0.59 within each study group).
5. Polymyxin B is represented by the colistin structure (121 rows).
6. Peptide representations are limited: an 8M-parameter ESM-2 model, terminal modifications (such as amidation) ignored, non-standard residues removed.
7. The leave-one-out evaluations are single runs without seed variation; rows without a reference form one group in leave-one-study-out.
8. No experimental validation.

## 7. Reproducing the analysis

```bash
pip install -r requirements.txt            # data scripts only
# 1. In the DBAASP search page, filter to peptides with synergy data and export the list as peptides.csv (needs an 'ID' column).
python scripts/01_download_dbaasp.py       # raw/<ID>.json for each peptide (about 15 min, resumable)
python scripts/02_clean_dataset.py         # synergy_clean.csv and antibiotic_smiles.csv
# 2. Open notebooks/synergy_amp_benchmark.ipynb in Google Colab, select a T4 GPU runtime,
#    upload synergy_clean.csv when prompted, then Runtime > Restart and run all (about 15-30 min).
```

Later DBAASP releases may change record counts slightly; small numeric differences are expected (two runs of the same code on different library builds differed by at most 0.01 in any reported metric).
The reported run used Python 3.13.16, numpy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1, scipy 1.16.3, xgboost 3.4.1, rdkit 2026.09.2, biopython 1.88, shap 0.54.1, torch 2.14.1 and transformers 5.22.1 (recorded in the notebook's last section).

## 8. Repository layout

```
scripts/     01_download_dbaasp.py   02_clean_dataset.py
notebooks/   synergy_amp_benchmark.ipynb
README.md    requirements.txt    .gitignore
```

## 9. References

* Pirtskhalava M, Amstrong AA, Grigolava M, Chubinidze M, Alimbarashvili E, Vishnepolsky B, Gabrielian A, Rosenthal A, Hurt DE, Tartakovsky M.
  DBAASP v3: database of antimicrobial/cytotoxic activity and structure of peptides as a resource for development of new therapeutics.
  *Nucleic Acids Research* 2021; 49(D1): D288-D297. https://doi.org/10.1093/nar/gkaa991
* Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 2023 (ESM-2).
* Chen T, Guestrin C. XGBoost: a scalable tree boosting system. *KDD* 2016.
* Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. *NeurIPS* 2017 (SHAP).
* RDKit, PubChem (structure lookup), Biopython, scikit-learn, SciPy.

## 10. Author and license

Author: Harshit Narayan Onkar


Code License: MIT License

Copyright (c) 2026 Harshit Narayan Onkar 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Data remain subject to the DBAASP terms of use.
