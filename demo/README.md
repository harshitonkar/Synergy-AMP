# Demo (Hugging Face Space)

Live: https://harshitonkar-synergy-amp-baselin.hf.space/

* `app.py`: Gradio app for the context-only baseline (antibiotic + bacterium). Peptide sequence is not an input.
* `requirements.txt`: dependencies used by the Space.
* `export_for_space.py`: a Colab cell that rebuilds the model and lookup files the app needs from `synergy_clean.csv`
  (`baseline_xgb.json`, `baseline_meta.json`, `known_combinations.csv`). Those generated files are not committed (see the root `.gitignore`).

To host your own copy: create a Gradio Space, upload `app.py`, `requirements.txt` and the three generated files.
Research demonstration only; not for clinical or laboratory decisions.
