# predictive-no-show-fvl

Predictive model for estimating patient no-shows in internal medicine outpatient consultations at Fundación Valle del Lili, using machine learning techniques to optimize appointment management and improve healthcare efficiency.

## Repository structure

```
.
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ raw/                  # original dataset
│  ├─ processed/            # preprocessed data
│  ├─ docs/                 # dataset dictionary / docs
│  └─ README.md             # data notes
├─ notebooks/
│  ├─ 01_eda.ipynb           # Exploratory Data Analysis
│  └─ 02_baseline_model.ipynb # baseline ML models
├─ src/
│  ├─ __init__.py
│  ├─ preprocessing.py       # dataset loading + cleaning
│  ├─ models.py              # model builders (baseline)
│  ├─ train.py               # training entrypoint
│  ├─ evaluate.py            # evaluation entrypoint
│  └─ visualize.py           # optional plotting helpers
├─ outputs/
│  ├─ figures/               # plots used in report
│  ├─ metrics/               # metrics/confusion matrices
│  └─ saved_models/          # trained model artifacts
├─ docs/                     # report/presentation (project deliverables)
└─ logs/                     # training logs / experiment notes
```

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Train a baseline model artifact:

```bash
python -m src.train
```

Evaluate a saved model:

```bash
python -m src.evaluate outputs/saved_models/logreg_baseline.joblib
```
