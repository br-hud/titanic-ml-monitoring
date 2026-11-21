# Titanic ML Monitoring Foundation

This repository is the reproducible training foundation for a larger machine learning monitoring project. It trains a Titanic survival classifier from four inputs and records the evidence needed by later API, metrics, dashboard, alerting, and drift work.

## What it demonstrates

- A dataset fetched by exact OpenML name and version
- An intentional raw-data failure that exposes the need for preprocessing
- Numeric missing-value imputation and categorical one-hot encoding
- A preprocessing and Random Forest pipeline trained as one object
- A repeatable train-test split and recorded evaluation metadata
- Reference statistics for transformed training features
- Dependency pinning and safe Git exclusions

## Dataset

The project uses OpenML's `titanic` dataset, version `1`, as documented in scikit-learn's mixed-type preprocessing example:

https://www.openml.org/d/40945

The model uses `pclass`, `sex`, `age`, and `fare`. This is an educational historical dataset, not a production decision dataset.

## Windows quick start

Open PowerShell in the repository folder.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py --preview
python train_model.py --raw-check
python train_model.py
```

The first data fetch requires internet access. OpenML data is then cached by scikit-learn.

## Generated artifacts

- `artifacts/model.pkl`: fitted preprocessing and model pipeline, kept local
- `artifacts/metadata.json`: dataset identity, feature list, split configuration, versions, accuracy, and confusion matrix
- `artifacts/reference_stats.json`: mean and population standard deviation for each transformed training feature

## Security note

The model uses Python pickle protocol 5. Loading a pickle can execute arbitrary code. Only load `model.pkl` when you created it yourself or fully trust its source. The model file is excluded from Git and can be regenerated with `python train_model.py`.

## Current scope

This repository trains and records the baseline. It does not yet expose predictions through an API or collect production metrics.

## Planned continuation

The next project will extend this same repository with monitored inference behavior while preserving the training pipeline and evidence created here.