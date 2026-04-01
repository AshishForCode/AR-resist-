# AI-Based Antibiotic Resistance Prediction Tool

Prototype for Problem Statement 2 (AMR prediction + recommendations)

## Components

- `src/amr_model.py`: model pipeline and training/prediction helpers
- `src/train.py`: CLI for dataset creation, training, and report
- `src/predict.py`: CLI for model inference on example patients
- `app/dashboard.py`: Streamlit dashboard with input forms, trend charts, recommendation engine
- `resources/` (optional): place real antibiogram CSVs here as `antibiogram.csv`

## Quickstart

1. Create virtual env

   `python -m venv .venv; .venv\Scripts\activate`

2. Install

   `pip install -r requirements.txt`

3. Train model

   `python src/train.py --output models/amr_model.joblib`

4. Run dashboard

   `streamlit run app/dashboard.py`

## Architecture

1. Synthetic or hospital patient data ingestion
2. Preprocessing: categorical encoding, numeric scaling
3. Model: MultiOutputClassifier(RandomForestClassifier)
4. Output: resistance probability per antibiotic
5. Bot: recommends antibiotic with lowest predicted resistance

## Notes
- Replace synthetic data generator with real antibiogram CSVs in production.
- Include clinical oversight in final validation.
