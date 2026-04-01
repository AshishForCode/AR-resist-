import os
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import dump, load

ANTIBIOTICS = ["ceftriaxone", "ciprofloxacin", "meropenem"]


def build_pipeline() -> Pipeline:
    cat_cols = ["sex", "sample_type", "pathogen"]
    num_cols = ["age", "prior_hospital_days", "previous_antibiotic_courses", "comorbidity_count", "wbc", "crp"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    return pipeline


def train_model(df: pd.DataFrame, output_path: Optional[str] = None) -> Pipeline:
    X = df[["age", "sex", "prior_hospital_days", "previous_antibiotic_courses", "comorbidity_count", "wbc", "crp", "sample_type", "pathogen"]]
    y = df[[f"res_{a}" for a in ANTIBIOTICS]]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dump(pipeline, output_path)

    return pipeline


def load_model(path: str) -> Pipeline:
    return load(path)


def predict_resistance_probabilities(pipeline: Pipeline, patient_data: pd.DataFrame) -> pd.DataFrame:
    # returns DataFrame with predicted prob of resistant for each antibiotic
    proba = pipeline.predict_proba(patient_data)
    # multioutput: list of arrays, each are [n_samples, 2], where index=1 is 'resistant'.
    resistant_probs = np.column_stack([c[:, 1] for c in proba])
    return pd.DataFrame(resistant_probs, columns=[f"prob_res_{a}" for a in ANTIBIOTICS])


def recommend_antibiotic(probs_df: pd.DataFrame) -> List[str]:
    recommendations = []
    for _, row in probs_df.iterrows():
        antibiotic = ANTIBIOTICS[int(np.argmin(row.values))]
        recommendations.append(antibiotic)
    return recommendations
