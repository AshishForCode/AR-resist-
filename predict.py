import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from amr_model import load_model, predict_resistance_probabilities, recommend_antibiotic, ANTIBIOTICS


def main():
    parser = argparse.ArgumentParser(description="Predict antibiotic recommendations")
    parser.add_argument("--model", default="models/amr_model.joblib", help="Model path")
    args = parser.parse_args()

    model = load_model(args.model)

    sample = pd.DataFrame([
        {
            "age": 72,
            "sex": "Male",
            "prior_hospital_days": 9,
            "previous_antibiotic_courses": 3,
            "comorbidity_count": 2,
            "wbc": 14.8,
            "crp": 85.0,
            "sample_type": "blood",
            "pathogen": "K_pneumoniae",
        }
    ])

    probs = predict_resistance_probabilities(model, sample)
    rec = recommend_antibiotic(probs)
    print("Input sample:")
    print(sample.to_dict(orient="records")[0])
    print("Predicted resistance probability:")
    print(probs.T)
    print("Recommended antibiotic:", rec[0])
    print("All candidates:", ANTIBIOTICS)


if __name__ == "__main__":
    main()
