import argparse
import os
import sys

import pandas as pd
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(__file__))
from amr_model import train_model, load_model, ANTIBIOTICS
from data_generator import generate_synthetic_data


def main():
    parser = argparse.ArgumentParser(description="Train antibiotic resistance prediction model")
    parser.add_argument("--output", default="models/amr_model.joblib", help="Output model path")
    parser.add_argument("--samples", type=int, default=2500, help="Number of synthetic samples to generate")
    parser.add_argument("--save-data", default="data/synthetic_amr_data.csv", help="Optional synthetic dataset save path")
    args = parser.parse_args()

    df = generate_synthetic_data(args.samples)
    os.makedirs(os.path.dirname(args.save_data), exist_ok=True)
    df.to_csv(args.save_data, index=False)

    pipeline = train_model(df, output_path=args.output)

    # Print coarse reporting
    X = df[["age", "sex", "prior_hospital_days", "previous_antibiotic_courses", "comorbidity_count", "wbc", "crp", "sample_type", "pathogen"]]
    y = df[[f"res_{a}" for a in ANTIBIOTICS]]
    y_pred = pipeline.predict(X)

    for idx, abx in enumerate(ANTIBIOTICS):
        print("\n---", abx, "---")
        print(classification_report(y.iloc[:, idx], y_pred[:, idx], zero_division=0))

    # load + verify
    loaded = load_model(args.output)
    print("Model saved and reloaded from", args.output)


if __name__ == "__main__":
    main()
