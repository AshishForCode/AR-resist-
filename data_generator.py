import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from amr_model import ANTIBIOTICS


def generate_synthetic_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.normal(58, 19, n_samples).clip(18, 95).astype(int)
    sex = rng.choice(["Male", "Female"], size=n_samples, p=[0.52, 0.48])
    prior_hospital_days = rng.poisson(3, n_samples)
    previous_abx = rng.poisson(1, n_samples)
    comorbidity_count = rng.integers(0, 4, n_samples)
    wbc = rng.normal(12.5, 4.1, n_samples).clip(3, 30)
    crp = rng.exponential(7, n_samples).clip(0.2, 300)
    sample_type = rng.choice(["blood", "urine", "wound", "respiratory"], n_samples)
    pathogen = rng.choice(["E_coli", "S_aureus", "K_pneumoniae", "P_aeruginosa", "Enterococcus"], n_samples)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "prior_hospital_days": prior_hospital_days,
        "previous_antibiotic_courses": previous_abx,
        "comorbidity_count": comorbidity_count,
        "wbc": wbc,
        "crp": crp,
        "sample_type": sample_type,
        "pathogen": pathogen,
    })

    # Create antibiotic resistance labels with some rules
    for abx in ANTIBIOTICS:
        base_rate = {"ceftriaxone": 0.25, "ciprofloxacin": 0.2, "meropenem": 0.12}[abx]
        risk = 0.01 * (df["age"] > 65) + 0.02 * (df["prior_hospital_days"] > 7) + 0.03 * (df["comorbidity_count"] > 2)
        risk += 0.05 * (df["pathogen"] == "P_aeruginosa")
        risk += 0.05 * (df["pathogen"] == "K_pneumoniae")

        if abx == "meropenem":
            risk += 0.1 * (df["sample_type"] == "blood")

        prob = np.clip(base_rate + risk, 0, 0.95)
        df[f"res_{abx}"] = (rng.random(n_samples) < prob).astype(int)

    return df
