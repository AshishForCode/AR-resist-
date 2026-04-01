import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.amr_model import load_model, predict_resistance_probabilities, recommend_antibiotic, ANTIBIOTICS
from src.data_generator import generate_synthetic_data

MODEL_PATH = "models/amr_model.joblib"


def setup_model():
    try:
        model = load_model(MODEL_PATH)
        st.success("Loaded trained model from {}".format(MODEL_PATH))
    except Exception:
        st.warning("Model not found. Train first (python src/train.py). Using on-the-fly model from synthetic data.")
        df = generate_synthetic_data(2000)
        model = st.session_state.get("amr_model")
        if model is None:
            from src.amr_model import train_model
            model = train_model(df)
            st.session_state["amr_model"] = model
        st.session_state["synthetic_data"] = df

    return model


def build_user_input() -> pd.DataFrame:
    st.sidebar.header("Patient features")
    age = st.sidebar.slider("Age", 18, 95, 62)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    prior_hospital_days = st.sidebar.number_input("Prior hospital days", 0, 60, 2)
    previous_abx = st.sidebar.number_input("Previous antibiotic courses", 0, 10, 1)
    comorbidity_count = st.sidebar.number_input("Comorbidity count", 0, 5, 1)
    wbc = st.sidebar.number_input("WBC", 3.0, 30.0, 11.5, 0.1)
    crp = st.sidebar.number_input("CRP", 0.0, 300.0, 15.0, 0.1)
    sample_type = st.sidebar.selectbox("Sample type", ["blood", "urine", "wound", "respiratory"])
    pathogen = st.sidebar.selectbox("Pathogen", ["E_coli", "S_aureus", "K_pneumoniae", "P_aeruginosa", "Enterococcus"])

    patient = pd.DataFrame([
        {
            "age": age,
            "sex": sex,
            "prior_hospital_days": prior_hospital_days,
            "previous_antibiotic_courses": previous_abx,
            "comorbidity_count": comorbidity_count,
            "wbc": wbc,
            "crp": crp,
            "sample_type": sample_type,
            "pathogen": pathogen,
        }
    ])

    return patient


def main():
    st.set_page_config(page_title="AMR Antibiotic Recommendation Dashboard", layout="wide")
    st.title("AI-Based Antibiotic Resistance Prediction Dashboard")

    model = setup_model()

    st.markdown("### Real-time patient prediction")
    patient = build_user_input()
    st.table(patient)

    if st.button("Predict resistance and recommend"):
        probs = predict_resistance_probabilities(model, patient)
        rec = recommend_antibiotic(probs)[0]

        st.subheader("Predicted resistance probabilities")
        st.dataframe(probs.T)

        st.success(f"Recommended antibiotic: {rec}")

        chart_data = probs.T
        chart_data.columns = ["Probability"]
        st.bar_chart(chart_data)

    st.markdown("---")

    if st.session_state.get("synthetic_data") is not None:
        st.markdown("### Historical simulated dataset (single view)")
        df = st.session_state.get("synthetic_data")
        st.dataframe(df.sample(min(25, len(df))))

        st.markdown("### Resistance trends")
        trend = df[[f"res_{a}" for a in ANTIBIOTICS]].mean().rename(lambda x: x.replace("res_", ""))
        st.bar_chart(trend)


if __name__ == "__main__":
    main()
