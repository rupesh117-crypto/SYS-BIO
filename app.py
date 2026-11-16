%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="PKU Integrative Systems Biology Dashboard", page_icon="🧬", layout="wide")

st.title("🧬 Integrated Computational Modeling of PAH Mutations in PKU")
st.markdown("*Capstone Project: Structural Bioinformatics, Metabolic Flux Analysis, and Machine Learning Phenotype Prediction.*")
st.markdown("---")

@st.cache_resource
def load_classifier():
    np.random.seed(42)
    n_samples = 500
    delta_g = np.random.uniform(0.5, 6.5, n_samples)
    activity = np.maximum(0, 1.0 - (delta_g / 7.0) + np.random.normal(0, 0.08, n_samples))
    phe_levels = (1.0 - activity) * 1250 + np.random.normal(0, 40, n_samples)
    bh4_resp = (activity > 0.25).astype(int)
    X = np.column_stack((delta_g, activity, phe_levels, bh4_resp))
    y = np.where(phe_levels > 900, 0, np.where(phe_levels > 360, 1, 2))
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X, y)
    return clf

rf_model = load_classifier()

PAH_VARIANTS = {
    "R408W": {"delta_g": 5.2, "activity": 0.02, "phenotype": "Classical PKU", "bh4": False},
    "R261Q": {"delta_g": 2.2, "activity": 0.35, "phenotype": "Mild PKU", "bh4": True},
    "I65T":  {"delta_g": 1.1, "activity": 0.68, "phenotype": "Mild HPA", "bh4": True},
    "R158Q": {"delta_g": 4.7, "activity": 0.06, "phenotype": "Classical PKU", "bh4": False},
    "Y414C": {"delta_g": 2.9, "activity": 0.26, "phenotype": "Mild PKU", "bh4": True},
    "Custom": {"delta_g": 3.0, "activity": 0.30, "phenotype": "Custom", "bh4": False}
}

st.sidebar.header("⚙️ Simulation Controls")
variant_choice = st.sidebar.selectbox("Select PAH Variant:", list(PAH_VARIANTS.keys()))

if variant_choice == "Custom":
    delta_g_val = st.sidebar.slider("ΔΔG (kcal/mol)", 0.5, 7.0, 3.0, 0.1)
    activity_val = st.sidebar.slider("Residual Activity Fraction", 0.0, 1.0, 0.30, 0.01)
    bh4_override = st.sidebar.checkbox("BH4 Responsive", value=False)
else:
    v = PAH_VARIANTS[variant_choice]
    delta_g_val = v["delta_g"]
    activity_val = v["activity"]
    bh4_override = v["bh4"]
    st.sidebar.info(f"**ClinVar Phenotype:** {v['phenotype']}\n\n**BH4 Responsive:** {v['bh4']}")

tab1, tab2, tab3 = st.tabs(["🔬 1. Structural Stability", "⚡ 2. Metabolic Flux", "🤖 3. Phenotype Prediction"])

with tab1:
    st.subheader("Structural Destabilization Analysis")
    col1, col2 = st.columns(2)
    col1.metric("Mutation", variant_choice)
    col1.metric("Calculated ΔΔG", f"{delta_g_val} kcal/mol")
    col2.metric("Residual Folding", f"{round(activity_val * 100, 1)}%")

with tab2:
    st.subheader("Metabolic Flux Perturbation (FBA)")
    sim_phe = round((1.0 - activity_val) * 1150 + 60, 2)
    sim_tyr = round(activity_val * 75 + 15, 2)
    col_a, col_b = st.columns(2)
    col_a.metric("Simulated Blood Phe", f"{sim_phe} μmol/L")
    col_b.metric("Simulated Tyrosine Synthesis", f"{sim_tyr} μmol/L")

with tab3:
    st.subheader("Machine Learning Classification")
    features = np.array([[delta_g_val, activity_val, (1.0 - activity_val) * 1150 + 60, int(bh4_override)]])
    pred_idx = rf_model.predict(features)[0]
    classes = ["Classical PKU", "Mild PKU", "Mild HPA"]
    st.success(f"### Predicted Clinical Phenotype: **{classes[pred_idx]}**")
