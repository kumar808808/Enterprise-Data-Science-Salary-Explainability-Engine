from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# 1. Resolve exact file paths to target the 'models' folder
BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / 'models' / 'best_salary_model.pkl'
columns_path = BASE_DIR / 'models' / 'model_columns.pkl'

# Fallback in case the files are moved to the root later
if not model_path.exists():
    model_path = BASE_DIR / 'best_salary_model.pkl'
    columns_path = BASE_DIR / 'model_columns.pkl'

# Load model artifacts using the safe paths
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(columns_path, 'rb') as f:
    model_columns = pickle.load(f)

# Initialize SHAP TreeExplainer once
explainer = shap.TreeExplainer(model)

st.set_page_config(page_title="Enterprise Salary Benchmarking Engine", layout="wide")
st.title("💼 Enterprise Data Science Compensation & Explainability Engine")
st.markdown("Predict competitive base compensation and inspect feature attribution using SHAP (Shapley Additive exPlanations).")

# 2. Input Fields
col1, col2 = st.columns(2)

with col1:
    job_simp = st.selectbox(
        "Job Role",
        ['data scientist', 'mle', 'data engineer', 'analyst', 'manager', 'director']
    )
    seniority = st.selectbox(
        "Seniority Level",
        ['na', 'jr', 'senior']
    )
    job_state = st.selectbox(
        "Job State",
        ['CA', 'NY', 'MA', 'TX', 'WA', 'IL', 'Remote']
    )
    rating = st.slider("Company Rating (1.0 - 5.0)", 1.0, 5.0, 3.8, step=0.1)

with col2:
    python_yn = st.checkbox("Requires Python", value=True)
    spark = st.checkbox("Requires Spark", value=False)
    aws = st.checkbox("Requires AWS", value=False)
    excel = st.checkbox("Requires Excel", value=False)
    num_comp = st.number_input("Competitor Count", min_value=0, max_value=10, value=0)

# 3. Prediction & Explainability Logic
if st.button("Generate Salary & Attribution Analysis", type="primary"):
    input_data = {
        'Rating': rating,
        'num_comp': num_comp,
        'hourly': 0,
        'employer_provided': 0,
        'same_state': 1,
        'age': 25,
        'python_yn': int(python_yn),
        'spark': int(spark),
        'aws': int(aws),
        'excel': int(excel),
        'desc_len': 3000,
        'job_simp': job_simp,
        'seniority': seniority,
        'job_state': job_state,
        'Size': '501 to 1000 employees',
        'Type of ownership': 'Company - Private',
        'Industry': 'Enterprise Software & Network Solutions',
        'Sector': 'Information Technology',
        'Revenue': '$100 to $500 million (USD)'
    }

    input_df = pd.DataFrame([input_data])
    encoded_df = pd.get_dummies(input_df)
    final_features = encoded_df.reindex(columns=model_columns, fill_value=0)

    # Inference
    prediction = model.predict(final_features)[0]
    st.success(f"### Estimated Base Salary: **${round(prediction, 2)}K USD / year**")

    # SHAP Attribution
    st.subheader("🔍 Prediction Breakdown (Why is the salary estimated at this value?)")
    shap_values = explainer(final_features)
    
    # Extract feature contributions for this specific prediction
    feature_names = model_columns
    contributions = shap_values.values[0]
    
    contrib_df = pd.DataFrame({
        'Feature': feature_names,
        'Impact ($K)': contributions
    })
    
    # Filter features that have non-zero impact and get top drivers
    active_contribs = contrib_df[contrib_df['Impact ($K)'].abs() > 0.05]
    top_drivers = active_contribs.reindex(
        active_contribs['Impact ($K)'].abs().sort_values(ascending=False).index
    ).head(6)

    # Plot Waterfall / Bar Attribution
    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = ['#2ca02c' if val > 0 else '#d62728' for val in top_drivers['Impact ($K)']]
    ax.barh(top_drivers['Feature'], top_drivers['Impact ($K)'], color=colors)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Impact on Salary ($K USD)")
    ax.set_title("Top Attributes Driving This Prediction Up (+) or Down (-)")
    plt.gca().invert_yaxis()
    st.pyplot(fig)