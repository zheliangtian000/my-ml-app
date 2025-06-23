import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load model
with open('ann_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Binary ANN Model Prediction Demo")

# Default values
default_values = {
    'age': 77,
    'sex': 1,
    'pc_prs': 5.457,
    't2dm_prs': 18.9721,
    'smoking_status': 1,
    'bmi': 27.2163,
    'alp': 67.2,
    'alt': 32.71,
    'hba1c': 36.5
}

st.header("Please Fill Features:")
age = st.number_input('Age', value=default_values['age'])
sex = st.selectbox('Sex', [0, 1], index=default_values['sex'])
pc_prs = st.number_input('PC Polygenic Risk Score', value=default_values['pc_prs'])
t2dm_prs = st.number_input('T2DM PRS', value=default_values['t2dm_prs'])
smoking_status = st.selectbox('Smoking Status', [0, 1, 2], index=default_values['smoking_status'])
bmi = st.number_input('BMI', value=default_values['bmi'])
alp = st.number_input('Alkaline Phosphatase', value=default_values['alp'])
alt = st.number_input('ALT', value=default_values['alt'])
hba1c = st.number_input('HbA1c', value=default_values['hba1c'])

input_df = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'pc_prs': pc_prs,
    't2dm_prs': t2dm_prs,
    'smoking_status': smoking_status,
    'bmi': bmi,
    'alp': alp,
    'alt': alt,
    'hba1c': hba1c
}])

if st.button('Predict'):
    # Predict probability
    if hasattr(model, 'named_steps'):
        scaler = model.named_steps.get('scaler')
        clf = model.named_steps.get('ann')
        input_scaled = scaler.transform(input_df)
        pred_prob = clf.predict_proba(input_scaled)[0, 1]
    else:
        pred_prob = model.predict_proba(input_df)[0, 1]

    st.markdown(f"### Probability of Class 1: **{pred_prob:.3%}**")

    # Vertical bar plot
    fig, ax = plt.subplots(figsize=(1.5, 5))
    ax.bar([0], [pred_prob], width=0.5, color='tomato')
    ax.set_ylim(0, 0.5)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylabel("Probability")
    ax.set_xticks([0])
    ax.set_xticklabels(["Patient"])
    ax.set_title("Predicted Probability of outcome")
    # Show value on bar
    ax.text(0, pred_prob + 0.01, f"{pred_prob:.3%}", ha='center', va='bottom', fontsize=13, color='black')
    # Remove top/right/bottom spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.grid(False)
    plt.tight_layout()
    st.pyplot(fig)
