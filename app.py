import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model (should be sklearn Pipeline: scaler + MLPClassifier)
with open('ann_model.pkl', 'rb') as f:
    model = pickle.load(f)

# --- Style improvements ---
st.markdown("""
    <style>
    .big-title {font-size: 2.4em !important; font-weight: bold; line-height: 1.1;}
    .section-header {font-size: 1.35em !important; font-weight: bold; margin-bottom: 0.5em;}
    .stButton>button {width: 100%; border: 1px solid tomato;}
    </style>
""", unsafe_allow_html=True)

# Layout: Title on top, two columns below
st.markdown('<div class="big-title">Binary ANN Model Prediction Demo</div>', unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1.2, 1.5], gap="large")

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

with col1:
    st.markdown('<div class="section-header">Please Fill Features:</div>', unsafe_allow_html=True)
    age = st.number_input('Age', value=default_values['age'])
    sex = st.selectbox('Sex', [0, 1], index=[0,1].index(default_values['sex']) if default_values['sex'] in [0,1] else 0)
    pc_prs = st.number_input('PC Polygenic Risk Score', value=default_values['pc_prs'])
    t2dm_prs = st.number_input('T2DM PRS', value=default_values['t2dm_prs'])
    smoking_status = st.selectbox('Smoking Status', [0, 1, 2], index=[0,1,2].index(default_values['smoking_status']) if default_values['smoking_status'] in [0,1,2] else 0)
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

    predict_clicked = st.button('Predict')

with col2:
    if 'predict_clicked' not in st.session_state:
        st.session_state.predict_clicked = False

    if predict_clicked:
        st.session_state.predict_clicked = True

    if st.session_state.predict_clicked:
        try:
            # 1. Pipeline 直接用model.predict_proba
            pred_prob = model.predict_proba(input_df)[0, 1]
            # 2. 若模型不是pipeline,则兼容原始MLP
            # elif hasattr(model, 'named_steps') and 'scaler' in model.named_steps and 'ann' in model.named_steps:
            #     scaler = model.named_steps['scaler']
            #     clf = model.named_steps['ann']
            #     input_scaled = scaler.transform(input_df)
            #     pred_prob = clf.predict_proba(input_scaled)[0, 1]
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            pred_prob = None

        if pred_prob is not None:
            st.markdown(
                f'<div style="font-size:1.15em; color:#ed4c2c; font-weight:bold; margin-bottom:12px">'
                f'Probability of Class 1: {pred_prob:.3%}'
                '</div>',
                unsafe_allow_html=True
            )

            # Plot - vertical bar
            fig, ax = plt.subplots(figsize=(2, 5))
            ax.bar([0], [pred_prob], width=0.5, color='tomato')
            ax.set_ylim(0, 1)
            ax.set_xlim(-0.8, 0.8)
            ax.set_ylabel("Probability", fontsize=14)
            ax.set_xticks([0])
            ax.set_xticklabels(["Patient"], fontsize=14)
            ax.set_title("Predicted Probability of outcome", fontsize=15, pad=16)
            ax.text(0, pred_prob + 0.015, f"{pred_prob:.3%}", ha='center', va='bottom', fontsize=16, color='black')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.grid(axis='y', linestyle='--', linewidth=0.3, alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)

# Footer style if you want
st.markdown("---")
st.markdown(
    "<div style='text-align:right; color:gray; font-size: 0.9em;'>"
    "© 2025 Binary ANN Model Demo"
    "</div>", unsafe_allow_html=True
)
