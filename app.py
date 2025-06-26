import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import os

# ----- 页面设置 -----
st.set_page_config(
    page_title="ANN Model Prediction",
    page_icon=":brain:",
    layout="centered"
)

# ----- 样式 -----
st.markdown("""
    <style>
    .big-title {font-size: 2.4em !important; font-weight: bold; line-height: 1.1;}
    .section-header {font-size: 1.35em !important; font-weight: bold; margin-bottom: 0.5em;}
    .stButton>button {width: 100%; border: 1px solid tomato;}
    .footer2025 {position: fixed; right: 24px; bottom: 12px; color: #A9A9A9; font-size: 16px; z-index: 100; font-weight: 600;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">ANN Model Prediction</div>', unsafe_allow_html=True)
st.markdown("---")

# ----- 默认值 -----
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

# ----- 特征输入 -----
st.markdown('<div class="section-header">Please Fill Features:</div>', unsafe_allow_html=True)
age = st.number_input('Age', value=default_values['age'], min_value=0, max_value=120)
sex = st.selectbox('Sex (0: Female, 1: Male)', [0, 1], index=[0,1].index(default_values['sex']))
pc_prs = st.number_input('PC Polygenic Risk Score', value=default_values['pc_prs'])
t2dm_prs = st.number_input('T2DM PRS', value=default_values['t2dm_prs'])
smoking_status = st.selectbox('Smoking Status (0:Never, 1:Former, 2:Current)', [0, 1, 2], index=[0,1,2].index(default_values['smoking_status']))
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

# ----- 加载模型 -----
@st.cache_resource
def load_model():
    model = joblib.load('ann_model.joblib')
    return model

model = load_model()

# ----- 预测 -----
predict_clicked = st.button('Predict')
if predict_clicked:
    try:
        pred_prob = model.predict_proba(input_df)[0, 1]
        threshold = 0.33
        pred_label = int(pred_prob >= threshold)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        pred_prob = None

    if pred_prob is not None:
        st.markdown(
            f'<div style="font-size:1.15em; color:#ed4c2c; font-weight:bold; margin-bottom:10px">'
            f'Predicted Probability of outcome: {pred_prob:.3%}'
            '</div>',
            unsafe_allow_html=True
        )
        st.progress(pred_prob, text=f"{pred_prob:.3%}")

        st.info(f"Predicted Label: **{pred_label}** (0=Negative, 1=Positive, threshold=0.33)")

        # ----- SHAP 力图 -----
        try:
            st.markdown('<div class="section-header">Model Interpretation (SHAP Force Plot):</div>', unsafe_allow_html=True)

            # 自动判断pipeline/scaler
            if hasattr(model, 'named_steps') and 'mlpclassifier' in model.named_steps:
                clf = model.named_steps['mlpclassifier']
                # scaler = model.named_steps['standardscaler'] # 不用手动
                explainer = shap.KernelExplainer(clf.predict_proba, shap.kmeans(input_df, 1))
                X_trans = model.named_steps['standardscaler'].transform(input_df)
            elif hasattr(model, 'predict_proba'):
                clf = model
                explainer = shap.KernelExplainer(clf.predict_proba, shap.kmeans(input_df, 1))
                X_trans = input_df
            else:
                raise Exception("Model type not supported for SHAP.")

            shap_values = explainer.shap_values(X_trans)
            # 单样本force_plot
            fig = shap.force_plot(
                explainer.expected_value[1], shap_values[1][0], X_trans[0],
                matplotlib=True, show=False, feature_names=input_df.columns
            )
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.warning(f"SHAP force plot failed: {e}")

# 可选：显示输入数据
with st.expander("Show Input Data"):
    st.write(input_df)

# ----- 右下角小标 -----
st.markdown('<div class="footer2025">2025 Binary ANN Model</div>', unsafe_allow_html=True)
