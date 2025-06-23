import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# 加载模型
with open('ann_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Binary ANN Model Prediction Demo")

# 默认值
default_values = {
    'age': 41,
    'sex': 1,
    'pc_prs': 3.69928,
    't2dm_prs': 20.9937,
    'smoking_status': 1,
    'bmi': 23.7508,
    'alp': 65.2,
    'alt': 26.62,
    'hba1c': 89.8
}

st.header("Please Fill Features:")
age = st.number_input('Age', value=default_values['age'])
sex = st.selectbox('Sex', [0, 1], index=default_values['sex'])  # 0-Female 1-Male
pc_prs = st.number_input('PC Polygenic Risk Score', value=default_values['pc_prs'])
t2dm_prs = st.number_input('T2DM PRS', value=default_values['t2dm_prs'])
smoking_status = st.selectbox('Smoking Status', [0, 1, 2], index=default_values['smoking_status']) # 0-Never, 1-Quit, 2-Smoking
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
    # 预测概率
    # 判断是否为pipeline模型（含scaler），适配不同保存方式
    if hasattr(model, 'named_steps'):
        scaler = model.named_steps.get('scaler')
        clf = model.named_steps.get('ann')
        input_scaled = scaler.transform(input_df)
        pred_prob = clf.predict_proba(input_scaled)[0, 1]
    else:
        pred_prob = model.predict_proba(input_df)[0, 1]

    st.markdown(f"### Probability of Class 1: **{pred_prob:.3%}**")

    # 概率柱状图
    st.markdown("#### Probability Visualization:")
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.barh(['Probability'], [pred_prob], color='#4f8dfd')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.text(pred_prob, 0, f"{pred_prob:.3%}", va='center', ha='left', fontsize=13, color='black')
    ax.set_yticks([])
    ax.set_title("Predicted Probability of Class 1")
    plt.tight_layout()
    st.pyplot(fig)
