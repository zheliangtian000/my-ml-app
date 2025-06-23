import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt

# 加载模型
with open('ann_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("二分类模型预测与解释力图演示")

# 用户输入
st.header("请填写特征：")
age = st.number_input('年龄 (age)', value=40)
sex = st.selectbox('性别 (sex)', [0, 1])  # 0-女 1-男，根据你的编码
pc_prs = st.number_input('PC多基因风险评分 (pc_prs)', value=0.0)
t2dm_prs = st.number_input('2型糖尿病PRS (t2dm_prs)', value=0.0)
smoking_status = st.selectbox('吸烟状态 (smoking_status)', [0, 1, 2]) # 按你的标签编码来（例：0-从不，1-已戒，2-吸烟）
bmi = st.number_input('BMI (bmi)', value=25.0)
alp = st.number_input('碱性磷酸酶 (alp)', value=60.0)
alt = st.number_input('丙氨酸氨基转移酶 (alt)', value=15.0)
hba1c = st.number_input('糖化血红蛋白 (hba1c)', value=5.5)

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

if st.button('预测'):
    # 预测概率
    pred_prob = model.predict_proba(input_df)[0, 1]
    st.markdown(f"### 预测为1的概率为: **{pred_prob:.3%}**")
    
    # SHAP解释（使用TreeExplainer或KernelExplainer）
    st.markdown("### 单样本SHAP力图解释")
    try:
        explainer = shap.Explainer(model, input_df)
        shap_values = explainer(input_df)
        # 绘制力图
        fig = shap.plots.force(shap_values[0], matplotlib=True, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP解释出错：{e}")

