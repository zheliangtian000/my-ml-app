import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# 加载模型
with open('ann_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("二分类人工神经网络（ANN）模型预测演示")

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

st.header("请填写特征：")
age = st.number_input('年龄 (age)', value=default_values['age'])
sex = st.selectbox('性别 (sex)', [0, 1], index=default_values['sex'])  # 0-女 1-男
pc_prs = st.number_input('PC多基因风险评分 (pc_prs)', value=default_values['pc_prs'])
t2dm_prs = st.number_input('2型糖尿病PRS (t2dm_prs)', value=default_values['t2dm_prs'])
smoking_status = st.selectbox('吸烟状态 (smoking_status)', [0, 1, 2], index=default_values['smoking_status']) # 0-从不，1-已戒，2-吸烟
bmi = st.number_input('BMI (bmi)', value=default_values['bmi'])
alp = st.number_input('碱性磷酸酶 (alp)', value=default_values['alp'])
alt = st.number_input('丙氨酸氨基转移酶 (alt)', value=default_values['alt'])
hba1c = st.number_input('糖化血红蛋白 (hba1c)', value=default_values['hba1c'])

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
    # 判断是否为pipeline模型（含scaler），适配不同保存方式
    if hasattr(model, 'named_steps'):
        scaler = model.named_steps.get('scaler')
        clf = model.named_steps.get('ann')
        input_scaled = scaler.transform(input_df)
        pred_prob = clf.predict_proba(input_scaled)[0, 1]
    else:
        pred_prob = model.predict_proba(input_df)[0, 1]

    st.markdown(f"### 预测为1的概率为: **{pred_prob:.3%}**")

    # 概率柱状图
    st.markdown("#### 预测概率可视化：")
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.barh(['概率'], [pred_prob], color='#4f8dfd')
    ax.set_xlim(0, 1)
    ax.set_xlabel("概率")
    ax.text(pred_prob, 0, f"{pred_prob:.3%}", va='center', ha='left', fontsize=13, color='black')
    ax.set_yticks([])
    ax.set_title("预测为1的概率（直观显示）")
    plt.tight_layout()
    st.pyplot(fig)
