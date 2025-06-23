import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt

# 加载模型
with open('ann_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("二分类ANN模型预测与SHAP力图解释")

# 默认值
DEFAULTS = dict(
    age=41,
    sex=1,
    pc_prs=3.69928,
    t2dm_prs=20.9937,
    smoking_status=1,
    bmi=23.7508,
    alp=65.2,
    alt=26.62,
    hba1c=89.8
)

# 用户输入
st.header("请填写特征：")
age = st.number_input('年龄 (age)', value=DEFAULTS['age'])
sex = st.selectbox('性别 (sex)', [0, 1], index=DEFAULTS['sex'])  # 0-女 1-男
pc_prs = st.number_input('PC多基因风险评分 (pc_prs)', value=DEFAULTS['pc_prs'])
t2dm_prs = st.number_input('2型糖尿病PRS (t2dm_prs)', value=DEFAULTS['t2dm_prs'])
smoking_status = st.selectbox('吸烟状态 (smoking_status)', [0, 1, 2], index=DEFAULTS['smoking_status'])
bmi = st.number_input('BMI (bmi)', value=DEFAULTS['bmi'])
alp = st.number_input('碱性磷酸酶 (alp)', value=DEFAULTS['alp'])
alt = st.number_input('丙氨酸氨基转移酶 (alt)', value=DEFAULTS['alt'])
hba1c = st.number_input('糖化血红蛋白 (hba1c)', value=DEFAULTS['hba1c'])

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
    # 自动判断Pipeline并标准化
    if hasattr(model, 'named_steps'):
        scaler = model.named_steps.get('scaler')
        clf = model.named_steps.get('ann')
        input_scaled = scaler.transform(input_df)
        predict_proba_func = lambda x: clf.predict_proba(scaler.transform(x))
    else:
        input_scaled = input_df.values
        predict_proba_func = model.predict_proba

    # 预测概率
    pred_prob = predict_proba_func(input_df)[0, 1]
    st.markdown(f"### 预测为1的概率为: **{pred_prob:.3%}**")

    # SHAP力图
    st.markdown("### 单样本SHAP力图解释")
    try:
        explainer = shap.KernelExplainer(predict_proba_func, input_df)
        shap_values = explainer.shap_values(input_df, nsamples=100)
        fig = shap.force_plot(explainer.expected_value[1], shap_values[1][0], input_df.iloc[0], matplotlib=True, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP解释出错：{e}")
