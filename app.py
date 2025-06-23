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

st.header("请填写特征：")
age = st.number_input('年龄 (age)', value=DEFAULTS['age'])
sex = st.selectbox('性别 (sex)', [0, 1], index=DEFAULTS['sex'])
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
    # Pipeline标准化（如果有的话）
    if hasattr(model, 'named_steps'):
        scaler = model.named_steps.get('scaler')
        clf = model.named_steps.get('ann')
        input_scaled = scaler.transform(input_df)
        predict_proba_func = lambda x: clf.predict_proba(scaler.transform(x))
    else:
        input_scaled = input_df.values
        predict_proba_func = model.predict_proba

    pred_prob = predict_proba_func(input_df)[0, 1]
    st.markdown(f"### 预测为1的概率为: **{pred_prob:.3%}**")

    st.markdown("### 单样本SHAP力图解释")
    try:
        # 用训练集一部分做背景集更稳定（实际建议你存一份训练集X_train样本，否则可以用输入自身）
        background = input_df  # 或者 X_train.sample(50, random_state=42)
        explainer = shap.KernelExplainer(predict_proba_func, background)
        shap_values = explainer.shap_values(input_df, nsamples=100)
        # 兼容二分类只有一组值
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)) and len(np.array(expected).shape) == 0:
            expected = expected
        elif isinstance(expected, (list, np.ndarray)) and len(expected) == 2:
            expected = expected[1]
            shap_v = shap_values[1][0]
        else:
            expected = expected[0] if isinstance(expected, (list, np.ndarray)) else expected
            shap_v = shap_values[0]
        # force_plot参数兼容
        fig = shap.force_plot(expected, shap_v, input_df.iloc[0], matplotlib=True, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP解释出错：{e}")
