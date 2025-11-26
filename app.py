import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, auc, roc_curve, 
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")

# ----------------------
# 1. 페이지 기본 설정
# ----------------------
st.set_page_config(
    page_title="하이브리드모형 동적 프레임워크",
    page_icon="📊",
    layout="wide"
)

# 전역 상태 관리
if "step" not in st.session_state:
    st.session_state.step = 0
if "data" not in st.session_state:
    st.session_state.data = {"merged": None}
if "preprocess" not in st.session_state:
    st.session_state.preprocess = {"imputer": None, "scaler": None, "encoders": None, "feature_cols": None, "target_col": None}
if "models" not in st.session_state:
    st.session_state.models = {"regression": None, "decision_tree": None, "mixed_weights": {"regression": 0.3, "decision_tree": 0.7}}
if "task" not in st.session_state:
    st.session_state.task = "logit"

# ----------------------
# 2. 사이드바
# ----------------------
st.sidebar.title("📌 작업 흐름")
st.sidebar.divider()
steps = ["초기 설정", "데이터 업로드", "데이터 시각화", "데이터 전처리 & 변수선택", "모델 학습", "모델 예측", "성능 평가"]
for i, step_name in enumerate(steps):
    if st.sidebar.button(step_name, key=f"btn_{i}"):
        st.session_state.step = i

st.sidebar.divider()
st.sidebar.subheader("핵심 설정")
st.session_state.task = st.sidebar.radio("작업 유형", options=["logit", "의사결정나무"], index=0)

if st.session_state.step >= 4:
    st.sidebar.subheader("가중치 조절")
    reg_weight = st.sidebar.slider("회귀 가중치", 0.0, 1.0, 0.3, 0.1)
    st.session_state.models["mixed_weights"]["regression"] = reg_weight
    st.session_state.models["mixed_weights"]["decision_tree"] = 1 - reg_weight

# ----------------------
# 메인 페이지 로직
# ----------------------
st.title("📊 하이브리드모형 동적 프레임워크")
st.divider()

# Step 0 ~ 2 생략 (기존과 동일하게 유지하거나 필요시 복구 가능)
# 편의를 위해 Step 0, 1, 2는 간단히 처리하고 Step 3에 집중합니다.

if st.session_state.step == 0:
    st.info("왼쪽 사이드바에서 '데이터 업로드'를 선택하세요.")

elif st.session_state.step == 1:
    st.subheader("📤 데이터 업로드")
    uploaded_file = st.file_uploader("파일 선택 (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.data["merged"] = df
        st.success(f"업로드 완료: {len(df)} 행")
        st.dataframe(df.head())

elif st.session_state.step == 2:
    st.subheader("📊 데이터 시각화")
    if st.session_state.data["merged"] is None:
        st.warning("데이터를 먼저 업로드하세요.")
    else:
        df = st.session_state.data["merged"]
        st.write("변수 간 상관관계 및 분포를 확인하는 단계입니다.")
        st.dataframe(df.describe())

# ==============================================================================
# [핵심 수정] Step 3: 데이터 전처리 및 지능형 변수 선택 (Stepwise / CART)
# ==============================================================================
elif st.session_state.step == 3:
    st.subheader("🧹 데이터 전처리 & 지능형 변수 선택")
    
    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 먼저 '데이터 업로드' 단계를 완료하세요
