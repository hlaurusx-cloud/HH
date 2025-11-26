import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, auc, roc_curve, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")

# ----------------------
# 1. 페이지 기본 설정 (단 한 번만 호출해야 함)
# ----------------------
st.set_page_config(
    page_title="하이브리드모형 동적 프레임워크（의사결정나무+회귀분석）",
    page_icon="📊",
    layout="wide"
)

# 전역 상태 관리
if "step" not in st.session_state:
    st.session_state.step = 0  
if "data" not in st.session_state:
    st.session_state.data = {"merged": None}  
if "preprocess" not in st.session_state:
    st.session_state.preprocess = {
        "imputer": None, "scaler": None, "encoders": {}, 
        "feature_cols": None, "target_col": None, "cat_modes": {} 
    }
if "models" not in st.session_state:
    st.session_state.models = {
        "regression": None, "decision_tree": None, 
        "mixed_weights": {"regression": 0.5, "decision_tree": 0.5}
    }
if "task" not in st.session_state:
    st.session_state.task = "logit" 

# ----------------------
# 2. 사이드바
# ----------------------
st.sidebar.title("📌 작업 흐름")
st.sidebar.divider()

steps = ["초기 설정", "데이터 업로드", "데이터 시각화", "데이터 전처리", "모델 학습", "모델 예측", "성능 평가"]
for i, step_name in enumerate(steps):
    if st.sidebar.button(step_name, key=f"btn_{i}"):
        st.session_state.step = i

st.sidebar.divider()
st.sidebar.subheader("핵심 설정")
st.session_state.task = st.sidebar.radio("작업 유형", options=["logit", "의사결정나무"], index=0)

if st.session_state.step >= 4:
    st.sidebar.subheader("하이브리드 가중치")
    reg_weight = st.sidebar.slider(
        "회귀 모델 가중치", 0.0, 1.0, 
        value=st.session_state.models["mixed_weights"]["regression"], step=0.1
    )
    st.session_state.models["mixed_weights"]["regression"] = reg_weight
    st.session_state.models["mixed_weights"]["decision_tree"] = 1 - reg_weight
    st.sidebar.text(f"트리 모델 가중치: {1 - reg_weight:.1f}")

# ----------------------
# 3. 메인 페이지 로직
# ----------------------
st.title("📊 하이브리드모형 동적 배포 프레임워크")

# Step 0: 초기 설정
if st.session_state.step == 0:
    st.subheader("🎉 환영합니다")
    st.markdown("""
    이 프레임워크는 데이터 업로드부터 예측까지의 전 과정을 자동화합니다.
    왼쪽 사이드바에서 **'데이터 업로드'**를 선택하여 시작하세요.
    """)

# Step 1: 데이터 업로드
elif st.session_state.step == 1:
    st.subheader("📤 데이터 업로드")
    tab1, tab2 = st.tabs(["📂 파일 업로드", "💾 기본 데이터"])
    
    def load_csv_safe(file_buffer):
        encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
        for enc in encodings:
            try:
                file_buffer.seek(0)
                df = pd.read_csv(file_buffer, encoding=enc)
                return df, enc
            except:
                continue
        return None, "fail"

    with tab1:
        uploaded_file = st.file_uploader("파일 선택 (CSV/Excel)", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df, enc = load_csv_safe(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                if df is not None:
                    st.session_state.data["merged"] = df.reset_index(drop=True)
                    st.success(f"업로드 성공 ({len(df)}행)")
                else:
                    st.error("파일을 읽을 수 없습니다.")
            except Exception as e:
                st.error(f"에러: {e}")

    with tab2:
        if st.button("기본 데이터 사용"):
            st.info("기본 데이터 기능은 데모 파일이 서버에 있어야 작동합니다.")

    if st.session_state.data["merged"] is not None:
        st.dataframe(st.session_state.data["merged"].head())

# Step 2: 시각화
elif st.session_state.step == 2:
    st.subheader("📊 데이터 시각화")
    if st.session_state.data["merged"] is None:
        st.warning("데이터를 먼저 업로드하세요.")
    else:
        df = st.session_state.data["merged"]
        all_cols = df.columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("X축", options=all_cols)
        with col2:
            y_var = st.selectbox("Y축 (수치형 권장)", options=["없음"] + all_cols)
            
        if y_var != "없음":
            try:
                fig = px.scatter(df, x=x_var, y=y_var, title=f"{x_var} vs {y_var}")
                st.plotly_chart(fig, width='stretch')
            except:
                st.error("해당 변수 조합으로 그래프를 그릴 수 없습니다.")

# Step 3: 데이터 전처리 (개선된 로직)
elif st.session_state.step == 3:
    st.subheader("🧹 데이터 전처리")
    
    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 먼저 '데이터 업로드' 단계를 완료하세요.")
    else:
        df_origin = st.session_state.data["merged"].copy()
        all_cols = df_origin.columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("🎯 타겟 변수 (Y)", options=all_cols)
        
        feature_candidates = [c for c in all_cols if c != target_col]
        with col2:
            selected_features = st.multiselect("📋 입력 변수 (X)", options=feature_candidates, default=feature_candidates[:5])
        
        if st.button("🚀 전처리 실행", type="primary"):
            if not selected_features:
                st.error("입력 변수를 선택해주세요.")
            else:
                with st.spinner("처리 중..."):
                    try:
                        clean_df = df_origin.dropna(subset=[target_col]).reset_index(drop=True)
                        X = clean_df[selected_features].copy()
                        y = clean_df[target_col].copy()

                        # 1. 무한대 처리
                        X = X.replace([np.inf, -np.inf], np.nan)

                        # 2. 타겟 인코딩
                        le_target = None
                        if st.session_state.task == "logit" and y.dtype == 'object':
                            le_target = LabelEncoder()
                            y = pd.Series(le_target.fit_transform(y), index=y.index)

                        num_cols = X.select_dtypes(include=['number']).columns.tolist()
                        cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()

                        imputer = SimpleImputer(strategy='mean')
                        scaler = StandardScaler()
                        encoders = {}
                        cat_modes = {}

                        # 3. 수치형 처리
                        if num_cols:
                            X_imputed = imputer.fit_transform(X[num_cols])
                            X_scaled = scaler.fit_transform(X_imputed)
                            X[num_cols] = pd.DataFrame(X_scaled, columns=num_cols, index=X.index)

                        # 4. 범주형 처리 (안전장치 포함)
                        for col in cat_cols:
                            X[col] = X[col].fillna("Unknown").astype(str)
                            # 최빈값 저장
                            mode_val = X[col].mode()[0]
                            cat_modes[col] = mode_val
                            
                            le = LabelEncoder()
                            trans = le.fit_transform(X[col])
                            X[col] = pd.Series(trans, index=X.index)
                            encoders[col] = le

                        final_features = num_cols + cat_cols
                        X = X[final_features]

                        st.session_state.preprocess.update({
                            "feature_cols": final_features,
                            "imputer": imputer if num_cols else None,
                            "scaler": scaler if num_cols else None,
                            "encoders": encoders,
                            "cat_modes": cat_modes,
                            "num_cols": num_cols,
                            "cat_cols": cat_cols,
                            "target_encoder": le_target
                        })
                        
                        st.session_state.data["X_processed"] = X
                        st.session_state.data["y_processed"] = y
                        st.success("완료!")
                        st.dataframe(X.head())

                    except Exception as e:
                        st.error(f"오류: {e}")

# Step 4: 모델 학습
elif st.session_state.step == 4:
    st.subheader("🚀 모델 학습")
    if "X_processed" in st.session_state.data:
        X = st.session_state.data["X_processed"]
        y = st.session_state.data["y_processed"]
        
        test_size = st.slider("테스트 비율", 0.1, 0.4, 0.2)
        
        if st.button("학습 시작"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            if st.session_state.task == "logit":
                m1 = LogisticRegression(max_iter=1000)
                m2 = DecisionTreeClassifier(max_depth=10)
            else:
                m1 = LinearRegression()
                m2 = DecisionTreeRegressor(max_depth=10)
            
            m1.fit(X_train, y_train)
            m2.fit(X_train, y_train)
            
            st.session_state.models["regression"] = m1
            st.session_state.models["decision_tree"] = m2
            st.session_state.data.update({"X_test": X_test, "y_test": y_test})
            st.success("학습 완료!")
    else:
        st.warning("전처리를 먼저 진행하세요.")

# Step 5: 모델 예측 (Step 3와 로직 일치화)
elif st.session_state.step == 5:
    st.subheader("🎯 모델 예측")
    
    if st.session_state.models["regression"] is None:
        st.warning("모델 학습을 먼저 진행하세요.")
    else:
        def predict_pipeline(input_df):
            pre = st.session_state.preprocess
            X = input_df.copy()
            
            # 1. 컬럼 채우기
            for col in pre["feature_cols"]:
                if col not in X.columns:
                    X[col] = 0
            
            # 2. 수치형
            if pre["num_cols"] and pre["imputer"]:
                for c in pre["num_cols"]:
                    X[c] = pd.to_numeric(X[c], errors='coerce')
                X_num = pre["imputer"].transform(X[pre["num_cols"]])
                X_num = pre["scaler"].transform(X_num)
                X[pre["num_cols"]] = pd.DataFrame(X_num, columns=pre["num_cols"], index=X.index)
            
            # 3. 범주형 (Step 3의 최빈값 활용)
            for col in pre["cat_cols"]:
                encoder = pre["encoders"][col]
                mode_val = pre["cat_modes"][col] # 저장해둔 최빈값
                classes = set(encoder.classes_)
                
                # 없는 값은 최빈값으로 대체하는 함수
                def safe_map(val):
                    s_val = str(val)
                    return s_val if s_val in classes else mode_val
                
                X[col] = X[col].fillna("Unknown").apply(safe_map)
                X[col] = encoder.transform(X[col])
            
            X = X[pre["feature_cols"]]
            
            reg = st.session_state.models["regression"]
            dt = st.session_state.models["decision_tree"]
            w = st.session_state.models["mixed_weights"]
            
            if st.session_state.task == "logit":
                p1 = reg.predict_proba(X)[:, 1]
                p2 = dt.predict_proba(X)[:, 1]
                final_p = w["regression"]*p1 + w["decision_tree"]*p2
                return (final_p >= 0.5).astype(int), final_p
            else:
                p1 = reg.predict(X)
                p2 = dt.predict(X)
                return w["regression"]*p1 + w["decision_tree"]*p2, None

        # 입력 UI
        mode = st.radio("입력 방식", ["직접 입력", "파일 업로드"])
        if mode == "직접 입력":
            with st.form("input"):
                st.markdown("값을 입력하세요 (범주형은 텍스트 입력)")
                input_data = {}
                cols = st.columns(3)
                for i, col in enumerate(st.session_state.preprocess["feature_cols"]):
                    with cols[i % 3]:
                        if col in st.session_state.preprocess["num_cols"]:
                            input_data[col] = st.number_input(col, value=0.0)
                        else:
                            # 기존에 학습된 클래스 보여주기
                            classes = st.session_state.preprocess["encoders"][col].classes_
                            input_data[col] = st.selectbox(col, options=classes)
                
                if st.form_submit_button("예측"):
                    df_in = pd.DataFrame([input_data])
                    pred, prob = predict_pipeline(df_in)
                    st.success(f"결과: {pred[0]}")
                    if prob is not None:
                        st.info(f"확률: {prob[0]:.2%}")
        else:
            up = st.file_uploader("CSV 업로드", type=["csv"])
            if up and st.button("일괄 예측"):
                df_batch = pd.read_csv(up)
                pred, prob = predict_pipeline(df_batch)
                df_batch["Prediction"] = pred
                st.dataframe(df_batch)

# Step 6: 평가
elif st.session_state.step == 6:
    st.subheader("📈 성능 평가")
    if "X_test" in st.session_state.data:
        X_test = st.session_state.data["X_test"]
        y_test = st.session_state.data["y_test"]
        
        # 간단한 평가 로직 재현
        reg = st.session_state.models["regression"]
        dt = st.session_state.models["decision_tree"]
        
        if st.session_state.task == "logit":
            acc1 = accuracy_score(y_test, reg.predict(X_test))
            acc2 = accuracy_score(y_test, dt.predict(X_test))
            st.write(f"회귀 정확도: {acc1:.2f}, 트리 정확도: {acc2:.2f}")
        else:
            r2_1 = r2_score(y_test, reg.predict(X_test))
            r2_2 = r2_score(y_test, dt.predict(X_test))
            st.write(f"회귀 R2: {r2_1:.2f}, 트리 R2: {r2_2:.2f}")
    else:
        st.warning("학습을 먼저 진행하세요.")
