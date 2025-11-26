import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings("ignore")

# ----------------------
# 1. 페이지 기본 설정 (파일 최상단에 위치)
# ----------------------
st.set_page_config(
    page_title="하이브리드 분석 프레임워크",
    page_icon="📊",
    layout="wide"
)

# 전역 상태 초기화
if "step" not in st.session_state:
    st.session_state.step = 0
if "data" not in st.session_state:
    st.session_state.data = {"merged": None}
if "preprocess" not in st.session_state:
    st.session_state.preprocess = {}
if "models" not in st.session_state:
    st.session_state.models = {"regression": None, "decision_tree": None, "mixed_weights": {"regression": 0.5, "decision_tree": 0.5}}
if "task" not in st.session_state:
    st.session_state.task = "logit"

# ----------------------
# 2. 사이드바 메뉴
# ----------------------
st.sidebar.title("📌 단계별 진행")
steps = ["1. 데이터 업로드", "2. 데이터 시각화", "3. 데이터 전처리", "4. 모델 학습", "5. 예측 및 결과"]
for i, s in enumerate(steps):
    if st.sidebar.button(s, key=f"nav_{i}"):
        st.session_state.step = i + 1  # 0번은 홈 화면이므로 1부터 시작

st.sidebar.divider()
st.sidebar.subheader("⚙️ 모델 설정")
st.session_state.task = st.sidebar.radio("분석 유형", ["logit (분류)", "regression (회귀)"])

# ----------------------
# 3. 메인 로직
# ----------------------
st.title("📊 하이브리드 분석 프레임워크")

# [Step 0] 홈 화면
if st.session_state.step == 0:
    st.info("왼쪽 사이드바에서 **'1. 데이터 업로드'**를 선택하여 시작하세요.")

# [Step 1] 데이터 업로드
elif st.session_state.step == 1:
    st.subheader("📤 데이터 업로드")
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                # 인코딩 자동 감지 시도
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='cp949')
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.data["merged"] = df
            st.success(f"✅ 업로드 성공: {len(df)}행 {len(df.columns)}열")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"파일 읽기 실패: {e}")

# [Step 2] 시각화
elif st.session_state.step == 2:
    st.subheader("📊 데이터 시각화")
    if st.session_state.data["merged"] is None:
        st.warning("데이터를 먼저 업로드하세요.")
    else:
        df = st.session_state.data["merged"]
        col1, col2 = st.columns(2)
        x_axis = col1.selectbox("X축 선택", df.columns)
        y_axis = col2.selectbox("Y축 선택", ["없음"] + list(df.columns))
        
        if st.button("그래프 그리기"):
            if y_axis != "없음":
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df[x_axis].value_counts())

# [Step 3] 전처리 (안전장치 강화됨)
elif st.session_state.step == 3:
    st.subheader("🧹 데이터 전처리")
    if st.session_state.data["merged"] is None:
        st.warning("데이터를 먼저 업로드하세요.")
    else:
        df = st.session_state.data["merged"].copy()
        
        # 변수 선택
        c1, c2 = st.columns(2)
        target = c1.selectbox("타겟 변수 (Y)", df.columns)
        feats = c2.multiselect("입력 변수 (X)", [c for c in df.columns if c != target])
        
        if st.button("🚀 전처리 실행"):
            if not feats:
                st.error("입력 변수를 하나 이상 선택하세요.")
            else:
                try:
                    # 1. 타겟 결측치 제거
                    df = df.dropna(subset=[target]).reset_index(drop=True)
                    
                    # 2. X, y 분리
                    X = df[feats].copy()
                    y = df[target].copy()
                    
                    # 3. 무한대(Inf)값 제거 (X와 y 모두)
                    X = X.replace([np.inf, -np.inf], np.nan)
                    
                    # y가 수치형일 경우 무한대 체크
                    if np.issubdtype(y.dtype, np.number):
                        y = y.replace([np.inf, -np.inf], np.nan)
                        # y에 NaN이 생기면 해당 행 제거
                        valid_idx = y.notna() & X.notna().all(axis=1) # 엄격한 기준
                        X = X[valid_idx]
                        y = y[valid_idx]
                    
                    # 4. 수치형/범주형 구분
                    num_cols = X.select_dtypes(include=['number']).columns.tolist()
                    cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
                    
                    # 5. 수치형 처리 (Impute + Scale)
                    imputer = SimpleImputer(strategy='mean')
                    scaler = StandardScaler()
                    if num_cols:
                        X[num_cols] = scaler.fit_transform(imputer.fit_transform(X[num_cols]))
                    
                    # 6. 범주형 처리 (최빈값 저장 + LabelEncoding)
                    encoders = {}
                    cat_modes = {}
                    for col in cat_cols:
                        X[col] = X[col].fillna("Unknown").astype(str)
                        cat_modes[col] = X[col].mode()[0] # 최빈값 저장
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col])
                        encoders[col] = le
                        
                    # 7. 타겟 인코딩 (분류 문제일 때)
                    le_target = None
                    if st.session_state.task == "logit" and y.dtype == 'object':
                        le_target = LabelEncoder()
                        y = le_target.fit_transform(y)
                        
                    # 상태 저장
                    st.session_state.preprocess = {
                        "feature_cols": feats, "num_cols": num_cols, "cat_cols": cat_cols,
                        "imputer": imputer, "scaler": scaler, "encoders": encoders, 
                        "cat_modes": cat_modes, "target_encoder": le_target
                    }
                    st.session_state.data["X_processed"] = X
                    st.session_state.data["y_processed"] = y
                    
                    st.success("전처리 완료!")
                    st.dataframe(X.head())
                    
                except Exception as e:
                    st.error(f"전처리 오류: {e}")

# [Step 4] 모델 학습
elif st.session_state.step == 4:
    st.subheader("🚀 모델 학습")
    if "X_processed" not in st.session_state.data:
        st.warning("전처리를 먼저 수행하세요.")
    else:
        X = st.session_state.data["X_processed"]
        y = st.session_state.data["y_processed"]
        
        if st.button("학습 시작"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if st.session_state.task == "logit":
                m1 = LogisticRegression(max_iter=1000)
                m2 = DecisionTreeClassifier(max_depth=5)
            else:
                m1 = LinearRegression()
                m2 = DecisionTreeRegressor(max_depth=5)
                
            m1.fit(X_train, y_train)
            m2.fit(X_train, y_train)
            
            st.session_state.models["regression"] = m1
            st.session_state.models["decision_tree"] = m2
            st.session_state.data["test_set"] = (X_test, y_test)
            st.success("학습 완료!")

# [Step 5] 예측
elif st.session_state.step == 5:
    st.subheader("🎯 예측 및 평가")
    if st.session_state.models["regression"] is None:
        st.warning("모델을 먼저 학습하세요.")
    else:
        # 평가 결과
        if "test_set" in st.session_state.data:
            X_test, y_test = st.session_state.data["test_set"]
            m1 = st.session_state.models["regression"]
            m2 = st.session_state.models["decision_tree"]
            
            pred1 = m1.predict(X_test)
            pred2 = m2.predict(X_test)
            
            if st.session_state.task == "logit":
                score1 = accuracy_score(y_test, pred1)
                score2 = accuracy_score(y_test, pred2)
                st.write(f"### 정확도: 회귀({score1:.2f}), 트리({score2:.2f})")
            else:
                score1 = r2_score(y_test, pred1)
                score2 = r2_score(y_test, pred2)
                st.write(f"### R2 Score: 회귀({score1:.2f}), 트리({score2:.2f})")

        st.divider()
        st.write("#### 🔮 새로운 데이터 예측")
        
        # 입력 폼 생성
        pre = st.session_state.preprocess
        with st.form("pred_form"):
            inputs = {}
            cols = st.columns(3)
            for i, col in enumerate(pre["feature_cols"]):
                with cols[i % 3]:
                    if col in pre["num_cols"]:
                        inputs[col] = st.number_input(col, value=0.0)
                    else:
                        opts = list(pre["encoders"][col].classes_)
                        inputs[col] = st.selectbox(col, opts)
            
            if st.form_submit_button("예측하기"):
                # 입력 데이터프레임 생성
                input_df = pd.DataFrame([inputs])
                
                # 전처리 파이프라인 적용
                X_new = input_df.copy()
                
                # 수치형 변환
                if pre["num_cols"]:
                    X_new[pre["num_cols"]] = pre["scaler"].transform(pre["imputer"].transform(X_new[pre["num_cols"]]))
                
                # 범주형 변환 (Safe Mode)
                for col in pre["cat_cols"]:
                    mode_val = pre["cat_modes"][col]
                    encoder = pre["encoders"][col]
                    classes = set(encoder.classes_)
                    # 모르는 값은 최빈값으로 대체
                    X_new[col] = X_new[col].apply(lambda x: x if x in classes else mode_val)
                    X_new[col] = encoder.transform(X_new[col])
                
                # 예측 수행
                m1 = st.session_state.models["regression"]
                m2 = st.session_state.models["decision_tree"]
                
                if st.session_state.task == "logit":
                    p1 = m1.predict_proba(X_new)[:, 1]
                    p2 = m2.predict_proba(X_new)[:, 1]
                    final_p = (p1 + p2) / 2
                    res = "성공 (1)" if final_p[0] >= 0.5 else "실패 (0)"
                    st.info(f"예측 결과: **{res}** (확률: {final_p[0]:.2%})")
                else:
                    p1 = m1.predict(X_new)
                    p2 = m2.predict(X_new)
                    final_v = (p1 + p2) / 2
                    st.info(f"예측 결과: **{final_v[0]:.2f}**")
