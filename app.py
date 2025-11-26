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

# ----------------------
# 1. 页面基本设置 (必须是第一个 Streamlit 命令)
# ----------------------
st.set_page_config(
    page_title="하이브리드모형 동적 프레임워크",
    page_icon="📊",
    layout="wide"
)

warnings.filterwarnings("ignore")

# ----------------------
# 全局状态管理 (Session State)
# ----------------------
if "step" not in st.session_state:
    st.session_state.step = 0  # 0:初始画面 1:上传 2:可视化 3:预处理 4:训练 5:预测 6:评估
if "data" not in st.session_state:
    st.session_state.data = {"merged": None}
if "preprocess" not in st.session_state:
    st.session_state.preprocess = {
        "imputer": None, "scaler": None, "encoders": None, 
        "feature_cols": None, "target_col": None
    }
if "models" not in st.session_state:
    st.session_state.models = {
        "regression": None, "decision_tree": None, 
        "mixed_weights": {"regression": 0.3, "decision_tree": 0.7}
    }
if "task" not in st.session_state:
    st.session_state.task = "logit"

# ----------------------
# 2. 侧边栏：导航 + 核心设置
# ----------------------
st.sidebar.title("📌 하이브리드모형 작업 흐름")
st.sidebar.divider()

# 导航按钮
steps = ["초기 설정", "데이터 업로드", "데이터 시각화", "데이터 전처리", "모델 학습", "모델 예측", "성능 평가"]
for i, step_name in enumerate(steps):
    if st.sidebar.button(step_name, key=f"btn_{i}"):
        st.session_state.step = i

# 核心设置
st.sidebar.divider()
st.sidebar.subheader("핵심 설정")

current_idx = 0 if st.session_state.task == "logit" else 1
new_task = st.sidebar.radio("작업 유형", options=["logit", "의사결정나무"], index=current_idx)
st.session_state.task = new_task

if st.session_state.step >= 4:
    st.sidebar.subheader("하이브리드모형 가중치")
    reg_weight = st.sidebar.slider(
        "회귀 분석 가중치（해석력 강함）",
        min_value=0.0, max_value=1.0, 
        value=st.session_state.models["mixed_weights"]["regression"], 
        step=0.1
    )
    st.session_state.models["mixed_weights"]["regression"] = reg_weight
    st.session_state.models["mixed_weights"]["decision_tree"] = 1.0 - reg_weight
    st.sidebar.text(f"의사결정나무 가중치（정확도 높음）：{1.0 - reg_weight:.1f}")

# ----------------------
# 3. 主页面内容
# ----------------------
st.title("📊 하이브리드모형 동적 배포 프레임워크")
st.markdown("**단일 원본 데이터 파일 업로드 후，시각화→전처리→학습→예측 전과정을 한 번에 완성**")
st.divider()

# ==============================================================================
# 逻辑流程
# ==============================================================================

# ----------------------
#  步骤 0：初始设置
# ----------------------
if st.session_state.step == 0:
    st.subheader("🎉 환영합니다")
    st.markdown("""
    본 프레임워크는 **데이터 수령 후 직접 업로드하여 사용**할 수 있으며，사전 전처리나 모델 학습이 필요 없습니다.
    
    1. **데이터 업로드**：단일 원본 파일（CSV/Parquet/Excel）을 업로드
    2. **데이터 시각화**：변수 탐색
    3. **데이터 전처리**：결측값 처리 및 인코딩
    4. **모델 학습**：「회귀+의사결정나무」
    5. **모델 예측**：단일/일괄 예측
    6. **성능 평가**：모델 비교
    
    ### 왼쪽 사이드바에서 **「데이터 업로드」**를 선택하여 시작하세요!
    """)

# ----------------------
#  步骤 1：数据上传
# ----------------------
elif st.session_state.step == 1:
    st.subheader("📤 데이터 업로드")
    
    tab1, tab2 = st.tabs(["📂 내 파일 업로드", "💾 서버 기본 데이터 사용"])
    
    def load_csv_safe(file_buffer):
        encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig', 'latin1']
        for enc in encodings:
            try:
                file_buffer.seek(0)
                df = pd.read_csv(file_buffer, encoding=enc)
                return df, enc
            except Exception:
                continue
        return None, "fail"

    with tab1:
        st.markdown("지원 형식：CSV、Parquet、Excel")
        uploaded_file = st.file_uploader("파일 선택", type=["csv", "parquet", "xlsx", "xls"])
        
        if uploaded_file:
            try:
                df = None
                if uploaded_file.name.endswith('.csv'):
                    df, enc_used = load_csv_safe(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                if df is not None:
                    st.session_state.data["merged"] = df.reset_index(drop=True)
                    st.success(f"✅ 업로드 성공! ({len(df):,} 행)")
                else:
                    st.error("❌ 파일 읽기 실패 (인코딩 확인 필요)")
            except Exception as e:
                st.error(f"❌ 오류: {e}")
    
    with tab2:
        DEFAULT_FILE = "combined_loan_data.csv"
        if st.button("기본 데이터 불러오기"):
            if os.path.exists(DEFAULT_FILE):
                with open(DEFAULT_FILE, 'rb') as f:
                    df, _ = load_csv_safe(f)
                if df is not None:
                    st.session_state.data["merged"] = df.reset_index(drop=True)
                    st.success("✅ 기본 데이터 로드 성공!")
                    st.rerun()
            else:
                st.error("⚠️ 서버에 기본 파일이 없습니다.")

    if st.session_state.data.get("merged") is not None:
        st.divider()
        st.markdown(f"### 현재 데이터 미리보기")
        st.dataframe(st.session_state.data["merged"].head(), use_container_width=True)

# ----------------------
#  步骤 2：可视化
# ----------------------
elif st.session_state.step == 2:
    st.subheader("📊 데이터 시각화")
    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 먼저 데이터를 업로드하세요.")
    else:
        df = st.session_state.data["merged"]
        all_cols = df.columns.tolist()
        
        selected_cols = st.multiselect("분석 변수 선택", all_cols, default=all_cols[:5])
        
        if selected_cols:
            df_vis = df[selected_cols]
            st.divider()
            
            c1, c2, c3 = st.columns(3)
            with c1: x_var = st.selectbox("X축", ["None"] + list(df_vis.columns))
            with c2: y_var = st.selectbox("Y축", ["None"] + list(df_vis.select_dtypes(include=np.number).columns))
            with c3: chart = st.selectbox("유형", ["Bar", "Scatter", "Box", "Line", "Hist"])
            
            if x_var != "None" and y_var != "None":
                try:
                    if chart == "Bar": fig = px.bar(df_vis, x=x_var, y=y_var)
                    elif chart == "Scatter": fig = px.scatter(df_vis, x=x_var, y=y_var)
                    elif chart == "Box": fig = px.box(df_vis, x=x_var, y=y_var)
                    elif chart == "Line": fig = px.line(df_vis, x=x_var, y=y_var)
                    elif chart == "Hist": fig = px.histogram(df_vis, x=y_var, color=x_var)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"시각화 오류: {e}")

# ----------------------
#  步骤 3：预处理
# ----------------------
elif st.session_state.step == 3:
    st.subheader("🧹 데이터 전처리")
    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 데이터가 없습니다.")
    else:
        df = st.session_state.data["merged"].copy()
        
        c1, c2 = st.columns(2)
        with c1: target_col = st.selectbox("🎯 타겟 변수 (Y)", df.columns)
        with c2: input_cols = st.multiselect("📋 입력 변수 (X)", [c for c in df.columns if c != target_col])
        
        if input_cols and st.button("🚀 전처리 실행"):
            try:
                # 1. Target NA Drop
                df = df.dropna(subset=[target_col]).reset_index(drop=True)
                X = df[input_cols].copy()
                y = df[target_col].copy()
                
                # 2. Target Encoding
                le_target = None
                if st.session_state.task == "logit" and y.dtype == 'object':
                    le_target = LabelEncoder()
                    y = pd.Series(le_target.fit_transform(y))
                
                # 3. Features Preprocessing
                num_cols = X.select_dtypes(include=np.number).columns.tolist()
                cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
                
                imputer = SimpleImputer(strategy='mean')
                scaler = StandardScaler()
                encoders = {}
                
                if num_cols:
                    X[num_cols] = scaler.fit_transform(imputer.fit_transform(X[num_cols]))
                
                for col in cat_cols:
                    X[col] = X[col].fillna("Unknown").astype(str)
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    encoders[col] = le
                
                # Save State
                final_cols = num_cols + cat_cols
                st.session_state.preprocess.update({
                    "feature_cols": final_cols, "num_cols": num_cols, "cat_cols": cat_cols,
                    "imputer": imputer if num_cols else None,
                    "scaler": scaler if num_cols else None,
                    "encoders": encoders, "target_encoder": le_target,
                    "target_col": target_col
                })
                
                st.session_state.data["X_processed"] = X[final_cols]
                st.session_state.data["y_processed"] = y
                st.success("✅ 전처리 완료!")
                st.dataframe(X.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"전처리 실패: {e}")

# ----------------------
#  步骤 4：模型训练
# ----------------------
elif st.session_state.step == 4:
    st.subheader("🚀 모델 학습")
    if "X_processed" not in st.session_state.data:
        st.warning("⚠️ 전처리를 먼저 수행하세요.")
    else:
        X = st.session_state.data["X_processed"]
        y = st.session_state.data["y_processed"]
        
        col1, col2 = st.columns(2)
        with col1: test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        with col2: reg_weight = st.slider("회귀 가중치", 0.0, 1.0, 0.5)
        
        if st.button("학습 시작"):
            try:
                # Split
                stratify = y if (st.session_state.task == "logit" and y.nunique() > 1) else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=stratify
                )
                
                # Models
                if st.session_state.task == "logit":
                    m1 = LogisticRegression(max_iter=1000)
                    m2 = DecisionTreeClassifier(max_depth=10, random_state=42)
                else:
                    m1 = LinearRegression()
                    m2 = DecisionTreeRegressor(max_depth=10, random_state=42)
                
                m1.fit(X_train, y_train)
                m2.fit(X_train, y_train)
                
                # Save
                st.session_state.models.update({"regression": m1, "decision_tree": m2})
                st.session_state.models["mixed_weights"] = {"regression": reg_weight, "decision_tree": 1-reg_weight}
                st.session_state.data.update({
                    "X_train": X_train, "X_test": X_test,
                    "y_train": y_train, "y_test": y_test
                })
                st.success(f"✅ 학습 완료 (Train: {len(X_train)}, Test: {len(X_test)})")
                
            except Exception as e:
                st.error(f"학습 오류: {e}")

# ----------------------
#  步骤 5：预测
# ----------------------
elif st.session_state.step == 5:
    st.subheader("🎯 예측")
    if st.session_state.models["regression"] is None:
        st.warning("⚠️ 모델 학습이 필요합니다.")
    else:
        # Prediction Helper
        def predict_row(input_data):
            pre = st.session_state.preprocess
            df = pd.DataFrame([input_data])
            
            # Numeric
            if pre["num_cols"]:
                df[pre["num_cols"]] = pre["scaler"].transform(pre["imputer"].transform(df[pre["num_cols"]]))
            
            # Category
            for c in pre["cat_cols"]:
                val = str(df.iloc[0][c])
                enc = pre["encoders"][c]
                df[c] = enc.transform([val])[0] if val in enc.classes_ else 0
                
            X_in = df[pre["feature_cols"]]
            w = st.session_state.models["mixed_weights"]
            m1 = st.session_state.models["regression"]
            m2 = st.session_state.models["decision_tree"]
            
            if st.session_state.task == "logit":
                p1 = m1.predict_proba(X_in)[:,1]
                p2 = m2.predict_proba(X_in)[:,1]
                prob = w["regression"]*p1 + w["decision_tree"]*p2
                return int(prob>=0.5), prob[0]
            else:
                p1 = m1.predict(X_in)
                p2 = m2.predict(X_in)
                return w["regression"]*p1 + w["decision_tree"]*p2, 0

        # Input Form
        if st.session_state.data["merged"] is not None:
            raw_cols = [c for c in st.session_state.data["merged"].columns if c != st.session_state.preprocess["target_col"]]
            
            with st.form("pred"):
                inputs = {}
                cols = st.columns(3)
                for i, c in enumerate(raw_cols[:9]): # Limit inputs for UI
                    inputs[c] = cols[i%3].text_input(c, "0")
                
                if st.form_submit_button("예측하기"):
                    try:
                        res, prob = predict_row(inputs)
                        st.metric("결과", f"{res:.4f}" if prob==0 else f"{res} ({prob:.1%})")
                    except Exception as e:
                        st.error(f"입력 오류: {e}")

# ----------------------
#  步骤 6：评估
# ----------------------
elif st.session_state.step == 6:
    st.subheader("📈 성능 평가")
    if "y_test" not in st.session_state.data:
        st.warning("⚠️ 모델 학습이 필요합니다.")
    else:
        X_test = st.session_state.data["X_test"]
        y_test = st.session_state.data["y_test"]
        m1 = st.session_state.models["regression"]
        m2 = st.session_state.models["decision_tree"]
        w = st.session_state.models["mixed_weights"]
        
        if st.session_state.task == "logit":
            p1 = m1.predict_proba(X_test)[:,1]
            p2 = m2.predict_proba(X_test)[:,1]
            p_mix = w["regression"]*p1 + w["decision_tree"]*p2
            
            acc = accuracy_score(y_test, (p_mix>=0.5).astype(int))
            auc_score = auc(*roc_curve(y_test, p_mix)[:2])
            st.metric("Hybrid ACC", f"{acc:.3f}")
            st.metric("Hybrid AUC", f"{auc_score:.3f}")
            
            fpr, tpr, _ = roc_curve(y_test, p_mix)
            fig = px.area(x=fpr, y=tpr, title="ROC Curve")
            st.plotly_chart(fig, use_container_width=True)
        else:
            p1 = m1.predict(X_test)
            p2 = m2.predict(X_test)
            p_mix = w["regression"]*p1 + w["decision_tree"]*p2
            
            r2 = r2_score(y_test, p_mix)
            rmse = np.sqrt(mean_squared_error(y_test, p_mix))
            st.metric("Hybrid R2", f"{r2:.3f}")
            st.metric("Hybrid RMSE", f"{rmse:.3f}")
            
            fig = px.scatter(x=y_test, y=p_mix, title="Actual vs Predicted")
            fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(dash='dash', color='red'))
        
