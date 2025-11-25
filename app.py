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

# 페이지 설정
st.set_page_config(page_title="하이브리드모형 프레임워크", layout="wide")

# ----------------------
# [수정 1] 초기화: Session State 설정 (필수)
# ----------------------
if "step" not in st.session_state:
    st.session_state.step = 0

if "data" not in st.session_state:
    st.session_state.data = {
        "merged": None, "X_processed": None, "y_processed": None,
        "X_train": None, "X_test": None, "y_train": None, "y_test": None
    }

if "preprocess" not in st.session_state:
    st.session_state.preprocess = {
        "target_col": None, "feature_cols": [], 
        "imputer": None, "scaler": None, "encoders": {}
    }

if "models" not in st.session_state:
    st.session_state.models = {
        "regression": None, "decision_tree": None,
        "mixed_weights": {"regression": 0.5, "decision_tree": 0.5}
    }

if "task" not in st.session_state:
    st.session_state.task = "logit"  # 기본값

# ----------------------
# [수정 2] 사이드바: 단계 이동 네비게이션
# ----------------------
with st.sidebar:
    st.title("🚀 하이브리드 프레임워크")
    
    steps = [
        "0. 홈 (Home)",
        "1. 데이터 업로드",
        "2. 데이터 시각화",
        "3. 데이터 전처리",
        "4. 모델 학습",
        "5. 모델 예측",
        "6. 성능 평가"
    ]
    
    # 현재 단계 표시 및 이동
    current_idx = st.session_state.step
    selected_step = st.radio("단계 선택:", steps, index=current_idx)
    st.session_state.step = steps.index(selected_step)
    
    st.divider()
    
    # 현재 상태 정보 표시
    st.markdown("### ℹ️ 현재 상태")
    if st.session_state.data['merged'] is not None:
        st.success("데이터 로드됨")
    else:
        st.warning("데이터 없음")
        
    st.info(f"작업 유형: {'분류 (Logit)' if st.session_state.task == 'logit' else '회귀 (Regression)'}")

# ==============================================================================
# 메인 로직 시작
# ==============================================================================

# ----------------------
# 단계 0：초기 설정（안내 페이지）
# ----------------------
if st.session_state.step == 0:
    st.subheader("🎉 하이브리드모형 동적 프레임워크에 오신 것을 환영합니다")
    st.markdown("""
    본 프레임워크는 **데이터 수령 후 직접 업로드하여 사용**할 수 있으며，사전 전처리나 모델 학습이 필요 없습니다. 핵심 흐름은 다음과 같습니다：
    
    1. **데이터 업로드**：단일 원본 파일（CSV/Parquet/Excel）을 업로드
    2. **데이터 시각화**：범주형 변수와 수치형 변수를 선택하여 다양한 그래프로 데이터 탐색
    3. **데이터 전처리**：결측값 채우기、범주형 특징 인코딩
    4. **모델 학습**：「회귀 분석+의사결정나무」하이브리드모형 학습
    5. **모델 예측**：단일 데이터 입력 또는 일괄 업로드 예측을 지원
    6. **성능 평가**：하이브리드모형과 단일 모형의 성능을 비교
    
    ### 적용 가능 환경
    - logit 작업（분류）：사용자가 서비스를 수락할지 여부、위반 여부等 이진 예측（모델：로지스틱 회귀+분류 의사결정나무）
    - 의사결정나무 작업（회귀）：판매량、금액、평점等 연속값 예측（모델：선형 회귀+회귀 의사결정나무）
    
    ### 왼쪽 사이드바에서 **「1. 데이터 업로드」**를 선택하여 시작하세요!
    """)

# ----------------------
# 단계 1：데이터 업로드
# ----------------------
elif st.session_state.step == 1:
    st.subheader("📤 데이터 업로드")
    
    tab1, tab2 = st.tabs(["📂 내 파일 업로드", "💾 서버 기본 데이터 사용"])
    
    with tab1:
        st.markdown("지원 형식：CSV、Parquet、Excel（.xlsx/.xls）")
        uploaded_file = st.file_uploader("데이터 파일 선택", type=["csv", "parquet", "xlsx", "xls"], key="single_file")
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.data["merged"] = df
            st.success(f"✅ 파일 업로드 성공! ({len(df):,} 행)")
    
    with tab2:
        DEFAULT_FILE_PATH = "combined_loan_data.csv" 
        st.info(f"💡 **기본 데이터 설명**: 대출 관련 통합 데이터 (`{DEFAULT_FILE_PATH}`)")
        
        if st.button("기본 데이터 불러오기", type="primary"):
            if os.path.exists(DEFAULT_FILE_PATH):
                try:
                    df_default = pd.read_csv(DEFAULT_FILE_PATH)
                    st.session_state.data["merged"] = df_default
                    st.success(f"✅ 기본 데이터 로드 성공! ({len(df_default):,} 행)")
                    st.rerun()
                except Exception as e:
                    st.error(f"파일 읽기 오류: {e}")
            else:
                st.error(f"⚠️ 파일을 찾을 수 없습니다: {DEFAULT_FILE_PATH}")

    if st.session_state.data.get("merged") is not None:
        df_merged = st.session_state.data["merged"]
        st.divider()
        st.markdown(f"### ✅ 현재 로드된 데이터 ({len(df_merged):,} 행)")
        st.dataframe(df_merged.head(5), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**열 이름 (상위 10개)**")
            st.write(", ".join(df_merged.columns.tolist()[:10]) + "...")
        with col2:
            st.write("**결측값 총 개수**")
            st.write(f"{df_merged.isnull().sum().sum()} 개")
        with col3:
            st.write("**데이터 유형**")
            st.write(df_merged.dtypes.value_counts().to_string())

# ----------------------
# 단계 2：데이터 시각화
# ----------------------
elif st.session_state.step == 2:
    st.subheader("📊 데이터 시각화")
    
    if st.session_state.data["merged"] is None:
        st.warning("먼저「데이터 업로드」단계를 완료하세요")
    else:
        df = st.session_state.data["merged"]
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_var = st.selectbox("📋 X축：범주형 변수", options=["선택 안 함"] + cat_cols, index=0)
            x_var = None if x_var == "선택 안 함" else x_var
        with col2:
            y_var = st.selectbox("📈 Y축：수치형 변수", options=num_cols, index=0 if num_cols else None)
        with col3:
            graph_types = [
                "막대 그래프（평균값）", "박스 플롯（분포）", "바이올린 플롯（분포+밀도）",
                "산점도（개별 데이터）", "선 그래프（추세）", "히스토그램（분포）"
            ]
            graph_type = st.selectbox("📊 그래프 유형", options=graph_types, index=0)
        
        st.divider()
        if y_var:
            if graph_type == "히스토그램（분포）":
                st.markdown(f"### {y_var} 분포（히스토그램）")
                plot_df = df[[y_var] + ([x_var] if x_var else [])].dropna()
                
                try:
                    bins = st.slider("구간 개수", 10, 100, 30, 5)
                    if x_var:
                        fig = px.histogram(plot_df, x=y_var, color=x_var, barmode="overlay", opacity=0.7, nbins=bins,
                                         title=f"{x_var}별 {y_var} 분포", color_discrete_sequence=px.colors.qualitative.Pastel)
                    else:
                        fig = px.histogram(plot_df, x=y_var, nbins=bins, title=f"{y_var} 전체 분포",
                                         color_discrete_sequence=["#636EFA"], marginal="box")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"그래프 생성 실패: {str(e)}")
            else:
                if not x_var:
                    st.warning("이 그래프 유형은 X축(범주형 변수) 선택이 필요합니다.")
                else:
                    st.markdown(f"### {x_var} vs {y_var} ({graph_type})")
                    plot_df = df[[x_var, y_var]].dropna()
                    try:
                        if graph_type == "막대 그래프（평균값）":
                            bar_data = plot_df.groupby(x_var)[y_var].mean().reset_index()
                            fig = px.bar(bar_data, x=x_var, y=y_var, color=x_var, title=f"{x_var}별 {y_var} 평균")
                        elif graph_type == "박스 플롯（분포）":
                            fig = px.box(plot_df, x=x_var, y=y_var, color=x_var, title=f"{x_var}별 {y_var} 분포")
                        elif graph_type == "바이올린 플롯（분포+밀도）":
                            fig = px.violin(plot_df, x=x_var, y=y_var, color=x_var, box=True)
                        elif graph_type == "산점도（개별 데이터）":
                            fig = px.scatter(plot_df, x=x_var, y=y_var, color=x_var, opacity=0.6)
                        elif graph_type == "선 그래프（추세）":
                            line_data = plot_df.groupby(x_var)[y_var].mean().reset_index()
                            fig = px.line(line_data, x=x_var, y=y_var, markers=True)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"그래프 생성 실패: {str(e)}")
        else:
            st.warning("Y축(수치형 변수)을 선택해주세요.")

# ----------------------
# 단계 3：데이터 전처리
# ----------------------
elif st.session_state.step == 3:
    st.subheader("🧹 데이터 전처리")
    
    if st.session_state.data["merged"] is None:
        st.warning("먼저「데이터 업로드」단계를 완료하세요")
    else:
        df_merged = st.session_state.data["merged"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 데이터 기본 정보")
            st.write(f"총 데이터: {len(df_merged):,} 행 × {len(df_merged.columns)} 열")
            st.dataframe(df_merged.dtypes.value_counts().reset_index(), use_container_width=True)
        with col2:
            st.markdown("### 결측값 분포")
            missing_info = df_merged.isnull().sum()[df_merged.isnull().sum() > 0].reset_index()
            if len(missing_info) > 0:
                missing_info.columns = ["필드명", "결측값"]
                st.dataframe(missing_info, use_container_width=True)
            else:
                st.success("결측값이 없습니다.")
        
        st.divider()
        st.markdown("### ⚙️ 전처리 설정")
        
        # 1. 타겟 열 선택
        if len(df_merged.columns) > 0:
            target_col = st.selectbox("타겟 열 선택 (예측 대상)", options=df_merged.columns, index=0)
            st.session_state.preprocess["target_col"] = target_col
        else:
            st.error("데이터에 열이 없습니다.")
            st.stop()
        
        # [수정 3] 분석 유형(Task) 선택 추가
        st.markdown("#### 분석 유형 선택")
        task_choice = st.radio("이 데이터의 예측 목표는 무엇입니까?", 
                             ["분류 (예: 합격/불합격, 0/1)", "회귀 (예: 가격, 수량, 점수)"])
        st.session_state.task = "logit" if "분류" in task_choice else "regression"
            
        # 2. 특징 열 선택
        exclude_cols = st.multiselect("제외할 열 선택 (ID 등 무관한 필드)", 
                                    options=[c for c in df_merged.columns if c != target_col])
        feature_cols = [c for c in df_merged.columns if c not in exclude_cols + [target_col]]
        st.session_state.preprocess["feature_cols"] = feature_cols
        
        if not feature_cols:
            st.warning("특징 열이 선택되지 않았습니다.")
            
        # 3. 전처리 옵션
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            impute_strategy = st.selectbox("수치형 결측값 채우기", ["중앙값", "평균값", "최빈값"])
            impute_strategy_map = {"중앙값": "median", "평균값": "mean", "최빈값": "most_frequent"}
        with col_p2:
            cat_encoding = st.selectbox("범주형 인코딩", ["레이블 인코딩", "원-핫 인코딩"])
            
        if st.button("전처리 및 변환 시작", type="primary"):
            if not feature_cols:
                st.error("특징 열이 없습니다.")
                st.stop()
            
            try:
                X = df_merged[feature_cols].copy()
                y = df_merged[target_col].copy()
                
                num_cols = X.select_dtypes(include=["int64", "float64"]).columns
                cat_cols = X.select_dtypes(include=["object", "category"]).columns
                
                # 수치형 처리
                imputer = SimpleImputer(strategy=impute_strategy_map[impute_strategy])
                if len(num_cols) > 0:
                    X[num_cols] = imputer.fit_transform(X[num_cols])
                    scaler = StandardScaler()
                    X[num_cols] = scaler.fit_transform(X[num_cols])
                else:
                    scaler = StandardScaler() # 빈 scaler
                
                # 범주형 처리
                encoders = {}
                for col in cat_cols:
                    X[col] = X[col].fillna("알 수 없음").astype(str)
                    if cat_encoding == "레이블 인코딩":
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col])
                        encoders[col] = le
                    else:
                        ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown='ignore')
                        ohe_result = ohe.fit_transform(X[[col]])
                        ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
                        X = pd.concat([X.drop(col, axis=1), pd.DataFrame(ohe_result, columns=ohe_cols)], axis=1)
                        encoders[col] = (ohe, ohe_cols)
                
                st.session_state.preprocess.update({
                    "imputer": imputer, "scaler": scaler, "encoders": encoders, 
                    "feature_cols": list(X.columns)
                })
                st.session_state.data["X_processed"] = X
                st.session_state.data["y_processed"] = y
                
                st.success("✅ 전처리 완료!")
                st.dataframe(X.head(3))
                
            except Exception as e:
                st.error(f"전처리 오류: {str(e)}")

# ----------------------
# 단계 4：모델 학습
# ----------------------
elif st.session_state.step == 4:
    st.subheader("🚀 하이브리드모형 학습")
    
    if "X_processed" not in st.session_state.data or st.session_state.data["X_processed"] is None:
        st.warning("먼저「데이터 전처리」단계를 완료하세요")
    else:
        X = st.session_state.data["X_processed"]
        y = st.session_state.data["y_processed"]
        
        st.markdown("### 1. 학습 설정")
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("테스트셋 비율", 0.1, 0.4, 0.2, 0.05)
        with col2:
            st.info(f"현재 작업 유형: **{st.session_state.task}**")
            
        # Stratify 로직
        stratify_param = None
        if st.session_state.task == "logit":
            if y.nunique() >= 2 and (y.value_counts() >= 2).all():
                stratify_param = y
                st.success("✅ 층화 추출(Stratified Sampling) 적용됨")
            else:
                st.warning("⚠️ 클래스 불균형 또는 샘플 부족으로 층화 추출 미적용")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify_param
        )
        
        # [수정 4] 가중치 설정 추가
        st.markdown("### 2. 하이브리드 가중치 설정")
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            reg_weight = st.slider("회귀분석(Logistic/Linear) 가중치", 0.0, 1.0, 0.5)
        with w_col2:
            st.metric("의사결정나무 가중치", f"{1.0 - reg_weight:.1f}")
            
        # 모델 정의
        if st.session_state.task == "logit":
            reg_model = LogisticRegression(max_iter=1000)
            dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
        else:
            reg_model = LinearRegression()
            dt_model = DecisionTreeRegressor(random_state=42, max_depth=10)
            
        if st.button("모델 학습 시작", type="primary"):
            with st.spinner("모델 학습 중..."):
                try:
                    reg_model.fit(X_train, y_train)
                    dt_model.fit(X_train, y_train)
                    
                    st.session_state.models["regression"] = reg_model
                    st.session_state.models["decision_tree"] = dt_model
                    st.session_state.models["mixed_weights"] = {
                        "regression": reg_weight, "decision_tree": 1.0 - reg_weight
                    }
                    
                    st.session_state.data.update({
                        "X_train": X_train, "X_test": X_test, 
                        "y_train": y_train, "y_test": y_test
                    })
                    
                    st.success("✅ 모델 학습 완료!")
                    st.markdown(f"**학습 데이터**: {len(X_train):,}개 | **테스트 데이터**: {len(X_test):,}개")
                    
                except Exception as e:
                    st.error(f"학습 실패: {str(e)}")

# ----------------------
# 단계 5：모델 예측
# ----------------------
elif st.session_state.step == 5:
    st.subheader("🎯 모델 예측")
    
    if st.session_state.models["regression"] is None:
        st.warning("먼저「모델 학습」단계를 완료하세요")
    else:
        def predict_pipeline(input_df):
            # 1. 전처리 적용
            preprocess = st.session_state.preprocess
            X = input_df.copy()
            
            num_cols = X.select_dtypes(include=["int64", "float64"]).columns
            cat_cols = X.select_dtypes(include=["object", "category"]).columns
            
            # 수치형 변환
            if preprocess["imputer"]:
                X[num_cols] = preprocess["imputer"].transform(X[num_cols])
                X[num_cols] = preprocess["scaler"].transform(X[num_cols])
            
            # 범주형 변환
            for col in cat_cols:
                X[col] = X[col].fillna("알 수 없음").astype(str)
                encoder = preprocess["encoders"].get(col)
                if encoder:
                    if isinstance(encoder, LabelEncoder):
                        # 미지의 값 처리
                        known_classes = set(encoder.classes_)
                        X[col] = X[col].apply(lambda x: x if x in known_classes else "알 수 없음")
                        # "알 수 없음"이 클래스에 없으면 추가 (임시 처리)
                        if "알 수 없음" not in known_classes:
                             # LabelEncoder는 동적 추가가 어려우므로 0으로 대체하거나 예외처리 필요
                             # 여기서는 편의상 가장 빈도 높은 값으로 대체 가정 또는 0
                             pass 
                        # transform 시 에러 방지를 위해 try-except 권장
                        try:
                            X[col] = encoder.transform(X[col])
                        except:
                            X[col] = 0
                    else:
                        # OneHotEncoder
                        ohe, ohe_cols = encoder
                        ohe_result = ohe.transform(X[[col]])
                        X = pd.concat([X.drop(col, axis=1), pd.DataFrame(ohe_result, columns=ohe_cols)], axis=1)
            
            # 컬럼 순서 맞추기
            missing_cols = set(preprocess["feature_cols"]) - set(X.columns)
            for c in missing_cols:
                X[c] = 0
            X = X[preprocess["feature_cols"]]
            
            # 2. 예측
            reg_model = st.session_state.models["regression"]
            dt_model = st.session_state.models["decision_tree"]
            weights = st.session_state.models["mixed_weights"]
            
            if st.session_state.task == "logit":
                reg_p = reg_model.predict_proba(X)[:, 1]
                dt_p = dt_model.predict_proba(X)[:, 1]
                mixed_p = weights["regression"] * reg_p + weights["decision_tree"] * dt_p
                pred = (mixed_p >= 0.5).astype(int)
                return pred, mixed_p
            else:
                reg_p = reg_model.predict(X)
                dt_p = dt_model.predict(X)
                mixed_p = weights["regression"] * reg_p + weights["decision_tree"] * dt_p
                return mixed_p, None

        mode = st.radio("예측 방식", ["단일 데이터 입력", "일괄 업로드 (CSV)"])
        
        if mode == "단일 데이터 입력":
            st.markdown("#### 데이터 입력")
            feature_cols = st.session_state.preprocess["feature_cols"]
            # 원본 데이터프레임 구조 참조 (인코딩 전)
            original_features = [c for c in st.session_state.data["merged"].columns 
                               if c not in [st.session_state.preprocess["target_col"]]]
            
            input_data = {}
            with st.form("pred_form"):
                cols = st.columns(3)
                for i, col in enumerate(original_features[:9]): # 최대 9개만 표시
                    with cols[i % 3]:
                        # 원본 데이터 타입 확인
                        col_type = st.session_state.data["merged"][col].dtype
                        if pd.api.types.is_numeric_dtype(col_type):
                            input_data[col] = st.number_input(col, value=0.0)
                        else:
                            opts = st.session_state.data["merged"][col].dropna().unique()
                            input_data[col] = st.selectbox(col, options=opts)
                submit = st.form_submit_button("예측하기")
            
            if submit:
                input_df = pd.DataFrame([input_data])
                pred, proba = predict_pipeline(input_df)
                st.divider()
                if st.session_state.task == "logit":
                    st.metric("예측 결과", "양성(Positive)" if pred[0]==1 else "음성(Negative)")
                    st.metric("확률", f"{proba[0]:.2%}")
                else:
                    st.metric("예측 값", f"{pred[0]:.4f}")
                    
        else:
            up_file = st.file_uploader("CSV 업로드", type=["csv"])
            if up_file:
                batch_df = pd.read_csv(up_file)
                if st.button("일괄 예측 시작"):
                    pred, proba = predict_pipeline(batch_df)
                    batch_df["Predicted"] = pred
                    if proba is not None:
                        batch_df["Probability"] = proba
                    st.dataframe(batch_df.head())
                    st.download_button("결과 다운로드", batch_df.to_csv().encode('utf-8'), "prediction.csv")

# ----------------------
# 단계 6：성능 평가
# ----------------------
elif st.session_state.step == 6:
    st.subheader("📈 모델 성능 평가")
    
    if st.session_state.models["regression"] is None:
        st.warning("먼저「모델 학습」단계를 완료하세요")
    else:
        X_test = st.session_state.data["X_test"]
        y_test = st.session_state.data["y_test"]
        
        reg_model = st.session_state.models["regression"]
        dt_model = st.session_state.models["decision_tree"]
        weights = st.session_state.models["mixed_weights"]
        
        if st.session_state.task == "logit":
            # 확률 계산
            reg_p = reg_model.predict_proba(X_test)[:, 1]
            dt_p = dt_model.predict_proba(X_test)[:, 1]
            mixed_p = weights["regression"] * reg_p + weights["decision_tree"] * dt_p
            
            # 예측값
            reg_pred = reg_model.predict(X_test)
            dt_pred = dt_model.predict(X_test)
            mixed_pred = (mixed_p >= 0.5).astype(int)
            
            # 평가 함수
            def get_metrics(y, pred, proba):
                return {
                    "ACC": accuracy_score(y, pred),
                    "AUC": auc(*roc_curve(y, proba)[:2])
                }
            
            m1 = get_metrics(y_test, reg_pred, reg_p)
            m2 = get_metrics(y_test, dt_pred, dt_p)
            m3 = get_metrics(y_test, mixed_pred, mixed_p)
            
            metrics = pd.DataFrame([m1, m2, m3], index=["회귀분석", "의사결정나무", "하이브리드"])
            st.table(metrics)
            
            # ROC 곡선
            fpr, tpr, _ = roc_curve(y_test, mixed_p)
            fig = px.area(x=fpr, y=tpr, title=f"ROC Curve (Hybrid AUC={m3['AUC']:.3f})", 
                        labels=dict(x="False Positive Rate", y="True Positive Rate"))
            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # 회귀 평가
            reg_pred = reg_model.predict(X_test)
            dt_pred = dt_model.predict(X_test)
            mixed_pred = weights["regression"] * reg_pred + weights["decision_tree"] * dt_pred
            
            def get_reg_metrics(y, pred):
                return {
                    "MAE": mean_absolute_error(y, pred),
                    "RMSE": np.sqrt(mean_squared_error(y, pred)),
                    "R2": r2_score(y, pred)
                }
            
            m1 = get_reg_metrics(y_test, reg_pred)
            m2 = get_reg_metrics(y_test, dt_pred)
            m3 = get_reg_metrics(y_test, mixed_pred)
            
            metrics = pd.DataFrame([m1, m2, m3], index=["선형회귀", "의사결정나무", "하이브리드"])
            st.table(metrics)
            
            # 예측 vs 실제
            fig = px.scatter(x=y_test, y=mixed_pred, title="실제값 vs 예측값 (Hybrid)", 
                           labels={"x": "실제값", "y": "예측값"})
            fig.add_shape(type='line', line=dict(dash='dash', color='red'), 
                        x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max())
            st.plotly_chart(fig, use_container_width=True)
            
        # 중요도 (Tree 기준)
        if hasattr(dt_model, "feature_importances_"):
            st.markdown("### 🌳 변수 중요도 (의사결정나무 기준)")
            imp_df = pd.DataFrame({
                "Feature": st.session_state.preprocess["feature_cols"],
                "Importance": dt_model.feature_importances_
            }).sort_values("Importance", ascending=False).head(10)
            
            fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation='h', title="Top 10 Feature Importance")
            st.plotly_chart(fig_imp, use_container_width=True)
