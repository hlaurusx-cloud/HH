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
# 1. 페이지 기본 설정
# ----------------------
st.set_page_config(
    page_title="하이브리드모형 동적 프레임워크（의사결정나무+회귀분석）",
    page_icon="📊",
    layout="wide"
)

# 전역 상태 관리（각 단계 데이터/모델 저장，새로고침 시 손실 방지）
if "step" not in st.session_state:
    st.session_state.step = 0  # 0:초기화면 1:데이터업로드 2:데이터시각화 3:데이터전처리 4:모델학습 5:예측 6:평가
if "data" not in st.session_state:
    st.session_state.data = {"merged": None}  # 단일 파일만 저장
if "preprocess" not in st.session_state:
    st.session_state.preprocess = {"imputer": None, "scaler": None, "encoders": None, "feature_cols": None, "target_col": None}
if "models" not in st.session_state:
    # 模型：regression（회귀분석）、decision_tree（의사결정나무）
    st.session_state.models = {"regression": None, "decision_tree": None, "mixed_weights": {"regression": 0.3, "decision_tree": 0.7}}
if "task" not in st.session_state:
    st.session_state.task = "logit"  # 기본값 logit（분류），의사결정나무（회귀）로 전환 가능

# ----------------------
# 2. 사이드바：단계导航 + 핵심 설정
# ----------------------
st.sidebar.title("📌 하이브리드모형 작업 흐름")
st.sidebar.divider()

# 단계导航 버튼（新增「데이터 시각화」단계）
steps = ["초기 설정", "데이터 업로드", "데이터 시각화", "데이터 전처리", "모델 학습", "모델 예측", "성능 평가"]
for i, step_name in enumerate(steps):
    if st.sidebar.button(step_name, key=f"btn_{i}"):
        st.session_state.step = i

# 핵심 설정（작업 유형 + 혼합 가중치）
st.sidebar.divider()
st.sidebar.subheader("핵심 설정")
st.session_state.task = st.sidebar.radio("작업 유형", options=["logit", "의사결정나무"], index=0)

if st.session_state.step >= 4:  # 모델 학습 후 가중치 조정 가능
    st.sidebar.subheader("하이브리드모형 가중치")
    reg_weight = st.sidebar.slider(
        "회귀 분석 가중치（해석력 강함）",
        min_value=0.0, max_value=1.0, value=st.session_state.models["mixed_weights"]["regression"], step=0.1
    )
    st.session_state.models["mixed_weights"]["regression"] = reg_weight
    st.session_state.models["mixed_weights"]["decision_tree"] = 1 - reg_weight
    st.sidebar.text(f"의사결정나무 가중치（정확도 높음）：{1 - reg_weight:.1f}")

# ----------------------
# 3. 메인 페이지：단계별 내용 표시
# ----------------------
st.title("📊 하이브리드모형 동적 배포 프레임워크")
st.markdown("**단일 원본 데이터 파일 업로드 후，시각화→전처리→학습→예측 전과정을 한 번에 완성**")
st.markdown("### 🧩 핵심 모델：회귀 분석（Regression）+ 의사결정나무（Decision Tree）")
st.divider()

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
# 단계 1：데이터 업로드 (인코딩 자동 해결 버전)
# ----------------------
elif st.session_state.step == 1:
    st.subheader("📤 데이터 업로드")
    
    tab1, tab2 = st.tabs(["📂 내 파일 업로드", "💾 서버 기본 데이터 사용"])
    
    # 인코딩 처리를 위한 내부 함수
    def load_csv_safe(file_buffer):
        # 시도할 인코딩 목록 (순서대로 시도)
        encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig', 'latin1']
        
        for enc in encodings:
            try:
                file_buffer.seek(0) # 파일 포인터 초기화
                df = pd.read_csv(file_buffer, encoding=enc)
                return df, enc # 성공하면 데이터와 인코딩 반환
            except UnicodeDecodeError:
                continue # 실패하면 다음 인코딩 시도
            except Exception as e:
                return None, str(e) # 기타 에러
        return None, "모든 인코딩 시도 실패"

    with tab1:
        st.markdown("지원 형식：CSV、Parquet、Excel（.xlsx/.xls）")
        uploaded_file = st.file_uploader("데이터 파일 선택", type=["csv", "parquet", "xlsx", "xls"], key="single_file")
        
        if uploaded_file:
            try:
                df = None
                # 확장자별 로드
                if uploaded_file.name.endswith('.csv'):
                    df, enc_used = load_csv_safe(uploaded_file)
                    if df is None:
                        st.error(f"❌ CSV 파일 읽기 실패: {enc_used}")
                    else:
                        st.caption(f"ℹ️ 감지된 인코딩: {enc_used}")
                        
                elif uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                if df is not None:
                    # 인덱스 초기화 (전처리 에러 방지용 필수)
                    df = df.reset_index(drop=True)
                    st.session_state.data["merged"] = df
                    st.success(f"✅ 파일 업로드 성공! ({len(df):,} 행)")
                
            except Exception as e:
                st.error(f"❌ 파일 처리 중 오류 발생: {e}")
    
    with tab2:
        DEFAULT_FILE_PATH = "combined_loan_data.csv" 
        st.info(f"💡 **기본 데이터 설명**: 대출 관련 통합 데이터 (`{DEFAULT_FILE_PATH}`)")
        
        if st.button("기본 데이터 불러오기", type="primary"):
            if os.path.exists(DEFAULT_FILE_PATH):
                # 기본 파일도 안전하게 로드 시도
                with open(DEFAULT_FILE_PATH, 'rb') as f:
                    df_default, enc_used = load_csv_safe(f)
                
                if df_default is not None:
                    st.session_state.data["merged"] = df_default.reset_index(drop=True)
                    st.success(f"✅ 기본 데이터 로드 성공! ({len(df_default):,} 행, 인코딩: {enc_used})")
                    st.rerun()
                else:
                    st.error("❌ 기본 파일을 읽을 수 없습니다 (인코딩 오류).")
            else:
                st.error(f"⚠️ 파일을 찾을 수 없습니다: {DEFAULT_FILE_PATH}")

    # 데이터 미리보기
    if st.session_state.data.get("merged") is not None:
        df_merged = st.session_state.data["merged"]
        st.divider()
        st.markdown(f"### ✅ 현재 로드된 데이터 ({len(df_merged):,} 행)")
        st.dataframe(df_merged.head(5), use_container_width=True)
# ----------------------
# 단계 2：데이터 시각화 (수정됨)
# ----------------------
elif st.session_state.step == 2:
    st.subheader("📊 데이터 시각화")
    
    # [수정된 부분] 따옴표가 닫히지 않았던 에러 해결
    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 먼저 '데이터 업로드' 단계를 완료하세요")
    else:
        df = st.session_state.data["merged"]
        
        # --- 변수 선택 (Variable Selection) ---
        st.markdown("### 1️⃣ 시각화할 변수 선택")
        all_cols = df.columns.tolist()
        default_selection = all_cols[:10] if len(all_cols) > 10 else all_cols
        
        selected_cols = st.multiselect(
            "분석 대상 변수 선택",
            options=all_cols,
            default=default_selection
        )
        
        if not selected_cols:
            st.error("⚠️ 최소 하나 이상의 변수를 선택해야 시각화가 가능합니다.")
        else:
            df_vis = df[selected_cols]
            st.divider()
            
            # --- 그래프 설정 ---
            st.markdown("### 2️⃣ 그래프 설정")
            cat_cols = df_vis.select_dtypes(include=["object", "category"]).columns.tolist()
            num_cols = df_vis.select_dtypes(include=["int64", "float64"]).columns.tolist()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                x_var = st.selectbox("📋 X축 (범주형)", ["선택 안 함"] + cat_cols)
                if x_var == "선택 안 함": x_var = None
            with col2:
                y_var = st.selectbox("📈 Y축 (수치형)", num_cols if num_cols else ["없음"])
            with col3:
                graph_type = st.selectbox("📊 그래프 유형", [
                    "막대 그래프", "박스 플롯", "산점도", "히스토그램", "선 그래프"
                ])
            
            st.divider()
            
            # 시각화 출력
            if y_var and y_var != "없음":
                try:
                    if graph_type == "히스토그램":
                        fig = px.histogram(df_vis, x=y_var, color=x_var, title=f"{y_var} 분포")
                    elif graph_type == "막대 그래프" and x_var:
                        avg_df = df_vis.groupby(x_var)[y_var].mean().reset_index()
                        fig = px.bar(avg_df, x=x_var, y=y_var, color=x_var, title=f"{x_var}별 {y_var} 평균")
                    elif graph_type == "박스 플롯" and x_var:
                        fig = px.box(df_vis, x=x_var, y=y_var, color=x_var, title=f"{x_var}별 {y_var} 분포")
                    elif graph_type == "산점도" and x_var:
                        fig = px.scatter(df_vis, x=x_var, y=y_var, color=x_var, title=f"{x_var} vs {y_var}")
                    elif graph_type == "선 그래프" and x_var:
                        line_df = df_vis.groupby(x_var)[y_var].mean().reset_index()
                        fig = px.line(line_df, x=x_var, y=y_var, markers=True, title=f"{x_var}별 {y_var} 추세")
                    else:
                        fig = None
                        st.info("X축 변수를 선택해주세요.")
                        
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"그래프 생성 오류: {e}")
            else:
                st.info("Y축 변수를 선택하면 그래프가 표시됩니다.")

# ----------------------
# 단계 3：데이터 전처리 & 변수 선택 (선택 변수만 처리)
# ----------------------
elif st.session_state.step == 3:
    st.subheader("🧹 데이터 전처리 & 변수 선택")
    
    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 먼저 '데이터 업로드' 단계를 완료하세요.")
    else:
        # 원본 데이터 안전하게 가져오기 (인덱스 리셋)
        df_origin = st.session_state.data["merged"].copy().reset_index(drop=True)
        all_cols = df_origin.columns.tolist()

        st.markdown("### 1️⃣ 분석에 사용할 변수 선택 (Selection)")
        
        # 1. 타겟 변수(Y) 선택
        target_col = st.selectbox("🎯 타겟 변수 (Y) 선택", options=all_cols)
        st.session_state.preprocess["target_col"] = target_col
        
        # 2. 입력 변수(X) 다중 선택
        # 타겟을 제외한 나머지 변수들을 옵션으로 제공
        feature_candidates = [c for c in all_cols if c != target_col]
        
        # 기본적으로 너무 많은 컬럼이 선택되지 않도록 상위 10개만 기본 선택
        default_feats = feature_candidates[:10] if len(feature_candidates) > 10 else feature_candidates
        
        selected_features = st.multiselect(
            "📋 입력 변수 (X) 선택 (이 변수들만 전처리합니다)",
            options=feature_candidates,
            default=default_feats
        )
        
        st.divider()

        if not selected_features:
            st.warning("⚠️ 분석할 변수를 최소 1개 이상 선택해주세요.")
        else:
            # 탭 구성
            tab_proc, tab_imp = st.tabs(["⚡ 선택된 변수 전처리 실행", "📊 변수 중요도 확인"])
            
            with tab_proc:
                st.markdown(f"**선택된 {len(selected_features)}개의 변수만 추출하여 전처리를 수행합니다.**")
                
                if st.button("🚀 전처리 시작 (결측치+인코딩)", type="primary"):
                    with st.spinner("데이터 변환 중..."):
                        try:
                            # [핵심 로직] 선택된 컬럼만 별도 데이터프레임으로 추출 (Subset)
                            # 인덱스를 리셋하여 차원 불일치 문제 원천 차단
                            X = df_origin[selected_features].copy().reset_index(drop=True)
                            y = df_origin[target_col].copy().reset_index(drop=True)
                            
                            # 수치형/범주형 자동 분류
                            num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                            
                            # 1. 수치형 처리 (Imputation + Scaling)
                            imputer = SimpleImputer(strategy='mean')
                            scaler = StandardScaler()
                            
                            if num_cols:
                                # fit_transform 결과는 numpy array이므로 다시 DF로 변환해야 안전함
                                X_num_arr = imputer.fit_transform(X[num_cols])
                                X_num_arr = scaler.fit_transform(X_num_arr)
                                X[num_cols] = pd.DataFrame(X_num_arr, columns=num_cols, index=X.index)
                            
                            # 2. 범주형 처리 (Label Encoding)
                            encoders = {}
                            for col in cat_cols:
                                # 결측치 처리 및 문자열 변환
                                X[col] = X[col].fillna("Unknown").astype(str)
                                
                                le = LabelEncoder()
                                # Series로 변환하여 할당 (인덱스 매칭 보장)
                                X[col] = pd.Series(le.fit_transform(X[col]), index=X.index)
                                encoders[col] = le
                            
                            # 3. 결과 저장 (전역 상태)
                            st.session_state.preprocess["feature_cols"] = selected_features
                            st.session_state.preprocess["imputer"] = imputer if num_cols else None
                            st.session_state.preprocess["scaler"] = scaler if num_cols else None
                            st.session_state.preprocess["encoders"] = encoders
                            
                            st.session_state.data["X_processed"] = X
                            st.session_state.data["y_processed"] = y
                            
                            st.success("✅ 전처리 완료! 데이터 형식이 모델 학습에 맞게 변환되었습니다.")
                            st.write("▼ 변환된 데이터 미리보기 (상위 3행)")
                            st.dataframe(X.head(3))
                            
                        except Exception as e:
                            st.error(f"❌ 전처리 오류 발생: {str(e)}")
                            st.warning("팁: 데이터에 섞여있는 특수문자나 인덱스 문제일 수 있습니다.")

            with tab_imp:
                if "X_processed" in st.session_state.data:
                    st.markdown("##### 🧬 변수 중요도 (Feature Importance)")
                    if st.button("중요도 분석 실행"):
                        X_p = st.session_state.data["X_processed"]
                        y_p = st.session_state.data["y_processed"]
                        
                        # 간단한 트리 모델로 중요도 산출
                        model = DecisionTreeClassifier(max_depth=5) if st.session_state.task == "logit" else DecisionTreeRegressor(max_depth=5)
                        model.fit(X_p, y_p)
                        
                        imp = pd.DataFrame({
                            "Feature": X_p.columns,
                            "Importance": model.feature_importances_
                        }).sort_values("Importance", ascending=False)
                        
                        fig = px.bar(imp, x="Importance", y="Feature", orientation='h', title="변수 중요도 Top Features")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("👈 먼저 [전처리 실행] 버튼을 눌러주세요.")
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
