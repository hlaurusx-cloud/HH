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
# 단계 2：데이터 시각화 (수정됨)
# ----------------------
elif st.session_state.step == 2:
    st.subheader("📊 데이터 시각화")
    
    if st.session_state.data["merged"] is None:
        st.warning("먼저「데이터 업로드」단계를 완료하세요")
    else:
        df = st.session_state.data["merged"]
        
        # --- [추가 기능] 변수 선택 (Variable Selection) ---
        st.markdown("### 1️⃣ 시각화할 변수 선택")
        st.caption("데이터의 변수가 많을 경우, 분석하고 싶은 변수만 선택해서 시각화할 수 있습니다.")
        
        all_cols = df.columns.tolist()
        # 기본적으로 상위 10개 혹은 전체 선택
        default_selection = all_cols[:10] if len(all_cols) > 10 else all_cols
        
        selected_cols = st.multiselect(
            "분석 대상 변수 선택 (다중 선택 가능)",
            options=all_cols,
            default=default_selection,
            placeholder="변수를 선택하세요..."
        )
        
        if not selected_cols:
            st.error("⚠️ 최소 하나 이상의 변수를 선택해야 시각화가 가능합니다.")
            st.stop()
            
        # 선택된 데이터만 필터링하여 시각화용 데이터프레임 생성
        df_vis = df[selected_cols]
        st.divider()
        
        # --- 기존 시각화 로직 (선택된 변수만 적용) ---
        st.markdown("### 2️⃣ 그래프 설정")
        
        # 필터링된 df_vis를 기준으로 데이터 타입 분류
        cat_cols = df_vis.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df_vis.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            # 선택된 변수 내에서만 옵션 제공
            x_var = st.selectbox("📋 X축：범주형 변수", options=["선택 안 함"] + cat_cols, index=0)
            x_var = None if x_var == "선택 안 함" else x_var
        with col2:
            # 선택된 변수 내에서만 옵션 제공
            y_var = st.selectbox("📈 Y축：수치형 변수", options=num_cols, index=0 if num_cols else None)
        with col3:
            graph_types = [
                "막대 그래프（평균값）", "박스 플롯（분포）", "바이올린 플롯（분포+밀도）",
                "산점도（개별 데이터）", "선 그래프（추세）", "히스토그램（분포）"
            ]
            graph_type = st.selectbox("📊 그래프 유형", options=graph_types, index=0)
        
        st.divider()
        
        # 시각화 그리기 로직 (df 대신 df_vis 사용해도 되지만, 컬럼이 존재하므로 df 사용 무방)
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
            st.warning("Y축(수치형 변수)을 선택하거나, 선택된 변수 목록에 수치형 데이터가 포함되어 있는지 확인해주세요.")
# ----------------------
# 단계 3：데이터 전처리 & Stepwise 변수 선택 (업그레이드)
# ----------------------
elif st.session_state.step == 3:
    st.subheader("🧹 데이터 전처리 & 자동 변수 선택 (Stepwise)")
    
    if st.session_state.data["merged"] is None:
        st.warning("먼저「데이터 업로드」단계를 완료하세요")
    else:
        df_merged = st.session_state.data["merged"]
        
        # 탭 구조 도입: 기본 전처리와 고급 변수 선택 분리
        tab_basic, tab_stepwise = st.tabs(["1️⃣ 기본 전처리 (필수)", "2️⃣ 변수 선택 과정 시각화 (Stepwise)"])
        
        # =========================================================
        # Tab 1: 기본 전처리 (결측치/인코딩) - Stepwise를 위해 필수
        # =========================================================
        with tab_basic:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 데이터 현황")
                st.write(f"데이터 크기: **{len(df_merged):,}** 행")
            with col2:
                st.markdown("### 결측값 확인")
                missing_sum = df_merged.isnull().sum()
                if missing_sum.sum() > 0:
                    st.dataframe(missing_sum[missing_sum > 0].reset_index(name="결측수"), height=100)
                else:
                    st.success("결측값이 없습니다.")
            
            st.divider()
            
            # 1. 타겟 및 분석 유형 설정
            target_col = st.selectbox("🎯 타겟 변수 (예측 대상)", df_merged.columns)
            st.session_state.preprocess["target_col"] = target_col
            
            task_choice = st.radio("분석 유형", ["분류 (Logistic)", "회귀 (Regression)"], horizontal=True)
            st.session_state.task = "logit" if "분류" in task_choice else "regression"
            
            # 2. 특징 변수 선택 (ID 등 제외)
            exclude_cols = st.multiselect("제외할 변수 (ID, 이름 등)", 
                                        options=[c for c in df_merged.columns if c != target_col])
            feature_cols = [c for c in df_merged.columns if c not in exclude_cols + [target_col]]
            
            # 3. 변환 옵션
            c1, c2 = st.columns(2)
            with c1:
                impute_method = st.selectbox("결측값 처리", ["중앙값(Median)", "평균(Mean)", "0으로 채움"])
            with c2:
                enc_method = st.selectbox("범주형 변수 처리", ["Label Encoding", "One-Hot Encoding"])

            if st.button("⚡ 전처리 실행 (변환)", type="primary", key="run_preprocess"):
                with st.spinner("데이터 변환 중..."):
                    try:
                        X = df_merged[feature_cols].copy()
                        y = df_merged[target_col].copy()
                        
                        # 수치형/범주형 분리
                        num_cols = X.select_dtypes(include=np.number).columns
                        cat_cols = X.select_dtypes(exclude=np.number).columns
                        
                        # 결측값 처리
                        imp_strat = "median" if "중앙값" in impute_method else "mean" if "평균" in impute_method else "constant"
                        imputer = SimpleImputer(strategy=imp_strat, fill_value=0)
                        
                        if len(num_cols) > 0:
                            X[num_cols] = imputer.fit_transform(X[num_cols])
                            scaler = StandardScaler()
                            X[num_cols] = scaler.fit_transform(X[num_cols])
                        else:
                            scaler = None

                        # 인코딩
                        encoders = {}
                        for col in cat_cols:
                            X[col] = X[col].fillna("Unknown").astype(str)
                            if "Label" in enc_method:
                                le = LabelEncoder()
                                X[col] = le.fit_transform(X[col])
                                encoders[col] = le
                            else:
                                # One-Hot (간소화)
                                le = LabelEncoder()
                                X[col] = le.fit_transform(X[col])
                                encoders[col] = le
                        
                        # 세션 저장
                        st.session_state.preprocess.update({
                            "imputer": imputer, "scaler": scaler, "encoders": encoders,
                            "feature_cols": list(X.columns)
                        })
                        st.session_state.data["X_processed"] = X
                        st.session_state.data["y_processed"] = y
                        
                        st.success("✅ 전처리 완료! 이제 옆의 [Stepwise 탭]에서 변수 선택 과정을 확인하세요.")
                        
                    except Exception as e:
                        st.error(f"오류 발생: {e}")

        # =========================================================
        # Tab 2: Stepwise 변수 선택 (프로세스 시각화 핵심)
        # =========================================================
        with tab_stepwise:
            st.markdown("### 🧬 Stepwise 변수 선택 과정")
            st.caption("알고리즘이 가장 중요한 변수부터 하나씩 추가하며 모델 성능 변화를 추적합니다.")
            
            if "X_processed" not in st.session_state.data:
                st.warning("⚠️ [기본 전처리] 탭에서 전처리를 먼저 수행해주세요.")
            else:
                X = st.session_state.data["X_processed"]
                y = st.session_state.data["y_processed"]
                
                col_conf1, col_conf2 = st.columns([1, 3])
                with col_conf1:
                    max_steps = st.number_input("최대 선택 변수 수", min_value=1, max_value=len(X.columns), value=min(10, len(X.columns)))
                    start_btn = st.button("🚀 변수 선택 시작", type="primary")
                
                if start_btn:
                    # 진행 상태 표시 공간
                    chart_placeholder = st.empty()
                    log_placeholder = st.empty()
                    
                    # 모델 설정
                    if st.session_state.task == "logit":
                        model = LogisticRegression(solver='liblinear')
                        metric_name = "Accuracy"
                    else:
                        model = LinearRegression()
                        metric_name = "R2 Score"
                    
                    # Stepwise (Forward Selection) 로직
                    selected_features = []
                    candidates = list(X.columns)
                    history = []
                    
                    # 진행률 바
                    pbar = st.progress(0)
                    
                    for step in range(max_steps):
                        best_score = -np.inf
                        best_feature = None
                        
                        # 남은 변수 중 하나씩 넣어보며 테스트
                        for feature in candidates:
                            current_subset = selected_features + [feature]
                            # 속도를 위해 단순 train/test split으로 평가
                            X_sub = X[current_subset]
                            X_tr, X_val, y_tr, y_val = train_test_split(X_sub, y, test_size=0.3, random_state=42)
                            
                            model.fit(X_tr, y_tr)
                            score = model.score(X_val, y_val)
                            
                            if score > best_score:
                                best_score = score
                                best_feature = feature
                        
                        if best_feature:
                            selected_features.append(best_feature)
                            candidates.remove(best_feature)
                            history.append({
                                "Step": step + 1,
                                "변수명": best_feature,
                                "Score": best_score,
                                "누적 변수": ", ".join(selected_features)
                            })
                            
                            # 실시간 그래프 업데이트
                            hist_df = pd.DataFrame(history)
                            fig = px.line(hist_df, x="Step", y="Score", markers=True, 
                                          title=f"변수 추가에 따른 성능 변화 ({metric_name})",
                                          text="변수명")
                            fig.update_traces(textposition="top center")
                            fig.update_layout(height=350)
                            chart_placeholder.plotly_chart(fig, use_container_width=True)
                            
                            # 실시간 로그 업데이트
                            log_placeholder.dataframe(
                                hist_df[["Step", "변수명", "Score"]].sort_values("Step", ascending=False), 
                                use_container_width=True
                            )
                            
                            pbar.progress((step + 1) / max_steps)
                        else:
                            break
                    
                    pbar.empty()
                    st.success("✅ 탐색 완료!")
                    
                    # 최종 적용 버튼
                    st.divider()
                    st.markdown("#### 🎯 최종 결정")
                    
                    col_final1, col_final2 = st.columns([3, 1])
                    with col_final1:
                        final_k = st.slider("상위 몇 개의 변수를 최종 모델에 사용하시겠습니까?", 
                                          1, len(history), len(history))
                    with col_final2:
                        if st.button("이 조합으로 확정"):
                            final_features = [h["변수명"] for h in history[:final_k]]
                            st.session_state.preprocess["feature_cols"] = final_features
                            st.session_state.data["X_processed"] = X[final_features]
                            st.success(f"{len(final_features)}개 변수가 선택되었습니다. '모델 학습' 단계로 이동하세요!")

        # =========================================================
        # Tab 2: Stepwise 변수 선택 (프로세스 시각화)
        # =========================================================
        with tab_stepwise:
            st.markdown("### 🧬 Stepwise (Forward Selection) 프로세스")
            st.info("💡 **전처리 실행 후** 사용 가능합니다. 알고리즘이 변수를 하나씩 추가하며 성능 변화를 추적합니다.")
            
            if "X_processed" not in st.session_state.data or st.session_state.data["X_processed"] is None:
                st.warning("⚠️ 먼저 '기본 전처리' 탭에서 전처리를 실행해주세요.")
            else:
                X = st.session_state.data["X_processed"]
                y = st.session_state.data["y_processed"]
                current_features = list(X.columns)
                
                col_s1, col_s2 = st.columns([1, 3])
                with col_s1:
                    st.write(f"**전체 변수 후보**: {len(current_features)}개")
                    max_vars = st.slider("최대 선택 변수 개수", 1, len(current_features), min(10, len(current_features)))
                    start_stepwise = st.button("🚀 Stepwise 시작")
                
                with col_s2:
                    if start_stepwise:
                        # Stepwise 로직 (간소화된 Forward Selection)
                        selected_features = []
                        remaining_features = current_features.copy()
                        history = []  # 그래프용 기록
                        
                        placeholder_chart = st.empty() # 차트 갱신용
                        placeholder_text = st.empty()  # 텍스트 갱신용
                        
                        # 모델 설정
                        if st.session_state.task == "logit":
                            est = LogisticRegression(max_iter=500, solver='liblinear')
                            scoring_metric = "Accuracy"
                        else:
                            est = LinearRegression()
                            scoring_metric = "R2 Score"
                            
                        # 반복 시작
                        progress_bar = st.progress(0)
                        
                        for i in range(max_vars):
                            best_score = -np.inf
                            best_feature = None
                            
                            # 남은 변수들 중 가장 좋은 것 하나 찾기
                            for feature in remaining_features:
                                trial_features = selected_features + [feature]
                                X_subset = X[trial_features]
                                
                                # 간단한 검증 (Train set 내에서 split)
                                X_tr, X_val, y_tr, y_val = train_test_split(X_subset, y, test_size=0.2, random_state=42)
                                est.fit(X_tr, y_tr)
                                score = est.score(X_val, y_val)
                                
                                if score > best_score:
                                    best_score = score
                                    best_feature = feature
                            
                            # 결과 기록
                            if best_feature:
                                selected_features.append(best_feature)
                                remaining_features.remove(best_feature)
                                history.append({"Step": i+1, "Added Feature": best_feature, "Score": best_score})
                                
                                # 실시간 업데이트
                                placeholder_text.markdown(f"**Step {i+1}**: `+ {best_feature}` 추가됨 (Score: {best_score:.4f})")
                                progress_bar.progress((i + 1) / max_vars)
                                
                                # 실시간 차트 그리기
                                df_hist = pd.DataFrame(history)
                                fig = px.line(df_hist, x="Step", y="Score", markers=True, 
                                              title=f"Stepwise Selection Process ({scoring_metric})",
                                              hover_data=["Added Feature"])
                                placeholder_chart.plotly_chart(fig, use_container_width=True)
                        
                        st.success("✅ 변수 선택 완료!")
                        
                        # 결과 테이블 표시
                        st.markdown("#### 📜 선택 과정 상세")
                        st.dataframe(pd.DataFrame(history), use_container_width=True)
                        
                        # 최종 적용 버튼
                        st.markdown("---")
                        if st.button("🎯 이 변수 조합 적용하기 (모델 학습에 사용)"):
                            st.session_state.preprocess["feature_cols"] = selected_features
                            st.session_state.data["X_processed"] = X[selected_features]
                            st.success(f"✅ 상위 {len(selected_features)}개 변수가 적용되었습니다! 이제 '모델 학습' 단계로 이동하세요.")
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
app.py
