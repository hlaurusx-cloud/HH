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
from sklearn.linear_model import LinearRegression, LogisticRegression  # 回归分析核心模型
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # 决策树核心模型
from sklearn.metrics import (
    accuracy_score, auc, roc_curve, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")


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
    
    ### 왼쪽「데이터 업로드」를 클릭하여 사용을 시작하세요！
    """)

# ----------------------
# 단계 1：데이터 업로드（단일 파일 또는 기본 파일）
# ----------------------
elif st.session_state.step == 1:
    st.subheader("📤 데이터 업로드")
    
    # 탭을 사용하여 '파일 업로드'와 '기본 데이터 사용'을 구분
    tab1, tab2 = st.tabs(["📂 내 파일 업로드", "💾 서버 기본 데이터 사용"])
    
    # --- 기능 1: 사용자가 직접 업로드 ---
    with tab1:
        st.markdown("지원 형식：CSV、Parquet、Excel（.xlsx/.xls）")
        uploaded_file = st.file_uploader("데이터 파일 선택", type=["csv", "parquet", "xlsx", "xls"], key="single_file")
    
    # --- 기능 2: 서버에 있는 기본 CSV 파일 로드 ---
    with tab2:
        # 여기에 지정하신 파일명을 입력했습니다.
        DEFAULT_FILE_PATH = "combined_loan_data.csv" 
        
        st.info(f"💡 **기본 데이터 설명**: 대출 관련 통합 데이터 (`{DEFAULT_FILE_PATH}`)")
        
        # 버튼 클릭 시 처리
        if st.button("기본 데이터 불러오기 (combined_loan_data.csv)", type="primary"):
            if os.path.exists(DEFAULT_FILE_PATH):
                # 파일을 읽어서 세션에 저장
                try:
                    df_default = pd.read_csv(DEFAULT_FILE_PATH)
                    st.session_state.data["merged"] = df_default
                    st.success(f"✅ 기본 데이터 로드 성공! ({len(df_default):,} 행)")
                    st.rerun()  # 데이터 로드 후 화면 갱신
                except Exception as e:
                    st.error(f"파일 읽기 오류: {e}")
            else:
                st.error(f"⚠️ 파일을 찾을 수 없습니다: {DEFAULT_FILE_PATH} (파일이 app.py와 같은 폴더에 있는지 확인하세요)")

    # --- 데이터가 로드된 상태인지 확인 및 표시 ---
    if st.session_state.data.get("merged") is not None:
        df_merged = st.session_state.data["merged"]
        
        st.divider()
        st.markdown(f"### ✅ 현재 로드된 데이터 ({len(df_merged):,} 행)")
        st.dataframe(df_merged.head(5), use_container_width=True)
        
        # 데이터 기본 정보 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**열 이름 (상위 10개)**")
            st.write(", ".join(df_merged.columns.tolist()[:10]) + ("..." if len(df_merged.columns) > 10 else ""))
        with col2:
            st.write("**결측값 총 개수**")
            st.write(f"{df_merged.isnull().sum().sum()} 개")
        with col3:
            st.write("**데이터 유형**")
            st.write(df_merged.dtypes.value_counts().to_string())
        
        # 업로드/로드 로직 처리 (이미 위에서 처리됨)
        
        # 다음 단계 안내
        st.divider()
        st.info("📊 데이터 탐색을 위해 왼쪽 사이드바에서 **「데이터 시각화」** 단계로 이동하세요")
        
# ----------------------
# 단계 2：데이터 시각화（新增！히스토그램 기능 추가）
# ----------------------
elif st.session_state.step == 2:
    st.subheader("📊 데이터 시각화")
    
    if st.session_state.data["merged"] is None:
        st.warning("먼저「데이터 업로드」단계를 완료하세요")
    else:
        df = st.session_state.data["merged"]
        
        # 1. 변수 유형 자동识别
        st.markdown("### 변수 선택")
        # 범주형 변수（object, category）
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        # 수치형 변수（int64, float64）
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        #  변수 선택 박스（선택 가능하도록）
        col1, col2, col3 = st.columns(3)
        with col1:
            x_var = st.selectbox("📋 X축：범주형 변수（선택 사항）", options=["선택 안 함"] + cat_cols, index=0)
            # X축이 "선택 안 함"인 경우 None 처리
            x_var = None if x_var == "선택 안 함" else x_var
        with col2:
            y_var = st.selectbox("📈 Y축：수치형 변수（필수）", options=num_cols, index=0 if num_cols else None, disabled=not num_cols)
        with col3:
            # 그래프 타입 선택（新增 히스토그램选项）
            graph_types = [
                "막대 그래프（평균값）", 
                "박스 플롯（분포）", 
                "바이올린 플롯（분포+밀도）",
                "산점도（개별 데이터）",
                "선 그래프（추세）",
                "히스토그램（분포）"  # 新增：히스토그램
            ]
            graph_type = st.selectbox("📊 그래프 유형", options=graph_types, index=0)
        
        # 2. 그래프 그리기（新增 히스토그램绘制逻辑）
        st.divider()
        if y_var:  # Y축（수치형 변수）만 있어도 히스토그램 가능
            if graph_type == "히스토그램（분포）":
                st.markdown(f"### {y_var} 분포（히스토그램）")
                
                # 그래프 데이터 준비（결측값 제거）
                plot_df = df[[y_var] + ([x_var] if x_var else [])].dropna()
                
                try:
                    # 히스토그램 옵션（구간 개수 조정）
                    bins = st.slider("히스토그램 구간 개수", min_value=10, max_value=100, value=30, step=5)
                    
                    # X축（범주형 변수） 선택 여부에 따라 그래프 분기
                    if x_var:  # 按类别分组的 히스토그램
                        fig = px.histogram(
                            plot_df, 
                            x=y_var,
                            color=x_var,  # 按类别区分颜色
                            barmode="overlay",  # 重叠显示
                            opacity=0.7,
                            nbins=bins,
                            title=f"{x_var}별 {y_var} 분포",
                            labels={y_var: y_var, x_var: x_var},
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                    else:  # 单变量 히스토그램
                        fig = px.histogram(
                            plot_df,
                            x=y_var,
                            nbins=bins,
                            title=f"{y_var} 전체 분포",
                            labels={y_var: y_var, "count": "빈도수"},
                            color_discrete_sequence=["#636EFA"],
                            marginal="box"  # 边缘添加 박스 플롯（분포 정보 강화）
                        )
                    
                    # 그래프 스타일 최적화
                    fig.update_layout(
                        width=1200, height=600,
                        xaxis_title_font=dict(size=14),
                        yaxis_title_font=dict(size=14),
                        title_font=dict(size=16, weight="bold")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 히스토그램 통계 정보
                    st.markdown("### 📋 분포 통계 정보")
                    stats = plot_df[y_var].describe().round(3)
                    stats_df = pd.DataFrame({
                        "통계량": ["개수", "평균값", "표준편차", "최소값", "제1사분위수", "중앙값", "제3사분위수", "최대값"],
                        "값": [
                            stats["count"], stats["mean"], stats["std"],
                            stats["min"], stats["25%"], stats["50%"],
                            stats["75%"], stats["max"]
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"히스토그램 생성 실패：{str(e)}")
            
            # 기존 그래프逻辑（保持不变）
            else:
                if not x_var:
                    st.warning("막대 그래프/박스 플롯/바이올린 플롯/산점도/선 그래프는 X축（범주형 변수）를 선택해야 합니다")
                    st.stop()
                
                st.markdown(f"### {x_var} vs {y_var} ({graph_type.split('（')[0]})")
                plot_df = df[[x_var, y_var]].dropna()
                
                try:
                    if graph_type == "막대 그래프（평균값）":
                        bar_data = plot_df.groupby(x_var)[y_var].mean().reset_index()
                        fig = px.bar(
                            bar_data, x=x_var, y=y_var, 
                            title=f"{x_var}별 {y_var} 평균값",
                            labels={y_var: f"{y_var} 평균값", x_var: x_var},
                            color=x_var, color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                    
                    elif graph_type == "박스 플롯（분포）":
                        fig = px.box(
                            plot_df, x=x_var, y=y_var,
                            title=f"{x_var}별 {y_var} 분포",
                            labels={y_var: y_var, x_var: x_var},
                            color=x_var, color_discrete_sequence=px.colors.qualitative.Set2
                        )
                    
                    elif graph_type == "바이올린 플롯（분포+밀도）":
                        fig = px.violin(
                            plot_df, x=x_var, y=y_var,
                            title=f"{x_var}별 {y_var} 분포 및 밀도",
                            labels={y_var: y_var, x_var: x_var},
                            color=x_var, box=True,
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                    
                    elif graph_type == "산점도（개별 데이터）":
                        fig = px.scatter(
                            plot_df, x=x_var, y=y_var,
                            title=f"{x_var} vs {y_var} 개별 데이터 분포",
                            labels={y_var: y_var, x_var: x_var},
                            color=x_var, opacity=0.6,
                            color_discrete_sequence=px.colors.qualitative.Vivid
                        )
                    
                    elif graph_type == "선 그래프（추세）":
                        line_data = plot_df.groupby(x_var)[y_var].mean().reset_index()
                        fig = px.line(
                            line_data, x=x_var, y=y_var,
                            title=f"{x_var}별 {y_var} 추세",
                            labels={y_var: y_var, x_var: x_var},
                            color_discrete_sequence=["#1f77b4"],
                            markers=True
                        )
                    
                    fig.update_layout(
                        width=1200, height=600,
                        xaxis_title_font=dict(size=14),
                        yaxis_title_font=dict(size=14),
                        title_font=dict(size=16, weight="bold")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 기존 통계 정보
                    st.markdown("### 📋 통계 정보")
                    # agg 대신 describe()를 사용하여 안전하게 통계량 추출
                    stats_desc = plot_df.groupby(x_var)[y_var].describe()
                    
                    # 필요한 열만 선택 및 순서 정렬
                    stats_df = stats_desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(3)
                    
                    # 한국어 컬럼명 변경
                    stats_df.columns = ["데이터 개수", "평균값", "표준편차", "최소값", "제1사분위수", "중앙값", "제3사분위수", "최대값"]
                    
                    st.dataframe(stats_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"그래프 생성 실패：{str(e)}")
        else:
            st.warning("Y축（수치형 변수）를 선택해야 합니다")
        
        # 下一步 안내
        st.divider()
        st.info("🔧 데이터 전처리를 위해 왼쪽 사이드바에서「데이터 전처리」단계로 이동하세요")

# ----------------------
# 3. 데이터 전처리（Step 3） - 컬럼명 중복/MultiIndex 오류 해결 버전
# ----------------------
elif st.session_state.step == 3:
    st.subheader("🛠 데이터 전처리 및 변수 선택 (Smart Stepwise)")
    
    # 1. 원본 데이터 로드
    df_raw = st.session_state.data["merged"]
    if df_raw is None:
        st.error("데이터가 없습니다. Step 1에서 데이터를 업로드하세요.")
        st.stop()

    # [오류 해결 핵심] 컬럼명 정리 (중복 제거 및 MultiIndex 병합)
    # 1) MultiIndex(여러 줄 헤더)일 경우 하나로 합치기
    if isinstance(df_raw.columns, pd.MultiIndex):
        st.warning("⚠️ 다중 헤더(MultiIndex)가 감지되어 단일 헤더로 병합합니다.")
        df_raw.columns = ['_'.join(map(str, col)).strip() for col in df_raw.columns.values]
    
    # 2) 컬럼명 중복 제거 (예: A, A -> A, A_1)
    if df_raw.columns.has_duplicates:
        st.warning("⚠️ 중복된 컬럼명이 감지되어 이름을 변경합니다 (예: Col -> Col_1).")
        new_columns = []
        seen = {}
        for col in df_raw.columns:
            col_str = str(col)
            if col_str in seen:
                seen[col_str] += 1
                new_columns.append(f"{col_str}_{seen[col_str]}")
            else:
                seen[col_str] = 0
                new_columns.append(col_str)
        df_raw.columns = new_columns
        # 정리된 데이터 세션에 다시 저장
        st.session_state.data["merged"] = df_raw

    # -------------------------------------------------------
    # [1] 타겟 변수 우선 선택 (스마트 필터링 적용)
    # -------------------------------------------------------
    st.markdown("### 1️⃣ 타겟 변수(예측 목표) 설정")
    
    # 타겟 후보군 필터링 (ID나 상수 제외)
    target_candidates = []
    dropped_candidates = [] 

    for col in df_raw.columns:
        # 조건 1: 모든 값이 다 다른 경우 (ID일 확률 높음) -> 50행 이상일 때만 체크
        if len(df_raw) > 50 and df_raw[col].nunique() == len(df_raw):
            dropped_candidates.append(col)
            continue
        # 조건 2: 값이 하나밖에 없는 경우 (상수)
        if df_raw[col].nunique() <= 1:
            dropped_candidates.append(col)
            continue
        target_candidates.append(col)
    
    # 만약 필터링 결과 남은게 없으면 원본 전체 사용
    if not target_candidates:
        target_candidates = df_raw.columns.tolist()

    # 세션 상태 초기화
    if "target_col_temp" not in st.session_state:
        st.session_state.target_col_temp = target_candidates[0]
    
    # 이전에 선택한 타겟이 목록에 없으면 리셋
    if st.session_state.target_col_temp not in target_candidates:
         st.session_state.target_col_temp = target_candidates[0]

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        target_col = st.selectbox(
            "예측할 타겟 컬럼 선택", 
            options=target_candidates,
            index=target_candidates.index(st.session_state.target_col_temp),
            key="target_selector"
        )
    with col_t2:
        if dropped_candidates:
            with st.popover("🗑 제외된 컬럼 보기"):
                st.write("ID 또는 상수로 판단되어 목록에서 제외됨:")
                st.write(dropped_candidates)

    st.session_state.target_col_temp = target_col
    st.session_state.preprocess["target_col"] = target_col

    st.divider()

    # -------------------------------------------------------
    # [2] 스마트 변수 선택 (Stepwise)
    # -------------------------------------------------------
    st.markdown("### 2️⃣ 변수 선택 (Stepwise)")
    
    # 선택된 변수 리스트 초기화
    if "selected_features_temp" not in st.session_state:
        st.session_state.selected_features_temp = [c for c in target_candidates if c != target_col]

    col_tool1, col_tool2 = st.columns([1, 3])
    
    with col_tool1:
        st.write("") 
        if st.button("🤖 AI 스마트 변수 선택\n(Stepwise 실행)", type="primary", use_container_width=True):
            with st.spinner("데이터 분석 중..."):
                try:
                    # 분석용 임시 데이터 복사
                    temp_df = df_raw.copy()
                    
                    # 1. 수치형 처리: NaN을 0으로 채움
                    num_temp = temp_df.select_dtypes(include=[np.number]).columns
                    if len(num_temp) > 0:
                        temp_df[num_temp] = temp_df[num_temp].fillna(0)
                    
                    # 2. 범주형 처리: NaN을 "unknown"으로 채우고 숫자 변환
                    cat_temp = temp_df.select_dtypes(exclude=[np.number]).columns
                    for c in cat_temp:
                        # 컬럼명 안전하게 처리 (문자열 변환)
                        temp_df[c] = temp_df[c].fillna("unknown").astype(str)
                        temp_df[c] = pd.factorize(temp_df[c])[0]
                    
                    # 3. X, y 분리
                    X_temp = temp_df.drop(columns=[target_col], errors='ignore')
                    # X에서도 유효한 컬럼만 남김
                    valid_features = [c for c in X_temp.columns if c in target_candidates]
                    X_temp = X_temp[valid_features]

                    y_temp = temp_df[target_col]
                    
                    # 4. 모델 중요도 산출
                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                    from sklearn.feature_selection import SelectFromModel

                    is_classification = False
                    if st.session_state.task == "logit":
                        is_classification = True
                    # 타겟 값의 종류가 적거나 문자열이면 분류로 간주
                    elif y_temp.dtype == 'object' or len(y_temp.unique()) < 20:
                        is_classification = True
                    
                    if is_classification:
                        model_sel = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                        if y_temp.dtype == 'object': 
                             y_temp = pd.factorize(y_temp)[0]
                    else:
                        model_sel = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

                    model_sel.fit(X_temp, y_temp)
                    
                    # 5. 중요도 평균 이상인 변수 선택
                    selector = SelectFromModel(model_sel, prefit=True, threshold="mean")
                    selected_indices = selector.get_support(indices=True)
                    recommended_features = X_temp.columns[selected_indices].tolist()
                    
                    st.session_state.selected_features_temp = recommended_features
                    st.success(f"✅ 분석 완료! {len(recommended_features)}개 중요 변수 선택됨.")
                    st.rerun()

                except Exception as e:
                    st.error(f"분석 오류 발생: {str(e)}")
                    st.write("힌트: 데이터의 컬럼명이 중복되었거나 비정상적인 값이 있을 수 있습니다.")

    with col_tool2:
        # 멀티 셀렉트 박스
        feature_options = [c for c in target_candidates if c != target_col]
        
        feature_cols = st.multiselect(
            "분석에 사용할 변수 (자동 선택됨)",
            options=feature_options,
            default=[c for c in st.session_state.selected_features_temp if c in feature_options],
            key="feature_multiselect"
        )

    if not feature_cols:
        st.warning("⚠️ 최소한 하나의 변수는 선택해야 합니다.")
        st.stop()
        
    st.session_state.preprocess["feature_cols"] = feature_cols
    
    # -------------------------------------------------------
    # [3] 전처리 상세 설정
    # -------------------------------------------------------
    st.divider()
    with st.expander("⚙️ 고급 전처리 설정 (결측치/인코딩)", expanded=False):
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            impute_strategy = st.selectbox("결측치 처리", ["중앙값(Median)", "평균값(Mean)", "0으로 채우기", "최빈값(Mode)"])
        with col_opt2:
            cat_encoding = st.selectbox("인코딩 방식", ["Label Encoding", "One-Hot Encoding"])

    strategy_map = {"중앙값(Median)": "median", "평균값(Mean)": "mean", "0으로 채우기": "constant", "최빈값(Mode)": "most_frequent"}
    
    # -------------------------------------------------------
    # [4] 전처리 실행
    # -------------------------------------------------------
    st.divider()
    if st.button("🚀 전처리 실행 및 데이터 생성", type="primary", use_container_width=True):
        try:
            final_cols = feature_cols + [target_col]
            df_final = df_raw[final_cols].copy()
            
            X = df_final[feature_cols]
            y = df_final[target_col]

            num_cols = X.select_dtypes(include=[np.number]).columns
            cat_cols = X.select_dtypes(exclude=[np.number]).columns

            imputer_args = {"strategy": strategy_map[impute_strategy]}
            if strategy_map[impute_strategy] == "constant":
                imputer_args["fill_value"] = 0
                
            imputer = SimpleImputer(**imputer_args)
            
            # 수치형 처리
            if len(num_cols) > 0:
                X[num_cols] = imputer.fit_transform(X[num_cols])
                scaler = StandardScaler()
                X[num_cols] = scaler.fit_transform(X[num_cols])
            else:
                scaler = None

            # 범주형 처리
            encoders = {}
            for col in cat_cols:
                X[col] = X[col].fillna("unknown").astype(str)
                if "Label" in cat_encoding:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    encoders[col] = le
                else:
                    ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown='ignore')
                    ohe_data = ohe.fit_transform(X[[col]])
                    # 컬럼명 생성 시 특수문자 제거 등 안전장치
                    new_cols = [f"{col}_{str(c).replace(' ', '_')}" for c in ohe.categories_[0][1:]]
                    X_ohe = pd.DataFrame(ohe_data, columns=new_cols, index=X.index)
                    X = pd.concat([X.drop(columns=[col]), X_ohe], axis=1)
                    encoders[col] = (ohe, new_cols)

            st.session_state.preprocess["imputer"] = imputer
            st.session_state.preprocess["scaler"] = scaler
            st.session_state.preprocess["encoders"] = encoders
            st.session_state.preprocess["feature_cols"] = list(X.columns)
            
            st.session_state.data["X_processed"] = X
            st.session_state.data["y_processed"] = y
            
            st.success("데이터 전처리 완료!")
            st.dataframe(X.head(3), use_container_width=True)

        except Exception as e:
            st.error(f"전처리 최종 실행 오류: {e}")
            
            
# ----------------------
# 단계 4：모델 학습（修复 stratify 参数错误）
# ----------------------
elif st.session_state.step == 4:
    st.subheader("🚀 하이브리드모형 학습（회귀 분석 + 의사결정나무）")
    
    # 전처리 완료 여부 확인
    if "X_processed" not in st.session_state.data or "y_processed" not in st.session_state.data:
        st.warning("먼저「데이터 전처리」단계를 완료하세요")
    else:
        X = st.session_state.data["X_processed"]
        y = st.session_state.data["y_processed"]
        
        # ----------------------
        # 核心修复：stratify 参数有效性校验
        # ----------------------
        st.markdown("### 학습 설정")
        test_size = st.slider("테스트集 비율", min_value=0.1, max_value=0.3, value=0.2, step=0.05)
        
        # stratify 사용 여부 결정（分类任务且目标变量类别数≥2时才使用）
        stratify_param = None
        if st.session_state.task == "logit":  # 分类任务
            y_unique_count = y.nunique()  # 目标变量唯一值数量
            if y_unique_count >= 2:
                # 进一步检查每个类别的样本数是否≥1
                y_value_counts = y.value_counts()
                if (y_value_counts >= 1).all():
                    stratify_param = y
                    st.info(f"✅分层抽样 적용：目标变量에 {y_unique_count} 个类别 존재（样本数：{y_value_counts.to_dict()}）")
                else:
                    st.warning(f"⚠️ 일부类别样本数为0，分层抽样禁用（自动转为普通随机抽样）")
            else:
                st.warning(f"⚠️ 目标变量只有 {y_unique_count} 个类别，分层抽样禁用（自动转为普通随机抽样）")
        else:
            st.info("ℹ️ 回归任务不支持分层抽样，使用普通随机抽样")
        
        # 数据 분할（修复后：根据校验结果决定是否使用 stratify）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, 
            stratify=stratify_param  # 校验后的参数
        )
        
        # 모델 선택（작업 유형에 따라）
        if st.session_state.task == "logit":  # 分类任务：로지스틱 회귀（회귀분석）+ 분류 의사결정나무
            reg_model = LogisticRegression(max_iter=1000)  # 分类用 회귀분석（로지스틱）
            dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)  # 分类 의사결정나무
        else:  # 回归任务：선형 회귀（회귀분석）+ 회귀 의사결정나무
            reg_model = LinearRegression()  # 回归用 회귀분석（선형）
            dt_model = DecisionTreeRegressor(random_state=42, max_depth=10)  # 回归 의사결정나무
        
        # 모델 학습
        if st.button("모델 학습 시작"):
            with st.spinner("모델 학습 중..."):
                try:
                    # 단일 모델 학습
                    reg_model.fit(X_train, y_train)
                    dt_model.fit(X_train, y_train)
                    
                    # 모델 저장
                    st.session_state.models["regression"] = reg_model
                    st.session_state.models["decision_tree"] = dt_model
                    
                    # 학습集/테스트集 저장
                    st.session_state.data["X_train"] = X_train
                    st.session_state.data["X_test"] = X_test
                    st.session_state.data["y_train"] = y_train
                    st.session_state.data["y_test"] = y_test
                    
                    st.success("모델 학습 완료！")
                    st.markdown("✅ 학습된 모델：")
                    st.markdown("- 회귀 분석（로지스틱/선형，해석력 강함）")
                    st.markdown("- 의사결정나무（분류/회귀，정확도 높음）")
                    st.markdown("- 하이브리드모형（전两者 가중融合）")
                    
                    # 训练集/测试集 정보 표시
                    st.markdown(f"📊 학습集：{len(X_train):,} 행 | 테스트集：{len(X_test):,} 행")
                    if st.session_state.task == "logit":
                        st.markdown(f"🎯 训练集类别分布：{y_train.value_counts().to_dict()}")
                        st.markdown(f"🎯 测试集类别分布：{y_test.value_counts().to_dict()}")
                except Exception as e:
                    st.error(f"모델 학습 실패：{str(e)}")

# -------------------------- 단계 5: 혼합 모델 예측 (완성 버전)--------------------------
def predict(input_data):
    """
    혼합 모델 예측 함수: 선형 회귀와 의사결정 트리의 예측 결과를 가중치에 따라 융합
    input_data: 전처리가 완료된 입력 데이터 (DataFrame)
    return: 최종 혼합 예측 결과, 예측 확률 (분류 작업 시 유효, 회귀 작업 시 None)
    """
    # 1. 모델 학습 시 사용한 특징 열 추출 (session_state에서 가져오기, 특징 차원 불일치 방지)
    feature_cols = st.session_state.get("feature_cols", [])
    if not feature_cols:
        st.error("특징 열을 찾을 수 없습니다! 먼저 모델 학습을 완료해주세요")
        return None, None
    
    # 입력 데이터에서 학습 시 사용한 특징만 남기기 (차원 오류 방지)
    X = input_data[feature_cols].copy()
    
    # 2. session_state에서 학습 완료된 모델과 가중치 로드 (이전 학습 단계에서 저장한 값)
    models = st.session_state.get("models", {})
    if not models or "regression" not in models or "decision_tree" not in models:
        st.error("모델이 학습되지 않았습니다! 먼저 모델 학습 단계를 완료해주세요")
        return None, None
    
    # 3. 두 모델의 융합 가중치 가져오기 (이전 학습 단계에서 session_state에 저장)
    reg_weight = models["mixed_weights"]["regression"]  # 선형 회귀 가중치
    dt_weight = models["mixed_weights"]["decision_tree"]  # 의사결정 트리 가중치
    
    # 4. 두 모델로 각각 예측 (분류/회귀 작업 구분)
    if st.session_state.task == "logit":  # 👉 분류 작업 (예측 확률 + 클래스)
        # 분류 모델은 확률 반환 (predict_proba): 2번째 열(인덱스 1)을 양성 클래스 확률로 사용
        reg_prob = models["regression"].predict_proba(X)[:, 1]  # 선형 회귀 양성 클래스 확률
        dt_prob = models["decision_tree"].predict_proba(X)[:, 1]  # 의사결정 트리 양성 클래스 확률
        
        # 가중치에 따른 확률 융합 (가중 평균)
        mixed_prob = reg_weight * reg_prob + dt_weight * dt_prob
        # 확률을 클래스로 변환 (임계값 0.5, 필요에 따라 조정 가능)
        mixed_pred = (mixed_prob > 0.5).astype(int)
        
        # 반환: 예측 클래스 (0/1), 예측 확률 (0-1)
        return mixed_pred, mixed_prob
    
    else:  # 👉 회귀 작업 (연속값 예측)
        # 회귀 모델은 직접 예측값 반환 (predict)
        reg_pred = models["regression"].predict(X)  # 선형 회귀 예측값
        dt_pred = models["decision_tree"].predict(X)  # 의사결정 트리 예측값
        
        # 가중치에 따른 예측값 융합 (가중 평균)
        mixed_pred = reg_weight * reg_pred + dt_weight * dt_pred
        
        # 회귀 작업은 확률이 없으므로, 예측값과 None 반환
        return mixed_pred, None


# -------------------------- (선택 사항) 예측 결과 호출 예시 (UI 로직에 맞게 조정)--------------------------
# UI에서 예측을触发하려면 아래 로직을 추가/수정하세요 (기존 버튼과 흐름에 맞춰调整)
if "models" in st.session_state and st.button("예측 시작"):
    # 전처리가 완료된 데이터 가져오기 (실제 전처리 후 데이터 변수명으로替换)
    input_data = st.session_state.get("preprocessed_data", None)
    if input_data is not None and not input_data.empty:
        pred_result, pred_prob = predict(input_data)
        
        # 예측 결과 표시 (작업 유형에 맞게调整)
        st.subheader("예측 결과")
        if st.session_state.task == "logit":
            # 분류 결과: 예측 클래스 + 양성 클래스 확률 표시
            input_data["예측 클래스"] = pred_result
            input_data["양성 클래스 확률"] = pred_prob.round(3)
            st.dataframe(input_data[["예측 클래스", "양성 클래스 확률"] + feature_cols], use_container_width=True)
        else:
            # 회귀 결과: 예측값 표시
            input_data["예측값"] = pred_result.round(3)
            st.dataframe(input_data[["예측값"] + feature_cols], use_container_width=True)
    else:
        st.warning("먼저 데이터를 업로드하고 전처리를 완료해주세요!")
# ----------------------
# 단계 6：성능 평가（하이브리드모형 vs 단일 모형）
# ----------------------
elif st.session_state.step == 6:
    st.subheader("📈 모델 성능 평가")
    
    if st.session_state.models["regression"] is None or st.session_state.models["decision_tree"] is None:
        st.warning("먼저「모델 학습」단계를 완료하세요")
    else:
        X_test = st.session_state.data["X_test"]
        y_test = st.session_state.data["y_test"]
        reg_model = st.session_state.models["regression"]
        dt_model = st.session_state.models["decision_tree"]
        reg_weight = st.session_state.models["mixed_weights"]["regression"]
        dt_weight = st.session_state.models["mixed_weights"]["decision_tree"]
        
        # 각 모델 예측 결과 계산
        if st.session_state.task == "logit":  # 분류任务 평가
            reg_pred = reg_model.predict(X_test)
            dt_pred = dt_model.predict(X_test)
            reg_proba = reg_model.predict_proba(X_test)[:, 1]
            dt_proba = dt_model.predict_proba(X_test)[:, 1]
            mixed_proba = reg_weight * reg_proba + dt_weight * dt_proba
            mixed_pred = (mixed_proba >= 0.5).astype(int)
            
            # 분류 지표 계산
            def calc_class_metrics(y_true, y_pred, y_proba):
                acc = accuracy_score(y_true, y_pred)
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc_score = auc(fpr, tpr)
                return {"정확도": acc, "AUC": auc_score}
            
            reg_metrics = calc_class_metrics(y_test, reg_pred, reg_proba)
            dt_metrics = calc_class_metrics(y_test, dt_pred, dt_proba)
            mixed_metrics = calc_class_metrics(y_test, mixed_pred, mixed_proba)
            
            metrics_df = pd.DataFrame({
                "모델": ["회귀 분석（로지스틱）", "의사결정나무（분류）", "하이브리드모형"],
                "정확도": [reg_metrics["정확도"], dt_metrics["정확도"], mixed_metrics["정확도"]],
                "AUC": [reg_metrics["AUC"], dt_metrics["AUC"], mixed_metrics["AUC"]]
            }).round(3)
        
        else:  # 회귀任务 평가
            reg_pred = reg_model.predict(X_test)
            dt_pred = dt_model.predict(X_test)
            mixed_pred = reg_weight * reg_pred + dt_weight * dt_pred
            
            # 회귀 지표 계산
            def calc_reg_metrics(y_true, y_pred):
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                return {"MAE": mae, "RMSE": rmse, "R²": r2}
            
            reg_metrics = calc_reg_metrics(y_test, reg_pred)
            dt_metrics = calc_reg_metrics(y_test, dt_pred)
            mixed_metrics = calc_reg_metrics(y_test, mixed_pred)
            
            metrics_df = pd.DataFrame({
                "모델": ["회귀 분석（선형）", "의사결정나무（회귀）", "하이브리드모형"],
                "MAE": [reg_metrics["MAE"], dt_metrics["MAE"], mixed_metrics["MAE"]],
                "RMSE": [reg_metrics["RMSE"], dt_metrics["RMSE"], mixed_metrics["RMSE"]],
                "R²": [reg_metrics["R²"], dt_metrics["R²"], mixed_metrics["R²"]]
            }).round(3)
        
        # 지표 비교 표시
        st.markdown("### 모델 성능 비교")
        st.dataframe(metrics_df, use_container_width=True)
        
        # 시각화 비교
        col1, col2 = st.columns(2)
        
        # logit（분류）작업 시각화
        if st.session_state.task == "logit":
            with col1:
                st.markdown("### ROC-AUC 곡선")
                fpr_reg, tpr_reg, _ = roc_curve(y_test, reg_proba)
                fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_proba)
                fpr_mixed, tpr_mixed, _ = roc_curve(y_test, mixed_proba)
                
                fig_auc = go.Figure()
                fig_auc.add_trace(go.Scatter(x=fpr_reg, y=tpr_reg, name=f"회귀 분석 (AUC={reg_metrics['AUC']:.3f})"))
                fig_auc.add_trace(go.Scatter(x=fpr_dt, y=tpr_dt, name=f"의사결정나무 (AUC={dt_metrics['AUC']:.3f})"))
                fig_auc.add_trace(go.Scatter(x=fpr_mixed, y=tpr_mixed, name=f"하이브리드모형 (AUC={mixed_metrics['AUC']:.3f})", line_dash="dash", line_width=3))
                fig_auc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="랜덤 추측", line_color="gray", line_dash="dot"))
                st.plotly_chart(fig_auc, use_container_width=True)
            
            with col2:
                st.markdown("### 혼동 행렬（하이브리드모형）")
                cm = confusion_matrix(y_test, mixed_pred)
                cm_df = pd.DataFrame(cm, index=["실제 음성", "실제 양성"], columns=["예측 음성", "예측 양성"])
                fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
                st.plotly_chart(fig_cm, use_container_width=True)
        
        # 의사결정나무（회귀）작업 시각화
        else:
            with col1:
                st.markdown("### 예측값 vs 실제값（하이브리드모형）")
                fig_pred = px.scatter(x=y_test, y=mixed_pred, title="실제값 vs 예측값", labels={"x": "실제값", "y": "예측값"})
                fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], line_color="red", name="이상적인 피팅 라인"))
                st.plotly_chart(fig_pred, use_container_width=True)
            
            with col2:
                st.markdown("### 잔차 그래프（하이브리드모형）")
                residuals = y_test - mixed_pred
                fig_res = px.scatter(x=mixed_pred, y=residuals, title="예측값 vs 잔차", labels={"x": "예측값", "y": "잔차"})
                fig_res.add_trace(go.Scatter(x=[mixed_pred.min(), mixed_pred.max()], y=[0, 0], line_color="red", name="잔차=0 라인"))
                st.plotly_chart(fig_res, use_container_width=True)
        
        # 모델 해석（특징 중요도：의사결정나무 기반）
        st.divider()
        st.markdown("### 모델 해석：핵심 특징 중요도")
        feature_importance = pd.DataFrame({
            "특징명": st.session_state.preprocess["feature_cols"],
            "중요도": dt_model.feature_importances_  # 의사결정나무의 특징 중요도
        }).sort_values("중요도", ascending=False).head(10)
        
        fig_importance = px.bar(feature_importance, x="중요도", y="특징명", orientation="h", color="중요도", color_continuous_scale="viridis")
        st.plotly_chart(fig_importance, use_container_width=True)
