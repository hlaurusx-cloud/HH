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
# 단계 3：데이터 전처리 (추가 수정: 타겟 변수 자동识别 및 경고)
# ----------------------
elif st.session_state.step == 3:
    st.subheader("🧹 데이터 전처리 & 변수 선택 (Final Fix)")
    
    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 먼저 '데이터 업로드' 단계를 완료하세요.")
    else:
        # 원본 데이터 로드
        df_origin = st.session_state.data["merged"].copy()
        all_cols = df_origin.columns.tolist()

        st.markdown("### 1️⃣ 분석 변수 설정")
        
        # 新增：自动识别非目标列（ID、索引等）
        def is_non_target_candidate(col):
            """判断列是否可能不适合作为目标变量（如ID、索引）"""
            keywords = ['id', 'index', '编号', '序号', 'key', '코드', '번호']  # 关键词列表
            return any(keyword in col.lower() for keyword in keywords)
        
        # 生成目标变量选项（标记不推荐的列）
        target_options = []
        for col in all_cols:
            if is_non_target_candidate(col):
                target_options.append(f"{col} ⚠️ (ID/索引列，不推荐)")
            else:
                target_options.append(col)
        
        col1, col2 = st.columns(2)
        with col1:
            target_display = st.selectbox("🎯 타겟 변수 (Y)", options=target_options)
            # 提取原始列名（去除标记）
            target_col = target_display.split(" ⚠️ ")[0]
        
        # 新增：如果选择了不推荐的列，显示警告
        if is_non_target_candidate(target_col):
            st.warning(f"⚠️ '{target_col}'는 ID/索引类列으로，目标变量(Y)로 사용하기 적합하지 않을 수 있습니다。\n请确认是否为预测할 타겟 값（如：销售额、是否违约等）。")
        
        feature_candidates = [c for c in all_cols if c != target_col]
        
        with col2:
            default_feats = feature_candidates[:10] if len(feature_candidates) > 10 else feature_candidates
            selected_features = st.multiselect(
                "📋 입력 변수 (X)",
                options=feature_candidates,
                default=default_feats
            )
        
        st.divider()

        if not selected_features:
            st.error("⚠️ 분석할 변수를 선택해주세요.")
        else:
            # 설정 저장
            st.session_state.preprocess["target_col"] = target_col
            
            tab1, tab2 = st.tabs(["⚡ 전처리 실행", "📊 중요도 분석"])
            
            with tab1:
                st.write(f"**Y(타겟) 결측치 제거** 및 **X(입력) 결측치 채우기**를 수행합니다.")
                
                if st.button("🚀 전처리 및 정제 시작", type="primary"):
                    with st.spinner("데이터 정제 중..."):
                        try:
                            # [핵심 1] 타겟(Y)이 비어있는 행 제거 (이게 없으면 NaN 에러 발생)
                            clean_df = df_origin.dropna(subset=[target_col]).reset_index(drop=True)
                            
                            dropped_count = len(df_origin) - len(clean_df)
                            if dropped_count > 0:
                                st.warning(f"⚠️ 타겟 변수({target_col})값이 비어있는 {dropped_count}개 행을 제거했습니다.")
                            
                            # X, y 분리
                            X = clean_df[selected_features].copy()
                            y = clean_df[target_col].copy()
                            
                            # [핵심 2] 타겟(Y) 데이터 인코딩 (문자일 경우 숫자로 변환)
                            # 회귀인데 Y가 문자면 에러, 분류면 자동 인코딩
                            le_target = None
                            if st.session_state.task == "logit" and y.dtype == 'object':
                                le_target = LabelEncoder()
                                y = pd.Series(le_target.fit_transform(y), index=y.index)
                                st.info("ℹ️ 타겟 변수가 문자열이라 자동으로 숫자로 변환(Encoding)되었습니다.")
                            
                            # X 데이터 전처리 시작
                            num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                            
                            # 1. 값이 없는(All-NaN) 수치형 컬럼 제외
                            valid_num_cols = [c for c in num_cols if X[c].notna().sum() > 0]
                            num_cols = valid_num_cols 

                            # 변환기 준비
                            imputer = SimpleImputer(strategy='mean')
                            scaler = StandardScaler()
                            encoders = {}

                            # 2. 수치형 처리
                            if num_cols:
                                # DataFrame 할당 시 index=X.index 필수
                                X_imputed = imputer.fit_transform(X[num_cols])
                                X_scaled = scaler.fit_transform(X_imputed)
                                X[num_cols] = pd.DataFrame(X_scaled, columns=num_cols, index=X.index)
                            
                            # 3. 범주형 처리
                            for col in cat_cols:
                                X[col] = X[col].fillna("Unknown").astype(str)
                                le = LabelEncoder()
                                trans = le.fit_transform(X[col])
                                X[col] = pd.Series(trans, index=X.index)
                                encoders[col] = le
                            
                            # 4. 최종 컬럼 정리
                            final_features = num_cols + cat_cols
                            X = X[final_features]
                            
                            # 5. 전역 상태 저장
                            st.session_state.preprocess.update({
                                "feature_cols": final_features,
                                "imputer": imputer if num_cols else None,
                                "scaler": scaler if num_cols else None,
                                "encoders": encoders,
                                "num_cols": num_cols,
                                "cat_cols": cat_cols,
                                "target_encoder": le_target # Y 인코더도 저장
                            })
                            
                            # 6. 처리된 데이터 저장
                            st.session_state.data["X_processed"] = X
                            st.session_state.data["y_processed"] = y
                            
                            st.success(f"✅ 전처리 완료! (데이터 수: {len(X)}행)")
                            st.dataframe(X.head(), use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"❌ 오류 발생: {str(e)}")
                            
            with tab2:
                if "X_processed" in st.session_state.data and st.session_state.data["X_processed"] is not None:
                    if st.button("🔍 변수 중요도 확인"):
                        # 저장된 처리 데이터 가져오기
                        X_p = st.session_state.data["X_processed"]
                        y_p = st.session_state.data["y_processed"]
                        
                        # NaN 체크 (디버깅용)
                        if X_p.isna().sum().sum() > 0 or y_p.isna().sum() > 0:
                            st.error("❌ 데이터에 여전히 결측치(NaN)가 남아있습니다. [전처리 실행] 버튼을 다시 눌러주세요.")
                        else:
                            try:
                                # 모델 피팅
                                if st.session_state.task == "logit":
                                    model = DecisionTreeClassifier(max_depth=5, random_state=42)
                                else:
                                    model = DecisionTreeRegressor(max_depth=5, random_state=42)
                                
                                model.fit(X_p, y_p)
                                
                                # 변수 중요도 계산
                                importance = pd.DataFrame({
                                    '변수': X_p.columns,
                                    '중요도': model.feature_importances_
                                }).sort_values(by='중요도', ascending=False)
                                
                                # 시각화
                                fig = px.bar(importance, x='변수', y='중요도', 
                                            title="변수 중요도 (의사결정나무 기준)",
                                            color='중요도', color_continuous_scale='Viridis')
                                st.plotly_chart(fig, use_container_width=True)
                                st.dataframe(importance, use_container_width=True)
                            except Exception as e:
                                st.error(f"변수 중요도 계산 오류: {e}")
                else:
                    st.info("먼저 [전처리 실행]을 완료해주세요.")

# ----------------------
# 步骤4：模型训练（补充完整）
# ----------------------
elif st.session_state.step == 4:
    st.subheader("🔍 모델 학습")
    
    if "X_processed" not in st.session_state.data or st.session_state.data["X_processed"] is None:
        st.warning("⚠️ 먼저 '데이터 전처리' 단계를 완료하세요.")
    else:
        X = st.session_state.data["X_processed"]
        y = st.session_state.data["y_processed"]
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 保存测试集用于后续评估
        st.session_state.data["X_test"] = X_test
        st.session_state.data["y_test"] = y_test
        
        if st.button("🚀 모델 학습 시작", type="primary"):
            with st.spinner("모델 학습 중..."):
                try:
                    # 根据任务类型选择模型
                    if st.session_state.task == "logit":
                        # 分类任务：逻辑回归 + 分类树
                        reg_model = LogisticRegression(max_iter=1000, random_state=42)
                        tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
                    else:
                        # 回归任务：线性回归 + 回归树
                        reg_model = LinearRegression()
                        tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
                    
                    # 训练模型
                    reg_model.fit(X_train, y_train)
                    tree_model.fit(X_train, y_train)
                    
                    # 保存模型
                    st.session_state.models["regression"] = reg_model
                    st.session_state.models["decision_tree"] = tree_model
                    
                    st.success("✅ 模型训练完成！")
                    st.info(f"• 回归模型: {reg_model.__class__.__name__}\n• 决策树模型: {tree_model.__class__.__name__}")
                except Exception as e:
                    st.error(f"❌ 模型训练失败: {e}")

# ----------------------
# 步骤5：模型预测（补充完整）
# ----------------------
elif st.session_state.step == 5:
    st.subheader("🔮 模型预测")
    
    if not st.session_state.models["regression"] or not st.session_state.models["decision_tree"]:
        st.warning("⚠️ 请先完成 '模型训练' 步骤")
    else:
        reg_model = st.session_state.models["regression"]
        tree_model = st.session_state.models["decision_tree"]
        preprocess = st.session_state.preprocess
        
        tab1, tab2 = st.tabs(["📝 单条数据输入", "📂 批量预测"])
        
        with tab1:
            st.markdown("### 输入特征值进行预测")
            input_data = {}
            
            # 数值型特征输入
            if preprocess["num_cols"]:
                st.subheader("数值型特征")
                for col in preprocess["num_cols"]:
                    input_data[col] = st.number_input(f"{col}", value=0.0)
            
            # 类别型特征输入
            if preprocess["cat_cols"]:
                st.subheader("类别型特征")
                for col in preprocess["cat_cols"]:
                    # 获取编码器中的类别
                    le = preprocess["encoders"][col]
                    classes = list(le.classes_)
                    selected = st.selectbox(f"{col}", classes)
                    input_data[col] = le.transform([selected])[0]
            
            if st.button("预测", type="primary"):
                # 构建输入DataFrame
                input_df = pd.DataFrame([input_data])[preprocess["feature_cols"]]
                
                # 应用预处理
                if preprocess["num_cols"]:
                    input_df[preprocess["num_cols"]] = preprocess["imputer"].transform(input_df[preprocess["num_cols"]])
                    input_df[preprocess["num_cols"]] = preprocess["scaler"].transform(input_df[preprocess["num_cols"]])
                
                # 混合预测
                reg_pred = reg_model.predict(input_df)[0]
                tree_pred = tree_model.predict(input_df)[0]
                weight_reg = st.session_state.models["mixed_weights"]["regression"]
                weight_tree = st.session_state.models["mixed_weights"]["decision_tree"]
                mixed_pred = reg_pred * weight_reg + tree_pred * weight_tree
                
                # 显示结果
                st.success("预测完成！")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("回归模型预测", f"{reg_pred:.4f}")
                with col2:
                    st.metric("决策树预测", f"{tree_pred:.4f}")
                with col3:
                    st.metric("混合模型预测", f"{mixed_pred:.4f}")
        
        with tab2:
            st.markdown("### 上传文件进行批量预测")
            uploaded_file = st.file_uploader("选择预测数据文件", type=["csv", "xlsx"])
            
            if uploaded_file:
                try:
                    # 加载数据
                    if uploaded_file.name.endswith('.csv'):
                        pred_df = pd.read_csv(uploaded_file)
                    else:
                        pred_df = pd.read_excel(uploaded_file)
                    
                    # 数据预处理
                    X_pred = pred_df[preprocess["feature_cols"]].copy()
                    
                    # 数值型处理
                    if preprocess["num_cols"]:
                        X_pred[preprocess["num_cols"]] = preprocess["imputer"].transform(X_pred[preprocess["num_cols"]])
                        X_pred[preprocess["num_cols"]] = preprocess["scaler"].transform(X_pred[preprocess["num_cols"]])
                    
                    # 类别型处理
                    for col in preprocess["cat_cols"]:
                        X_pred[col] = X_pred[col].fillna("Unknown").astype(str)
                        # 未见过的类别处理
                        le = preprocess["encoders"][col]
                        X_pred[col] = X_pred[col].apply(lambda x: x if x in le.classes_ else "Unknown")
                        X_pred[col] = le.transform(X_pred[col])
                    
                    # 预测
                    reg_preds = reg_model.predict(X_pred)
                    tree_preds = tree_model.predict(X_pred)
                    weight_reg = st.session_state.models["mixed_weights"]["regression"]
                    weight_tree = st.session_state.models["mixed_weights"]["decision_tree"]
                    mixed_preds = reg_preds * weight_reg + tree_preds * weight_tree
                    
                    # 添加结果
                    pred_df["回归模型预测"] = reg_preds
                    pred_df["决策树预测"] = tree_preds
                    pred_df["混合模型预测"] = mixed_preds
                    
                    st.success(f"批量预测完成！共 {len(pred_df)} 条数据")
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # 下载选项
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        "下载预测结果",
                        csv,
                        "prediction_results.csv",
                        "text/csv",
                        key="download-csv"
                    )
                except Exception as e:
                    st.error(f"预测失败: {e}")

# ----------------------
# 步骤6：性能评估（补充完整）
# ----------------------
elif st.session_state.step == 6:
    st.subheader("📈 模型性能评估")
    
    if "X_test" not in st.session_state.data or not st.session_state.models["regression"]:
        st.warning("⚠️ 请先完成 '模型训练' 步骤")
    else:
        X_test = st.session_state.data["X_test"]
        y_test = st.session_state.data["y_test"]
        reg_model = st.session_state.models["regression"]
        tree_model = st.session_state.models["decision_tree"]
        weight_reg = st.session_state.models["mixed_weights"]["regression"]
        weight_tree = st.session_state.models["mixed_weights"]["decision_tree"]
        
        # 预测结果
        reg_preds = reg_model.predict(X_test)
        tree_preds = tree_model.predict(X_test)
        mixed_preds = reg_preds * weight_reg + tree_preds * weight_tree
        
        # 评估指标计算
        if st.session_state.task == "logit":
            # 分类任务指标
            reg_acc = accuracy_score(y_test, reg_preds.round())
            tree_acc = accuracy_score(y_test, tree_preds.round())
            mixed_acc = accuracy_score(y_test, mixed_preds.round())
            
            st.subheader("分类准确率 (Accuracy)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("回归模型", f"{reg_acc:.4f}")
            with col2:
                st.metric("决策树模型", f"{tree_acc:.4f}")
            with col3:
                st.metric("混合模型", f"{mixed_acc:.4f}")
            
            # 混淆矩阵
            st.subheader("混淆矩阵 (混合模型)")
            cm = confusion_matrix(y_test, mixed_preds.round())
            fig = px.imshow(cm, 
                           labels=dict(x="预测值", y="实际值", color="数量"),
                           x=["0", "1"], y=["0", "1"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            # 回归任务指标
            def regression_metrics(y_true, y_pred):
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                return mae, rmse, r2
            
            reg_mae, reg_rmse, reg_r2 = regression_metrics(y_test, reg_preds)
            tree_mae, tree_rmse, tree_r2 = regression_metrics(y_test, tree_preds)
            mixed_mae, mixed_rmse, mixed_r2 = regression_metrics(y_test, mixed_preds)
            
            st.subheader("回归评估指标")
            metrics_df = pd.DataFrame({
                "模型": ["回归模型", "决策树模型", "混合模型"],
                "MAE": [reg_mae, tree_mae, mixed_mae],
                "RMSE": [reg_rmse, tree_rmse, mixed_rmse],
                "R²": [reg_r2, tree_r2, mixed_r2]
            })
            st.dataframe(metrics_df, use_container_width=True)
            
            # 预测vs实际值可视化
            st.subheader("预测值 vs 实际值")
            sample_df = pd.DataFrame({
                "实际值": y_test.sample(100),
                "混合模型预测值": mixed_preds[y_test.sample(100).index]
            })
            fig = px.scatter(sample_df, x="实际值", y="混合模型预测值", title="预测值 vs 实际值 (抽样)")
            fig.add_trace(go.Scatter(x=[sample_df["实际值"].min(), sample_df["实际值"].max()],
                                    y=[sample_df["实际值"].min(), sample_df["实际值"].max()],
                                    mode="lines", name="理想线", line=dict(dash="dash", color="red")))
            st.plotly_chart(fig, use_container_width=True)
