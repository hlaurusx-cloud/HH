import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
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

# 忽略警告 & 页面设置
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Hybrid Analysis Framework (Auto-Clean)",
    page_icon="🧹",
    layout="wide"
)

# --------------------------------------------------------------------------
# 1. 全局状态初始化 (Session State)
# --------------------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 0  
if "data" not in st.session_state:
    st.session_state.data = {"merged": None, "X_processed": None, "y_processed": None}
if "preprocess" not in st.session_state:
    st.session_state.preprocess = {"imputer": None, "scaler": None, "encoders": None, "feature_cols": None, "target_col": None}
if "models" not in st.session_state:
    st.session_state.models = {"regression": None, "decision_tree": None, "mixed_weights": {"regression": 0.5, "decision_tree": 0.5}}
if "task" not in st.session_state:
    st.session_state.task = "logit" 

# --------------------------------------------------------------------------
# 2. 侧边栏 (Sidebar)
# --------------------------------------------------------------------------
st.sidebar.title("📌 Process Steps")
steps = ["0. Start", "1. Upload Data", "2. Visualization", "3. Preprocessing", "4. Model Training", "5. Prediction", "6. Evaluation"]

for i, step_name in enumerate(steps):
    if st.sidebar.button(step_name, key=f"btn_{i}"):
        st.session_state.step = i

st.sidebar.divider()
st.sidebar.subheader("⚙️ Settings")
st.session_state.task = st.sidebar.radio(
    "Task Type", 
    options=["logit", "decision_tree"], 
    index=0,
    help="logit: Classification (0/1) | decision_tree: Regression (Numeric)"
)

if st.session_state.step >= 4:
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚖️ Model Weights")
    reg_weight = st.sidebar.slider(
        "Regression Weight", 0.0, 1.0, 
        value=st.session_state.models["mixed_weights"]["regression"], step=0.1
    )
    st.session_state.models["mixed_weights"]["regression"] = reg_weight
    st.session_state.models["mixed_weights"]["decision_tree"] = 1.0 - reg_weight
    st.sidebar.info(f"Tree Weight: {1.0 - reg_weight:.1f}")

# --------------------------------------------------------------------------
# 3. 主逻辑 (Main Logic)
# --------------------------------------------------------------------------
st.title("⚡ High-Performance Hybrid Framework")

# [Step 0] 初始画面
if st.session_state.step == 0:
    st.markdown("""
    ### 👋 Welcome!
    
    此版本已包含 **Target Variable 自动清洗功能**。
    
    #### 🧹 自动清洗逻辑:
    * **回归任务**: 自动删除 Target 中的非数字、NaN、Infinity。
    * **分类任务**: 自动删除 Target 中的空值。
    
    👈 请点击左侧 **'1. Upload Data'** 开始。
    """)

# [Step 1] 数据上传
elif st.session_state.step == 1:
    st.subheader("📂 Upload Data")
    
    def load_csv_safe(file_buffer):
        encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
        for enc in encodings:
            try:
                file_buffer.seek(0)
                df = pd.read_csv(file_buffer, encoding=enc)
                return df, enc
            except:
                continue
        return None, None

    uploaded_file = st.file_uploader("Select CSV / Excel / Parquet", type=["csv", "xlsx", "parquet"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df, enc = load_csv_safe(uploaded_file)
                if df is not None:
                    st.success(f"✅ Loaded CSV! (Encoding: {enc})")
                    st.session_state.data["merged"] = df
                else:
                    st.error("❌ Failed to read CSV encoding.")
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
                st.session_state.data["merged"] = df
                st.success("✅ Loaded Excel!")
            else:
                df = pd.read_parquet(uploaded_file)
                st.session_state.data["merged"] = df
                st.success("✅ Loaded Parquet!")
                
        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.data["merged"] is not None:
        st.dataframe(st.session_state.data["merged"].head())

# [Step 2] 可视化
elif st.session_state.step == 2:
    st.subheader("📊 Visualization")
    if st.session_state.data["merged"] is None:
        st.warning("Please upload data first.")
    else:
        df = st.session_state.data["merged"]
        all_cols = df.columns.tolist()
        
        c1, c2, c3 = st.columns(3)
        with c1: x_var = st.selectbox("X Axis", all_cols)
        with c2: y_var = st.selectbox("Y Axis", all_cols, index=1 if len(all_cols)>1 else 0)
        with c3: chart_type = st.selectbox("Chart Type", ["Scatter", "Bar", "Box", "Histogram"])
        
        if chart_type == "Scatter":
            st.plotly_chart(px.scatter(df, x=x_var, y=y_var), use_container_width=True)
        elif chart_type == "Bar":
            st.plotly_chart(px.bar(df, x=x_var, y=y_var), use_container_width=True)
        elif chart_type == "Box":
            st.plotly_chart(px.box(df, x=x_var, y=y_var), use_container_width=True)
        elif chart_type == "Histogram":
            st.plotly_chart(px.histogram(df, x=x_var), use_container_width=True)

# [Step 3] 数据预处理 (核心修改：清洗 Target)
elif st.session_state.step == 3:
    st.subheader("🛠️ Preprocessing & Feature Selection")
    
    if st.session_state.data["merged"] is None:
        st.error("No data uploaded.")
    else:
        df = st.session_state.data["merged"]
        cols = df.columns.tolist()
        
        c1, c2 = st.columns([1, 2])
        with c1:
            target_col = st.selectbox("🎯 Target Variable (Y)", cols)
        with c2:
            candidates = [c for c in cols if c != target_col]
            selected_features = st.multiselect("📋 Input Features (X)", candidates, default=candidates[:5])
        
        st.divider()
        
        if st.button("🚀 Run Preprocessing", type="primary"):
            if not selected_features:
                st.error("Please select at least 1 feature.")
            else:
                with st.spinner("Cleaning Target Variable & Processing..."):
                    try:
                        clean_df = df.copy()
                        original_len = len(clean_df)
                        
                        # --- 核心修改：清洗 Target ---
                        # 1. 删除 Target 为空的行
                        clean_df = clean_df.dropna(subset=[target_col])
                        
                        # 2. 如果是回归任务(decision_tree)，必须保证 Target 是数字
                        if st.session_state.task != "logit":
                            # 强制转为数字，错误变 NaN
                            clean_df[target_col] = pd.to_numeric(clean_df[target_col], errors='coerce')
                            # 再次删除 NaN
                            clean_df = clean_df.dropna(subset=[target_col])
                            # 删除无穷大 inf
                            clean_df = clean_df[~clean_df[target_col].isin([np.inf, -np.inf])]
                        
                        cleaned_len = len(clean_df)
                        dropped_count = original_len - cleaned_len
                        
                        if dropped_count > 0:
                            st.warning(f"⚠️ 已自动删除 Target 异常的 {dropped_count} 行数据 (NaN/Inf/Text).")
                        
                        # --- 正常的特征处理 ---
                        X = clean_df[selected_features].copy()
                        y = clean_df[target_col].copy()
                        
                        le_target = None
                        if st.session_state.task == "logit" and y.dtype == 'object':
                            le_target = LabelEncoder()
                            y = le_target.fit_transform(y)
                            st.info("ℹ️ Target converted to numbers for Classification.")
                        
                        num_cols = X.select_dtypes(include=['number']).columns.tolist()
                        cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
                        
                        imputer = SimpleImputer(strategy='mean')
                        scaler = StandardScaler()
                        encoders = {}
                        
                        if num_cols:
                            X[num_cols] = imputer.fit_transform(X[num_cols])
                            X[num_cols] = scaler.fit_transform(X[num_cols])
                            
                        for col in cat_cols:
                            X[col] = X[col].fillna("Unknown").astype(str)
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col])
                            encoders[col] = le
                            
                        st.session_state.data["X_processed"] = X
                        st.session_state.data["y_processed"] = y
                        st.session_state.preprocess = {
                            "feature_cols": selected_features,
                            "target_col": target_col,
                            "imputer": imputer, "scaler": scaler, "encoders": encoders,
                            "target_encoder": le_target
                        }
                        
                        st.success(f"✅ Preprocessing Done!")
                        st.markdown(f"**Final Data Shape**: {X.shape[0]} rows, **{X.shape[1]} cols**")
                        st.dataframe(X.head())
                        
                    except Exception as e:
                        st.error(f"Error during preprocessing: {e}")

# [Step 4] 模型训练
elif st.session_state.step == 4:
    st.subheader("🤖 Model Training")
    
    if st.session_state.data["X_processed"] is None:
        st.warning("Please finish Step 3 first.")
    else:
        X = st.session_state.data["X_processed"]
        y = st.session_state.data["y_processed"]
        
        st.info(f"Using **{X.shape[1]} features** for training.")
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
        
        if st.button("🔥 Start Training", type="primary"):
            with st.spinner("Training Models..."):
                try:
                    stratify_opt = y if (st.session_state.task == "logit" and len(np.unique(y)) > 1) else None
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=stratify_opt
                    )
                    
                    if st.session_state.task == "logit":
                        reg_model = LogisticRegression(n_jobs=-1, max_iter=1000, random_state=42)
                        dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
                    else:
                        reg_model = LinearRegression(n_jobs=-1)
                        dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)
                    
                    reg_model.fit(X_train, y_train)
                    dt_model.fit(X_train, y_train)
                    
                    st.session_state.models["regression"] = reg_model
                    st.session_state.models["decision_tree"] = dt_model
                    st.session_state.data.update({
                        "X_train": X_train, "X_test": X_test, 
                        "y_train": y_train, "y_test": y_test
                    })
                    
                    st.success("✅ Training Complete!")
                    c1, c2 = st.columns(2)
                    c1.metric("Train Set", f"{len(X_train):,}")
                    c2.metric("Test Set", f"{len(X_test):,}")
                    
                except Exception as e:
                    st.error(f"Training Error: {e}")

# [Step 5] 预测
elif st.session_state.step == 5:
    st.subheader("🔮 Prediction")
    
    if st.session_state.models["regression"] is None:
        st.error("Model not trained.")
    else:
        def make_prediction(input_df):
            pp = st.session_state.preprocess
            X_input = input_df.copy()
            
            for col in pp["feature_cols"]:
                if col not in X_input.columns:
                    X_input[col] = 0
            X_input = X_input[pp["feature_cols"]]
            
            num_cols = X_input.select_dtypes(include=['number']).columns
            cat_cols = X_input.select_dtypes(exclude=['number']).columns
            
            if len(num_cols) > 0 and pp["imputer"]:
                X_input[num_cols] = pp["imputer"].transform(X_input[num_cols])
                X_input[num_cols] = pp["scaler"].transform(X_input[num_cols])
                
            for col in cat_cols:
                X_input[col] = X_input[col].fillna("Unknown").astype(str)
                if col in pp["encoders"]:
                    le = pp["encoders"][col]
                    X_input[col] = X_input[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    X_input[col] = le.transform(X_input[col])

            reg = st.session_state.models["regression"]
            dt = st.session_state.models["decision_tree"]
            w_reg = st.session_state.models["mixed_weights"]["regression"]
            w_dt = st.session_state.models["mixed_weights"]["decision_tree"]
            
            if st.session_state.task == "logit":
                p1 = reg.predict_proba(X_input)[:, 1]
                p2 = dt.predict_proba(X_input)[:, 1]
                final_prob = (p1 * w_reg) + (p2 * w_dt)
                final_pred = (final_prob >= 0.5).astype(int)
                return final_pred, final_prob
            else:
                p1 = reg.predict(X_input)
                p2 = dt.predict(X_input)
                final_pred = (p1 * w_reg) + (p2 * w_dt)
                return final_pred, None

        mode = st.radio("Input Mode", ["Single Input", "File Upload"])
        
        if mode == "Single Input":
            input_data = {}
            feats = st.session_state.preprocess["feature_cols"]
            cols_ui = st.columns(3)
            for i, f in enumerate(feats):
                with cols_ui[i % 3]:
                    input_data[f] = st.text_input(f, "0")
            
            if st.button("Predict"):
                df_single = pd.DataFrame([input_data])
                for c in df_single.columns:
                    try: df_single[c] = pd.to_numeric(df_single[c])
                    except: pass
                
                pred, prob = make_prediction(df_single)
                st.info(f"Result: {pred[0]}")
                if prob is not None:
                    st.write(f"Probability: {prob[0]:.4f}")
                    
        else:
            up = st.file_uploader("Upload CSV for Prediction", type=["csv"])
            if up:
                df_batch = pd.read_csv(up)
                if st.button("Predict Batch"):
                    preds, probs = make_prediction(df_batch)
                    df_batch["Prediction"] = preds
                    if probs is not None:
                        df_batch["Probability"] = probs
                    st.dataframe(df_batch)

# [Step 6] 评估
elif st.session_state.step == 6:
    st.subheader("📈 Evaluation")
    
    if st.session_state.data["X_test"] is None:
        st.error("Model not trained.")
    else:
        X_test = st.session_state.data["X_test"]
        y_test = st.session_state.data["y_test"]
        reg = st.session_state.models["regression"]
        dt = st.session_state.models["decision_tree"]
        w_reg = st.session_state.models["mixed_weights"]["regression"]
        w_dt = st.session_state.models["mixed_weights"]["decision_tree"]
        
        if st.session_state.task == "logit":
            p1 = reg.predict_proba(X_test)[:, 1]
            p2 = dt.predict_proba(X_test)[:, 1]
            p_final = (p1 * w_reg) + (p2 * w_dt)
            y_pred = (p_final >= 0.5).astype(int)
            
            acc = accuracy_score(y_test, y_pred)
            try:
                auc_score = auc(*roc_curve(y_test, p_final)[:2])
            except:
                auc_score = 0.0
            
            st.metric("Accuracy", f"{acc:.4f}")
            st.metric("AUC Score", f"{auc_score:.4f}")
            
            fpr, tpr, _ = roc_curve(y_test, p_final)
            fig = px.area(x=fpr, y=tpr, title="ROC Curve")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            p1 = reg.predict(X_test)
            p2 = dt.predict(X_test)
            p_final = (p1 * w_reg) + (p2 * w_dt)
            
            mae = mean_absolute_error(y_test, p_final)
            rmse = np.sqrt(mean_squared_error(y_test, p_final))
            r2 = r2_score(y_test, p_final)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{mae:.4f}")
            c2.metric("RMSE", f"{rmse:.4f}")
            c3.metric("R2 Score", f"{r2:.4f}")
            
            fig = px.scatter(x=y_test, y=p_final, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted")
            st.plotly_chart(fig, use_container_width=True)
