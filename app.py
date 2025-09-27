import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.title("🤖 AI-Driven Adaptive Scheduling")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Production_Load" in df and "Deadline_Hours" in df:
        df["Urgency"] = df["Production_Load"] / (df["Deadline_Hours"] + 1e-3)
    if "Available_Operators" in df and "Available_Machines" in df:
        df["Operator_Machine_Ratio"] = df["Available_Operators"] / (df["Available_Machines"] + 1)
    if "Expected_Runtime_Min" in df and "Machine_Efficiency" in df:
        df["Adjusted_Runtime"] = df["Expected_Runtime_Min"] / (df["Machine_Efficiency"] + 1e-3)
    if "Production_Load" in df and "Available_Operators" in df:
        df["Load_per_operator"] = df["Production_Load"] / (df["Available_Operators"] + 1)
    if "Shift" in df:
        df["Shift_binary"] = df["Shift"].apply(lambda x: 1 if str(x).lower() == "night" else 0)
    return df

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    df = add_engineered_features(df)

    st.write("✅ Dataset loaded successfully with engineered features!")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()
    st.subheader("Select Features and Target Columns")

    input_cols = st.multiselect(
        "Select Input Columns (X)", 
        all_columns, 
        default=[c for c in all_columns if c not in ["Machine", "Manpower"]]
    )
    output_cols = st.multiselect(
        "Select Output Columns (y)", 
        all_columns, 
        default=["Machine", "Manpower"]
    )

    if input_cols and output_cols and st.button("🚀 Train Model"):
        X = df[input_cols]
        y = df[output_cols]
        X_encoded = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        st.subheader("📊 Model Accuracy")
        st.write(f"✅ R² Score: {r2*100:.2f}%")

        st.session_state["model"] = model
        st.session_state["features"] = X_encoded.columns
        st.session_state["output_cols"] = output_cols
        st.session_state["input_cols"] = input_cols
        st.session_state["df"] = df

if "model" in st.session_state:
    st.subheader("🔧 Predict for New Input")

    df = st.session_state["df"]
    input_cols = st.session_state["input_cols"]

    input_data = {}
    for col in input_cols:
        if df[col].dtype in ["int64", "float64"]:
            val = st.number_input(
                f"{col}", 
                float(df[col].min()), 
                float(df[col].max()), 
                float(df[col].mean())
            )
            input_data[col] = val
        else:
            options = df[col].unique().tolist()
            val = st.selectbox(f"{col}", options)
            input_data[col] = val

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])

        input_df = add_engineered_features(input_df)

        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=st.session_state["features"], fill_value=0)

        prediction = st.session_state["model"].predict(input_encoded)
        prediction = np.round(prediction).astype(int)
        prediction = np.atleast_2d(prediction)

        st.success("🎯 Predictions:")
        for i, col in enumerate(st.session_state["output_cols"]):
            st.write(f"**{col}:** {prediction[0][i]}")
else:
    st.info("Please upload a CSV, select columns, and click 🚀 Train Model")
