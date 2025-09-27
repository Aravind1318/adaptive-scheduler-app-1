import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.title("🤖 AI-Driven Adaptive Scheduling with Smart Runtime Scaling")

# -------------------------------
# Feature Engineering Function
# -------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Production_Load" in df and "Deadline_Hours" in df:
        df["Urgency"] = df["Production_Load"] / (df["Deadline_Hours"] + 1e-3)

    if "Available_Operators" in df and "Available_Machines" in df:
        df["Operator_Machine_Ratio"] = df["Available_Operators"] / (df["Available_Machines"] + 1)

    # Explicit runtime scaling formula
    if "Production_Load" in df and "Available_Operators" in df and "Available_Machines" in df:
        df["Scaled_Runtime"] = df["Production_Load"] / ((df["Available_Operators"] * df["Available_Machines"]) + 1e-3)

    if "Machine_Efficiency" in df and "Scaled_Runtime" in df:
        df["Adjusted_Runtime"] = df["Scaled_Runtime"] / (df["Machine_Efficiency"] + 1e-3)

    if "Shift" in df:
        df["Shift_binary"] = df["Shift"].apply(lambda x: 1 if str(x).lower() == "night" else 0)

    return df


# -------------------------------
# Upload & Preprocess Dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = add_engineered_features(df)

    st.write("✅ Dataset with engineered features:")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()

    # Select input and target
    st.subheader("⚙️ Select Features and Target")
    input_cols = st.multiselect(
        "Select Input Columns (X)", 
        all_columns, 
        default=[c for c in all_columns if c not in ["Expected_Runtime_Min"]]
    )
    output_cols = st.multiselect(
        "Select Target Column (y)", 
        all_columns, 
        default=["Expected_Runtime_Min"]
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


# -------------------------------
# Prediction Mode
# -------------------------------
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

        ml_prediction = st.session_state["model"].predict(input_encoded)[0]

        # Hybrid prediction = formula + ML correction
        if "Adjusted_Runtime" in input_df:
            base_runtime = input_df["Adjusted_Runtime"].iloc[0]
        elif "Scaled_Runtime" in input_df:
            base_runtime = input_df["Scaled_Runtime"].iloc[0]
        else:
            base_runtime = ml_prediction

        final_runtime = (0.7 * base_runtime) + (0.3 * ml_prediction)

        st.success("🎯 Predictions:")
        for col in st.session_state["output_cols"]:
            if col == "Expected_Runtime_Min":
                st.write(f"**{col}:** {final_runtime:.2f}")
            else:
                st.write(f"**{col}:** {ml_prediction:.2f}")
else:
    st.info("Please upload a CSV, select columns, and train the model first.")
