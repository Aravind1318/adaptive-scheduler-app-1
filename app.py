import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.title("🤖 AI-Driven Adaptive Scheduling")


# ----------------------------
# Feature Engineering Function
# ----------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Urgency = load vs deadline
    if "Production_Load" in df and "Deadline_Hours" in df:
        df["Urgency"] = df["Production_Load"] / (df["Deadline_Hours"] + 1e-3)

    # Operator-to-Machine ratio
    if "Available_Operators" in df and "Available_Machines" in df:
        df["Operator_Machine_Ratio"] = df["Available_Operators"] / (df["Available_Machines"] + 1)

    # Capacity-based features
    if "Available_Operators" in df and "Machine_Efficiency" in df:
        df["Capacity"] = df["Available_Operators"] * df["Machine_Efficiency"]

    if "Production_Load" in df and "Available_Operators" in df:
        df["Load_per_operator"] = df["Production_Load"] / (df["Available_Operators"] + 1)

    if "Production_Load" in df and "Capacity" in df:
        df["Load_per_capacity"] = df["Production_Load"] / (df["Capacity"] + 1e-3)

    # Shift encoding
    if "Shift" in df:
        df["Shift_binary"] = df["Shift"].apply(lambda x: 1 if str(x).lower() == "night" else 0)

    return df


# ----------------------------
# Upload CSV
# ----------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    df = add_engineered_features(df)

    st.write("✅ Dataset loaded successfully with engineered features!")
    st.dataframe(df.head())

    # ----------------------------
    # Select Features and Targets
    # ----------------------------
    all_columns = df.columns.tolist()
    st.subheader("Select Features and Target Columns")

    input_cols = st.multiselect(
        "Select Input Columns (X)",
        all_columns,
        default=[c for c in all_columns if c not in ["Machine", "Manpower", "Expected_Runtime_Min"]]
    )
    output_cols = st.multiselect(
        "Select Output Columns (y)",
        all_columns,
        default=["Machine", "Manpower"]
    )

    # ----------------------------
    # Train Model
    # ----------------------------
    if input_cols and output_cols and st.button("🚀 Train Model"):
        X = df[input_cols]
        y = df[output_cols]

        # Encode categoricals
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )

        # Train Random Forest
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        if y.shape[1] == 1:
            r2 = r2_score(y_test, y_pred)
        else:
            r2 = r2_score(y_test, y_pred, multioutput="uniform_average")

        st.subheader("📊 Model Accuracy")
        st.write(f"✅ R² Score: {r2*100:.2f}%")

        # Save in session state
        st.session_state["model"] = model
        st.session_state["features"] = X_encoded.columns
        st.session_state["output_cols"] = output_cols
        st.session_state["input_cols"] = input_cols
        st.session_state["df"] = df


# ----------------------------
# Prediction Section
# ----------------------------
if "model" in st.session_state:
    st.subheader("🔧 Predict for New Input")

    df = st.session_state["df"]
    input_cols = st.session_state["input_cols"]

    input_data = {}
    for col in input_cols:
        if df[col].dtype in ["int64", "float64"]:
            val = st.number_input(f"{col}", value=None, placeholder=f"Enter {col}")
            input_data[col] = val
        else:
            options = df[col].dropna().unique().tolist()
            val = st.selectbox(f"{col}", options)
            input_data[col] = val

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])

        # Apply feature engineering
        input_df = add_engineered_features(input_df)

        # Encode like training
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=st.session_state["features"], fill_value=0)

        prediction = st.session_state["model"].predict(input_encoded)
        prediction = np.atleast_2d(prediction)

        st.success("🎯 Predictions:")
        for i, col in enumerate(st.session_state["output_cols"]):
            st.write(f"**{col}:** {round(prediction[0][i], 2)}")
else:
    st.info("Please upload a CSV, select columns, and click 🚀 Train Model")
