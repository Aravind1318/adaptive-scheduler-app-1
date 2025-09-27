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
    if "production_load" in df and "deadline_hours" in df:
        df["urgency"] = df["production_load"] / (df["deadline_hours"] + 1e-3)
    if "available_operators" in df and "available_machines" in df:
        df["operator_machine_ratio"] = df["available_operators"] / (df["available_machines"] + 1)
    if "expected_runtime_min" in df and "machine_efficiency" in df:
        df["adjusted_runtime"] = df["expected_runtime_min"] / (df["machine_efficiency"] + 1e-3)
    if "production_load" in df and "available_operators" in df:
        df["load_per_operator"] = df["production_load"] / (df["available_operators"] + 1)
    if "shift" in df:
        df["shift_binary"] = df["shift"].apply(lambda x: 1 if str(x).lower() == "night" else 0)
    return df

# ----------------------------
# Upload CSV
# ----------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Add engineered features automatically
    df = add_engineered_features(df)

    st.write("✅ Dataset loaded successfully with engineered features!")
    st.dataframe(df.head())

    # ----------------------------
    # Lock outputs: machine + manpower
    # ----------------------------
    all_columns = df.columns.tolist()
    output_cols = [col for col in ["machine", "manpower"] if col in all_columns]

    if not output_cols:
        st.error("❌ Your dataset must include 'machine' and/or 'manpower' as output columns.")
    else:
        st.subheader("Select Feature Columns (Inputs)")
        input_cols = st.multiselect(
            "Select Input Columns (X)", 
            all_columns, 
            default=[c for c in all_columns if c not in output_cols]
        )

        # ----------------------------
        # Train button
        # ----------------------------
        if input_cols and st.button("🚀 Train Model"):
            X = df[input_cols]
            y = df[output_cols]

            # Encode categorical variables
            X_encoded = pd.get_dummies(X, drop_first=True)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=0.2, random_state=42
            )

            # Train Random Forest
            model = RandomForestRegressor(
                n_estimators=500,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            # Evaluate accuracy
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            st.subheader("📊 Model Accuracy")
            st.write(f"✅ R² Score: {r2*100:.2f}%")

            # Save model in session state
            st.session_state["model"] = model
            st.session_state["features"] = X_encoded.columns
            st.session_state["output_cols"] = output_cols
            st.session_state["input_cols"] = input_cols
            st.session_state["df"] = df

# ----------------------------
# Prediction section
# ----------------------------
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

        # Apply same feature engineering to new input
        input_df = add_engineered_features(input_df)

        # Apply same encoding as training
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=st.session_state["features"], fill_value=0)

        prediction = st.session_state["model"].predict(input_encoded)
        prediction = np.round(prediction[0]).astype(int)

        st.success("🎯 Predictions:")
        for i, col in enumerate(st.session_state["output_cols"]):
            st.write(f"**{col}: {prediction[i]}**")
else:
    st.info("Please upload a CSV, then click 🚀 Train Model")
