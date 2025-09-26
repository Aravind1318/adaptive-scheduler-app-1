import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.title("AI-Driven Adaptive Scheduling (Generic CSV Upload)")

# ----------------------------
# Upload CSV
# ----------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Dataset loaded successfully!")
    st.dataframe(df.head())

    # ----------------------------
    # Select input & output columns
    # ----------------------------
    all_columns = df.columns.tolist()

    st.subheader("Select Features and Target Columns")
    input_cols = st.multiselect("Select Input Columns (X)", all_columns, default=all_columns[:-2])
    output_cols = st.multiselect("Select Output Columns (y)", all_columns, default=all_columns[-2:])

    if input_cols and output_cols:
        X = df[input_cols]
        y = df[output_cols]

        # ----------------------------
        # Train-test split
        # ----------------------------
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ----------------------------
        # Train Random Forest
        # ----------------------------
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        # Accuracy
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        st.subheader("Model Accuracy (R² Score)")
        st.write(f"✅ Accuracy: {r2*100:.2f}%")

        # ----------------------------
        # Input form for predictions
        # ----------------------------
        st.subheader("🔧 Predict Outputs from New Input")

        input_data = {}
        for col in input_cols:
            if df[col].dtype in ["int64", "float64"]:
                val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                input_data[col] = val
            else:
                options = df[col].unique().tolist()
                val = st.selectbox(f"{col}", options)
                input_data[col] = val

        # Prediction
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)
            prediction = np.round(prediction[0]).astype(int)

            for i, col in enumerate(output_cols):
                st.success(f"🔹 Predicted {col}: {prediction[i]}")

else:
    st.info("Please upload a CSV file to proceed.")
