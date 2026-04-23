import streamlit as st
import kagglehub
import pandas as pd
import numpy as np
import pickle
import os

# -----------------------------
# LOAD OR CREATE MODEL
# -----------------------------
def load_or_create_model():
    if os.path.exists("model.pkl") and os.path.getsize("model.pkl") > 0:
        try:
            with open("model.pkl", "rb") as f:
                return pickle.load(f)
        except:
            pass  # fallback to training

    # Download dataset
    path = kagglehub.dataset_download("wisam1985/advanced-iot-agriculture-2024")
    file_path = path + "/Advanced_IoT_Dataset.csv"

    # Read dataset safely
    try:
        df = pd.read_csv(file_path)
    except:
        df = pd.read_csv(file_path, encoding="latin1")

    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 🔥 FIX: handle categorical columns
    X = pd.get_dummies(X)

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier()
    model.fit(X_scaled, y)

    columns = X.columns.tolist()

    # Save model
    with open("model.pkl", "wb") as f:
        pickle.dump((model, scaler, le, columns), f)

    return model, scaler, le, columns


model, scaler, le, columns = load_or_create_model()

# -----------------------------
# TITLE
# -----------------------------
st.title("🌱 Smart Agriculture Prediction App")
st.write("Enter input values to predict output")

# -----------------------------
# USER INPUT
# -----------------------------
user_input = []

for col in columns:
    val = st.number_input(f"{col}", value=0.0)
    user_input.append(val)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):
    try:
        input_data = pd.DataFrame([user_input], columns=columns)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        result = le.inverse_transform(prediction)

        st.success(f"Prediction: {result[0]}")

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------
# OPTIONAL: SHOW DATASET
# -----------------------------
st.subheader("Dataset Preview")

if st.button("Show Dataset"):
    path = kagglehub.dataset_download("wisam1985/advanced-iot-agriculture-2024")
    file_path = path + "/Advanced_IoT_Dataset.csv"

    try:
        df = pd.read_csv(file_path)
    except:
        df = pd.read_csv(file_path, encoding="latin1")

    st.dataframe(df.head())