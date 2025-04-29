import streamlit as st
import joblib
import numpy as np
import base64

# Load model and scaler
model = joblib.load("voting_classifier_gbx_model.pkl")
scaler = joblib.load("scaler.pkl")

# Background image setup
def set_background(image_file):
    with open(image_file, "rb") as img_file:
        b64_img = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Custom CSS
def inject_css():
    st.markdown("""
        <style>
        .title {
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            color: #002244;
            margin-bottom: 30px;
        }
        .prediction-box {
            background-color: #2563eb;
            color: white;
            font-size: 20px;
            padding: 15px;
            border-radius: 10px;
            margin-top: 30px;
            text-align: center;
        }
        .prediction-box-diabetic {
            background-color: #dc2626;
        }
        .stButton>button {
            background-color: #1976d2;
            color: white;
            border-radius: 6px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            margin-top: 15px;
        }
        .stButton>button:hover {
            background-color: #125bb5;
        }
        .stForm > div:first-child {
            background: transparent;
            padding: 0;
        }
        </style>
    """, unsafe_allow_html=True)

# Main App
def main():
    # --- Authentication Section ---
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔒 Login Page")

        # Hardcoded credentials (you can improve later)
        USERNAME = "admin"
        PASSWORD = "admin@25"

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == USERNAME and password == PASSWORD:
                st.session_state.authenticated = True
                st.success("Login successful! 🎉")
                st.rerun()  # Refresh the page after login
            else:
                st.error("Invalid username or password. Please try again.")
        return  # Stop the app here if not logged in

    # --- Main Diabetes Prediction App ---

    set_background(r"D:/hellodiabetes/assests/background.jpg")
    inject_css()

    with st.container():
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<div class="title">🩺 Diabetes Prediction System</div>', unsafe_allow_html=True)

        with st.form("diabetes_form"):
            col1, col2 = st.columns(2, gap="large")

            with col1:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                age = st.number_input("Age", min_value=0, max_value=120, step=1)
                hypertension = st.selectbox("Hypertension", [0, 1])
                heart_disease = st.selectbox("Heart Disease", [0, 1])
                smoking_history = st.selectbox("Smoking History", ['never', 'No Info', 'former', 'current', 'not current'])
                pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
                dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.4f")

            with col2:
                bmi = st.number_input("BMI (kg/m²)", min_value=0.0, format="%.2f")
                hba1c = st.number_input("HbA1c Level (%)", min_value=0.0, format="%.2f")
                glucose = st.number_input("Glucose Level (mmol/L)", min_value=0.0, format="%.2f")
                bp = st.number_input("Blood Pressure (mmHg)", min_value=0.0, format="%.2f")
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, format="%.2f")
                insulin = st.number_input("Insulin (μU/mL)", min_value=0.0, format="%.2f")

            submit = st.form_submit_button("Predict")

            if submit:
                gender_encoded = {"Male": 0, "Female": 1, "Other": 2}[gender]
                smoking_encoded = {
                    'never': 0, 'No Info': 1, 'former': 2, 'current': 3, 'not current': 4
                }[smoking_history]

                features = np.array([
                    gender_encoded, age, hypertension, heart_disease,
                    smoking_encoded, bmi, hba1c, glucose, pregnancies,
                    bp, skin_thickness, insulin, dpf
                ]).reshape(1, -1)

                scaled_input = scaler.transform(features)
                prediction = model.predict(scaled_input)[0]

                if prediction == 1:
                    st.markdown('<div class="prediction-box prediction-box-diabetic">⚠ The model predicts the patient <strong>has diabetes</strong>.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-box">✅ The model predicts the patient <strong>does not have diabetes</strong>.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # Close glass container

if __name__ == "__main__":
    main()
