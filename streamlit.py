import streamlit as st
import pandas as pd
import pickle
from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.models.predict_model import evaluate_model
from src.models.visualization import plot_residuals
from sklearn.preprocessing import MinMaxScaler
with open("README.md", "r") as f:
    readme_content = f.read()

st.title("🎓 UCLA Admission Chance Predictor")

model, scaler = None, None

if st.sidebar.button("Train Model"):
    with st.spinner("Training..."):
        df = load_and_preprocess_data("final.csv")
        X = build_features(df)
        y = df['Admit_Chance']
        model, scaler, X_test_scaled, y_test = train_model(X, y)
        rmse, r2 = evaluate_model(model, X_test_scaled, y_test)
        st.success(f"Model trained. RMSE: {rmse:.4f}, R²: {r2:.4f}")
        st.write("### Residual Plot")
        plot_residuals(y_test, model.predict(X_test_scaled))

try:
    with open("models/nn_model.pkl", "rb") as f:
        model = pickle.load(f)
    df = load_and_preprocess_data("final.csv")
    X = build_features(df)
    scaler = MinMaxScaler().fit(X)
except:
    st.warning("Please train the model first.")

st.header("📋 Enter Applicant Information")
with st.form("admission_form"):
    GRE = st.slider("GRE Score", 260, 340, 320)
    TOEFL = st.slider("TOEFL Score", 0, 120, 110)
    UR = st.slider("University Rating", 1, 5, 3)
    SOP = st.slider("Statement of Purpose", 1.0, 5.0, 3.5)
    LOR = st.slider("Letter of Recommendation", 1.0, 5.0, 3.5)
    CGPA = st.slider("CGPA", 0.0, 10.0, 8.5)
    Research = st.selectbox("Research Experience", [0, 1])
    submit = st.form_submit_button("Predict Admission Chance")

if submit and model and scaler:
    input_df = pd.DataFrame([{
        'GRE_Score': GRE,
        'TOEFL_Score': TOEFL,
        'University_Rating': UR,
        'SOP': SOP,
        'LOR': LOR,
        'CGPA': CGPA,
        'Research': Research
    }])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    st.subheader("🎯 Predicted Admission Chance")
    st.success(f"{pred * 100:.2f}%")