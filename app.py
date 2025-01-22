# app.py
import streamlit as st
import joblib
import numpy as np

# Load artifacts
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')
mean_values = joblib.load('mean_values.pkl')

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")

st.title("Diabetes Risk Prediction")
st.write("Enter your health metrics to assess diabetes risk")

# Sample data buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🧪 Load High-Risk Sample"):
        st.session_state.update({
            'pregnancies': 7,
            'glucose': 180,
            'bp': 64,
            'skin_thickness': 25,
            'insulin': 0,  # Will be imputed
            'bmi': 33.9,
            'dpf': 0.745,
            'age': 48
        })
        st.success("High-risk sample loaded! Click 'Assess Risk'")

with col2:
    if st.button("🌱 Load Low-Risk Sample"):
        st.session_state.update({
            'pregnancies': 2,
            'glucose': 90,
            'bp': 70,
            'skin_thickness': 27,
            'insulin': 80,
            'bmi': 22.1,
            'dpf': 0.324,
            'age': 28
        })
        st.success("Low-risk sample loaded! Click 'Assess Risk'")

with st.form("patient_details"):
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input(
            'Pregnancies', 
            min_value=0, 
            max_value=20, 
            value=st.session_state.get('pregnancies', 0)
        )
        glucose = st.number_input(
            'Glucose (mg/dL)', 
            min_value=0, 
            max_value=300, 
            value=st.session_state.get('glucose', 0)
        )
        bp = st.number_input(
            'Blood Pressure (mm Hg)', 
            min_value=0, 
            max_value=150, 
            value=st.session_state.get('bp', 0)
        )
        skin_thickness = st.number_input(
            'Skin Thickness (mm)', 
            min_value=0, 
            max_value=100, 
            value=st.session_state.get('skin_thickness', 0)
        )
    
    with col2:
        insulin = st.number_input(
            'Insulin (μU/mL)', 
            min_value=0, 
            max_value=900, 
            value=st.session_state.get('insulin', 0)
        )
        bmi = st.number_input(
            'BMI', 
            min_value=0.0, 
            max_value=70.0, 
            value=st.session_state.get('bmi', 0.0),
            step=0.1
        )
        dpf = st.number_input(
            'Diabetes Pedigree Function', 
            min_value=0.0, 
            max_value=2.5, 
            value=st.session_state.get('dpf', 0.0),
            step=0.001,
            format="%.3f"
        )
        age = st.number_input(
            'Age', 
            min_value=10, 
            max_value=120, 
            value=st.session_state.get('age', 25)
        )
    
    submitted = st.form_submit_button("Assess Risk")

if submitted:
    # Create input array
    raw_input = [
        pregnancies,
        glucose,
        bp,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]
    
    # Handle zero values using training data means
    features_to_impute = {
        'Glucose': 1,
        'BloodPressure': 2,
        'SkinThickness': 3,
        'Insulin': 4,
        'BMI': 5
    }
    
    for feature, idx in features_to_impute.items():
        if raw_input[idx] == 0:
            raw_input[idx] = mean_values[feature]
    
    # Scale features
    scaled_input = scaler.transform([raw_input])
    
    # Make prediction
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1]
    
    # Display results
    st.subheader("Assessment Results")
    if prediction[0] == 1:
        st.error(f"High risk of diabetes ({probability:.1%} probability)")
        st.write("🔍 Key risk factors in this profile:")
        st.write("- Elevated glucose levels" if glucose > 140 else "")
        st.write("- High BMI" if bmi > 30 else "")
        st.write("- Advanced age" if age > 45 else "")
        st.write("Consult your healthcare provider for further evaluation")
    else:
        st.success(f"Low risk of diabetes ({probability:.1%} probability)")
        st.write("✅ Positive indicators in this profile:")
        st.write("- Normal glucose levels" if glucose <= 140 else "")
        st.write("- Healthy BMI" if bmi <= 25 else "")
        st.write("- Younger age" if age <= 45 else "")
        st.write("Maintain healthy lifestyle habits")
    
    st.markdown("---")
    st.caption("ℹ️ Note: These example profiles are synthetic and for demonstration purposes only. Always consult a medical professional for health assessments.")
