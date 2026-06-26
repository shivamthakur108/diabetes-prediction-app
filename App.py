import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open('best_diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Page config
st.set_page_config(
    page_title="AI Diabetes Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Title
st.markdown("""
    <h1 style='text-align: center; color: #1a237e;'>
    🏥 AI-Based Multi-Model Prediction System
    </h1>
    <h4 style='text-align: center; color: #283593;'>
    Diabetes Prediction Dashboard
    </h4>
    <hr>
""", unsafe_allow_html=True)

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Models", "6")
col2.metric("Best Model", "ANN")
col3.metric("Best Accuracy", "97.19%")
col4.metric("Dataset Size", "1,00,000")

st.markdown("<hr>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🔮 Predict Diabetes", "📊 Model Comparison"])

with tab1:
    st.subheader("Enter Patient Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age", 1, 100, 30)
        hypertension = st.selectbox("Hypertension", [0, 1], 
                                     format_func=lambda x: "Yes" if x==1 else "No")

    with col2:
        heart_disease = st.selectbox("Heart Disease", [0, 1],
                                      format_func=lambda x: "Yes" if x==1 else "No")
        smoking_history = st.selectbox("Smoking History", 
                                        ["never", "current", "former", 
                                         "ever", "not current", "No Info"])
        bmi = st.slider("BMI", 10.0, 60.0, 25.0, step=0.1)

    with col3:
        hba1c = st.slider("HbA1c Level", 3.0, 15.0, 5.5, step=0.1)
        glucose = st.slider("Blood Glucose Level", 50, 300, 100)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 Predict Diabetes Risk", use_container_width=True):

        # Encoding
        gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
        smoking_map = {'No Info': 0, 'current': 1, 'ever': 2,
                       'former': 3, 'never': 4, 'not current': 5}

        input_data = np.array([[
            gender_map[gender], age, hypertension,
            heart_disease, smoking_map[smoking_history],
            bmi, hba1c, glucose
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        confidence = round(max(probability) * 100, 2)

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == 1:
            st.error(f"⚠️ Result: DIABETIC — Confidence: {confidence}%")
        else:
            st.success(f"✅ Result: NON-DIABETIC — Confidence: {confidence}%")

        # Details
        col_a, col_b = st.columns(2)
        col_a.info(f"Non-Diabetic Probability: {round(probability[0]*100, 2)}%")
        col_b.warning(f"Diabetic Probability: {round(probability[1]*100, 2)}%")

with tab2:
    st.subheader("📊 Model Performance Comparison")

    import pandas as pd

    comparison_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree', 
                  'Random Forest', 'SVM', 'XGBoost', 'ANN'],
        'Accuracy (%)': [96.03, 95.16, 97.00, 96.45, 97.14, 97.19],
        'Remarks': ['Good', 'Good', 'Excellent', 
                    'Very Good', 'Excellent', '🏆 BEST']
    })

    st.dataframe(comparison_df, use_container_width=True)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6','#1abc9c']
    bars = ax.bar(comparison_df['Model'], 
                  comparison_df['Accuracy (%)'], 
                  color=colors, edgecolor='black')

    for bar, acc in zip(bars, comparison_df['Accuracy (%)']):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.05,
                f'{acc}%', ha='center', fontweight='bold')

    ax.set_ylim(93, 99)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align: center; color: gray;'>
    AI-Based Multi-Model Prediction System | 
    Amity University Noida | Shivam | A2345924083
    </p>
""", unsafe_allow_html=True)