import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Employee Salary Prediction Multi-model comparision", layout="centered", initial_sidebar_state="expanded")
st.title("Employee Salary Prediction Multi-model comparision")
st.write("This app allows you to compare different regression models for predicting employee salaries based on various features. You can select the model you want to use, and the app will display the performance metrics and visualizations for that model.")

base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, 'model_results')

@st.cache_resource

def load_model():

    linear_model = joblib.load(os.path.join(model_dir, 'linear_regression_model.pkl'))
    decision_model = joblib.load(os.path.join(model_dir, 'decision_tree_model.pkl'))
    random_forest_model = joblib.load(os.path.join(model_dir,'random_forest_model.pkl'))
    standard_scaler_model = joblib.load(os.path.join(model_dir,'standard_scaler.pkl'))
    min_max_scaler_model = joblib.load(os.path.join(model_dir,'min_max_scaler.pkl'))
    encoder = joblib.load(os.path.join(model_dir,'label_encoder.pkl'))

    return linear_model,decision_model,random_forest_model,standard_scaler_model,min_max_scaler_model,encoder

linear_model,decision_model,random_forest_model,standard_scaler_model,min_max_scaler_model,encoder = load_model()

st.title("💰 Employee Salary Prediction")

st.write(
    "Predict an employee's annual salary using "
    "Machine Learning."
)

st.divider()

st.sidebar.header("⚙️ Model Settings")

model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ]
)



st.subheader("👤 Employee Information")


col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    years_experience = st.number_input(
        "Years of Experience",
        min_value=0,
        max_value=60,
        value=5,
        step=1
    )

    num_skills = st.number_input(
        "Number of Skills",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )


with col2:

    education_level = st.selectbox(
        "Education Level",
        encoder.classes_
    )

    performance_score = st.number_input(
        "Performance Score",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )

    remote_work = st.number_input(
        "Remote Work",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=1.0
    )


st.divider()


# =========================================================
# Prediction Button
# =========================================================

if st.button(
    "🔮 Predict Salary",
    use_container_width=True
):

    # -----------------------------------------------------
    # Encode Education
    # -----------------------------------------------------

    education_encoded = encoder.transform(
        [education_level]
    )[0]


    # -----------------------------------------------------
    # Create Input DataFrame
    # -----------------------------------------------------

    input_data = pd.DataFrame({

        "age": [age],

        "years_experience": [
            years_experience
        ],

        "education_level": [
            education_encoded
        ],

        "performance_score": [
            performance_score
        ],

        "num_skills": [
            num_skills
        ],

        "remote_work": [
            remote_work
        ]
    })


    # -----------------------------------------------------
    # Select Model
    # -----------------------------------------------------

    if model_name == "Linear Regression":

        model = linear_model

    elif model_name == "Decision Tree":

        model = decision_model

    else:

        model = random_forest_model


    # -----------------------------------------------------
    # Make Prediction
    # -----------------------------------------------------

    prediction_scaled = model.predict(
        input_data
    )


    # -----------------------------------------------------
    # Convert Standardized Salary Back to USD
    # -----------------------------------------------------

    prediction_usd = standard_scaler_model.inverse_transform(
        np.array(prediction_scaled).reshape(-1, 1)
    )[0][0]


    # -----------------------------------------------------
    # Display Result
    # -----------------------------------------------------

    st.success("Prediction Completed Successfully!")

    st.metric(
        label="💵 Predicted Annual Salary",
        value=f"${prediction_usd:,.2f}"
    )


    # -----------------------------------------------------
    # Display Input Data
    # -----------------------------------------------------

    st.subheader("📊 Employee Details")

    display_data = pd.DataFrame({

        "Age": [age],

        "Experience": [
            years_experience
        ],

        "Education": [
            education_level
        ],

        "Performance Score": [
            performance_score
        ],

        "Number of Skills": [
            num_skills
        ],

        "Remote Work": [
            remote_work
        ]
    })

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )
