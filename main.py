import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)

# -------------------------------
# Load Model and Preprocessor
# -------------------------------
@st.cache_resource
def load_artifacts():
    model_data = joblib.load("model.joblib")
    preprocessor = joblib.load("preprocessing.joblib")
    
    # Handles dictionary or direct model saving from joblib
    if isinstance(model_data, dict):
        model = model_data.get("XGboost") or list(model_data.values())[0]
    else:
        model = model_data

    return model, preprocessor

try:
    model, preprocessor = load_artifacts()
except Exception as e:
    st.error("Unable to load the model or preprocessor files. Make sure 'model.joblib' and 'preprocessor.joblib' are in the same directory.")
    st.error(str(e))
    st.stop()

# -------------------------------
# Frontend Design & Inputs
# -------------------------------
st.title("✈️ Tourism Package Takeup Prediction")
st.write(
    "Provide customer profile and pitch details below to predict "
    "whether they are likely to purchase the travel package."
)

st.divider()

with st.form("prediction_form"):
    st.subheader("👤 Customer & Pitch Information")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        
        type_of_contact = st.selectbox(
            "Type of Contact",
            ["Self Enquiry", "Company Invited"]
        )
        
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        
        duration_of_pitch = st.number_input(
            "Duration of Pitch (Minutes)",
            min_value=1.0,
            value=15.0,
            step=1.0
        )
        
        occupation = st.selectbox(
            "Occupation",
            ["Salaried", "Free Lancer", "Small Business", "Large Business"]
        )

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        
        number_of_followups = st.number_input(
            "Number of Follow-ups",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )
        
        product_pitched = st.selectbox(
            "Product Pitched",
            ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
        )
        
        preferred_property_star = st.selectbox(
            "Preferred Property Star",
            [3.0, 4.0, 5.0]
        )
        
        marital_status = st.selectbox(
            "Marital Status",
            ["Unmarried", "Married", "Divorced"]
        )

    with col3:
        number_of_trips = st.number_input(
            "Number of Trips (Annually)",
            min_value=0,
            max_value=30,
            value=3,
            step=1
        )
        
        passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        
        own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        monthly_income = st.number_input(
            "Monthly Income ($)",
            min_value=0.0,
            value=22000.0,
            step=500.0
        )

    st.subheader("👨‍👩‍👧 Family & Visiting Details")
    col4, col5 = st.columns(2)
    
    with col4:
        number_of_person_visiting = st.number_input(
            "Number of Persons Visiting",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )

    with col5:
        number_of_children_visiting = st.number_input(
            "Number of Children Visiting (Age < 5)",
            min_value=0,
            max_value=5,
            value=1,
            step=1
        )

    st.divider()
    predict_button = st.form_submit_button("🔍 Predict Package Takeup", use_container_width=True)

# -------------------------------
# Prediction Logic
# -------------------------------
if predict_button:
    # Feature engineering applied during training: TotalVisiting = Persons + Children
    total_visiting = number_of_person_visiting + number_of_children_visiting

    # Construct dataframe using the EXACT feature column names expected by preprocessor
    input_data = pd.DataFrame({
        "Age": [age],
        "TypeofContact": [type_of_contact],
        "CityTier": [city_tier],
        "DurationOfPitch": [duration_of_pitch],
        "Occupation": [occupation],
        "Gender": [gender],
        "NumberOfFollowups": [number_of_followups],
        "ProductPitched": [product_pitched],
        "PreferredPropertyStar": [preferred_property_star],
        "MaritalStatus": [marital_status],
        "NumberOfTrips": [number_of_trips],
        "Passport": [passport],
        "PitchSatisfactionScore": [pitch_satisfaction_score],
        "OwnCar": [own_car],
        "Designation": ["Manager"], # Default safe fallback if designation isn't collected via UI, or add a selector if needed
        "MonthlyIncome": [monthly_income],
        "TotalVisiting": [total_visiting]
    })

    try:
        # Transform input features using the saved preprocessor pipeline
        processed_data = preprocessor.transform(input_data)

        # Run prediction
        prediction = model.predict(processed_data)[0]

        st.divider()
        st.subheader("📊 Prediction Result")

        if int(prediction) == 1:
            st.success("🎉 Prediction: Customer Will Take the Package (`ProdTaken = 1`)")
            st.write("The model suggests high interest from this customer profile.")
        else:
            st.warning("⚠️ Prediction: Customer Will NOT Take the Package (`ProdTaken = 0`)")
            st.write("The model suggests this customer is unlikely to purchase.")

        # Display probability breakdown if available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(processed_data)[0]
            st.write("### Confidence Breakdown")
            for class_label, probability in zip(model.classes_, probabilities):
                label = "Will Take Package" if int(class_label) == 1 else "Will Not Take Package"
                st.write(f"**{label}**: {probability:.2%}")

    except Exception as e:
        st.error("Prediction failed due to mismatched features or preprocessing layout.")
        st.code(str(e))
        st.info("Ensure all training columns match the feature mapping in your pipeline.")

