import streamlit as st
import requests
import pandas as pd

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Smart Crop Recommendation Dashboard",
    page_icon="🌱",
    layout="wide"
)

# -------------------------
# Session state defaults
# -------------------------
default_values = {
    "N": 90.0,
    "P": 42.0,
    "K": 43.0,
    "temperature": 20.8,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9
}

rice_preset = {
    "N": 90.0,
    "P": 42.0,
    "K": 43.0,
    "temperature": 20.8,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9
}

wheat_preset = {
    "N": 80.0,
    "P": 40.0,
    "K": 20.0,
    "temperature": 15.0,
    "humidity": 60.0,
    "ph": 6.8,
    "rainfall": 75.0
}

cotton_preset = {
    "N": 120.0,
    "P": 46.0,
    "K": 19.0,
    "temperature": 27.0,
    "humidity": 75.0,
    "ph": 6.2,
    "rainfall": 110.0
}

if "inputs" not in st.session_state:
    st.session_state.inputs = default_values.copy()

# -------------------------
# Header
# -------------------------
st.title("🌱 Smart Crop Recommendation & Risk Advisory System")
st.markdown("""
This system predicts the **most suitable crop** based on soil and environmental conditions,  
and also estimates the **cultivation risk level** with a basic advisory message.
""")

st.markdown("---")

# -------------------------
# Preset section
# -------------------------
st.subheader("🌾 Quick Sample Presets")
st.markdown("Use these preset examples for fast demo/testing, or manually enter values from the sidebar.")

preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

with preset_col1:
    if st.button("Default Sample"):
        st.session_state.inputs = default_values.copy()

with preset_col2:
    if st.button("Rice-like Condition"):
        st.session_state.inputs = rice_preset.copy()

with preset_col3:
    if st.button("Wheat-like Condition"):
        st.session_state.inputs = wheat_preset.copy()

with preset_col4:
    if st.button("Cotton-like Condition"):
        st.session_state.inputs = cotton_preset.copy()

st.markdown("---")

# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.header("📥 Enter Input Parameters")

N = st.sidebar.number_input("Nitrogen (N)", value=st.session_state.inputs["N"])
P = st.sidebar.number_input("Phosphorus (P)", value=st.session_state.inputs["P"])
K = st.sidebar.number_input("Potassium (K)", value=st.session_state.inputs["K"])
temperature = st.sidebar.number_input("Temperature (°C)", value=st.session_state.inputs["temperature"])
humidity = st.sidebar.number_input("Humidity (%)", value=st.session_state.inputs["humidity"])
ph = st.sidebar.number_input("pH", value=st.session_state.inputs["ph"])
rainfall = st.sidebar.number_input("Rainfall (mm)", value=st.session_state.inputs["rainfall"])

predict_button = st.sidebar.button("🚀 Predict Crop & Risk")

# -------------------------
# Input interpretation function
# -------------------------
def interpret_inputs(temp, hum, ph_value, rain):
    insights = []

    # Temperature
    if temp < 15:
        insights.append("🌡️ Cool climate condition detected")
    elif temp <= 30:
        insights.append("🌡️ Moderate to warm climate condition detected")
    else:
        insights.append("🌡️ High temperature condition detected")

    # Humidity
    if hum < 40:
        insights.append("💧 Low humidity environment")
    elif hum <= 75:
        insights.append("💧 Moderate humidity environment")
    else:
        insights.append("💧 High humidity environment")

    # pH
    if ph_value < 6:
        insights.append("🧪 Soil appears acidic")
    elif ph_value <= 7.5:
        insights.append("🧪 Soil pH appears near-neutral / suitable")
    else:
        insights.append("🧪 Soil appears alkaline")

    # Rainfall
    if rain < 50:
        insights.append("🌧️ Low rainfall condition")
    elif rain <= 150:
        insights.append("🌧️ Moderate rainfall condition")
    else:
        insights.append("🌧️ High rainfall condition")

    return insights

# -------------------------
# Main layout
# -------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Input Summary")

    input_df = pd.DataFrame({
        "Parameter": [
            "Nitrogen (N)",
            "Phosphorus (P)",
            "Potassium (K)",
            "Temperature (°C)",
            "Humidity (%)",
            "pH",
            "Rainfall (mm)"
        ],
        "Value": [N, P, K, temperature, humidity, ph, rainfall]
    })

    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("🧠 Input Interpretation")
    insights = interpret_inputs(temperature, humidity, ph, rainfall)

    for item in insights:
        st.info(item)

st.markdown("---")

# -------------------------
# Prediction section
# -------------------------
st.subheader("🌾 Prediction Output")

if predict_button:
    payload = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)

        if response.status_code == 200:
            result = response.json()

            recommended_crop = result["recommended_crop"]
            risk_level = result["risk_level"]
            advisory = result["advisory"]

            st.success("✅ Prediction Successful!")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric(label="🌾 Recommended Crop", value=recommended_crop)

            with result_col2:
                st.metric(label="⚠️ Risk Level", value=risk_level)

            with result_col3:
                st.metric(label="💡 Advisory Status", value="Available")

            st.markdown("### 📌 Advisory Message")
            st.write(advisory)

            st.markdown("### 📝 Risk Interpretation")

            if str(risk_level).lower() == "safe":
                st.success("This input condition appears favorable for cultivation.")
            elif str(risk_level).lower() == "moderate":
                st.warning("Moderate agricultural risk detected. Monitor conditions carefully.")
            elif str(risk_level).lower() == "risky":
                st.error("High agricultural risk detected. Precaution is recommended.")
            else:
                st.info("Risk interpretation unavailable.")

        else:
            st.error(f"API Error: {response.status_code}")
            st.write(response.text)

    except Exception as e:
        st.error("❌ Could not connect to API. Please make sure FastAPI is running.")
        st.write(str(e))

else:
    st.info("Select a preset or manually enter values, then click **Predict Crop & Risk** from the sidebar.")

st.markdown("---")

# -------------------------
# Workflow explanation
# -------------------------
st.subheader("⚙️ System Workflow")
st.markdown("""
This project follows a simple MLOps-based architecture:

1. **User enters input values** manually or using sample presets  
2. **Dashboard sends data to FastAPI backend**  
3. **Backend loads trained ML models**  
4. **Prediction result is returned**  
5. **Dashboard displays crop recommendation and risk analysis**
""")

st.subheader("🛠 Technologies Used")
st.markdown("""
- **Machine Learning:** Scikit-learn  
- **Backend API:** FastAPI  
- **Frontend Dashboard:** Streamlit  
- **MLOps / CI:** GitHub Actions  
- **Model Tracking:** MLflow  
""")