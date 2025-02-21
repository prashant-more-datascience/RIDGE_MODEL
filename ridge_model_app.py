import streamlit as st
import pickle
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# Load the trained Ridge model and StandardScaler
with open("ridge_model.pkl", "rb") as model_file:
    ridge_model = pickle.load(model_file)

with open("ridgescaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Hugging Face API Configuration
HF_API_KEY = "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"  # Replace with your actual Hugging Face API key
HF_MODEL = (
    "mistralai/Mistral-7B-Instruct-v0.3"  # Choose a model suitable for text analysis
)


def suggest_car_modifications(
    acceleration, displacement, weight, horsepower, cylinders
):
    prompt = f"""
   A car has the following specifications:
- 🏎 **Acceleration:** {acceleration} sec (0-60 mph)
- 🛠 **Engine Displacement:** {displacement} cc
- ⚖️ **Weight:** {weight} kg
- 🔥 **Horsepower:** {horsepower} HP
- 🔩 **Cylinders:** {cylinders}

### **🔹 Optimize for Better Fuel Efficiency**
Suggest optimized values for these attributes to improve fuel efficiency.
#### **Suggested Adjustments to Car Specs**
- Recommend **new values** for acceleration, displacement, weight, horsepower, and cylinders.  
- Justify **why each change** will increase fuel economy.  
- Provide a **numerical comparison** (e.g., “Reducing weight by 10% can improve fuel efficiency by ~5%”).  

#### **Specific Component Upgrades**
- Suggest **exact engine modifications** (e.g., downsizing, hybrid conversion, turbocharging).  
- Recommend **transmission improvements** (e.g., switching to CVT, dual-clutch, or 8-speed automatic).   
- Propose **fuel system optimizations** (e.g., fuel injectors, eco-friendly fuel types).  

#### ** Cost-Effective Modifications**
- If the user **cannot afford a full engine upgrade**, suggest **smaller changes** like tire pressure optimization, better engine tuning, or fuel additives.  
- Explain which **changes give the highest improvement for the lowest cost**.  

💡 **Provide practical and accurate suggestions. Avoid general answers.**
    """

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 700,  # Ensure enough space for output
            "temperature": 0.7,  # Higher creativity for better suggestions
            "do_sample": True,  # Prevent deterministic responses
            "return_full_text": False,  # ✅ THIS STOPS ECHOING PROMPT!
        },
    }

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        json=payload,
        headers=headers,
    )

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error: {response.status_code} - {response.text}"


# Set background image with animation
def set_bg_from_url(image_url):
    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)


# Set animated background (Replace with your car image URL)
car_image_url = "https://i.pinimg.com/736x/d9/4a/64/d94a643f26453333ad4354daad504b9c.jpg"  # Example URL
set_bg_from_url(car_image_url)

# Streamlit App Title
st.title("🚗 Fuel Efficiency Prediction (MPG) & Optimization Tips")
st.write(
    "Enter vehicle details to predict fuel efficiency (MPG) and get tips to improve it."
)

# Input fields
acceleration = st.number_input("Acceleration (0-60 mph in sec)")
displacement = st.number_input("Displacement", min_value=0.0, format="%.2f")
weight = st.number_input("Weight", min_value=0.0, format="%.2f")
horsepower = st.number_input("Horsepower", min_value=0.0, format="%.2f")
cylinders = st.slider("Cylinders", 2, 12, 6)

# Predict Button
if st.button("Predict MPG & Get Suggestions"):
    # Prepare input data
    input_data = np.array([[acceleration, displacement, weight, horsepower, cylinders]])

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    mpg_prediction = ridge_model.predict(input_scaled)[0]

    # Display result
    st.success(f"Predicted MPG: {mpg_prediction:.2f}")

    # Visualization of Predicted MPG using Gauge Chart
    st.subheader("📊 Predicted Fuel Efficiency (Gauge Chart)")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=mpg_prediction,
            title={"text": "Miles Per Gallon (MPG)"},
            gauge={
                "axis": {"range": [0, 50]},
                "bar": {"color": "purple"},
                "steps": [
                    {"range": [0, 15], "color": "#ff4d4d"},
                    {"range": [15, 30], "color": "#ffd633"},
                    {"range": [30, 50], "color": "#33cc33"},
                ],
                "bgcolor": "rgba(0,0,0,0)",  # Fully transparent background
                "borderwidth": 2,
                "bordercolor": "gray",
            },
        )
    )
    fig.update_layout(
        transition_duration=500,  # Smooth animation effect
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        font=dict(color="white"),  # Adjust font color for better visibility
        template="plotly_dark",
    )
    st.plotly_chart(fig)

    # Get LLM suggestions
    llm_response = suggest_car_modifications(
        acceleration, displacement, weight, horsepower, cylinders
    )

    # Display results
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.subheader("🔍 Prediction Results")
    st.write(f"**Predicted MPG (Miles Per Gallon):** {mpg_prediction:.2f}")

    st.subheader("💡 AI Suggestions for Better Fuel Efficiency")
    st.write(llm_response)
    st.markdown("</div>", unsafe_allow_html=True)
