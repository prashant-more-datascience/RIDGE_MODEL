import streamlit as st
import pickle
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
import time

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


# Function to Get Chatbot Responses
def chat_with_bot():
    # Unique key for chat input (avoids duplicate errors)
    chat_input_key = f"chat_input_{len(st.session_state.chat_history)}"

    user_input = st.text_input(
        "You:", key=chat_input_key, placeholder="Ask a car-related question..."
    )

    if user_input:
        # Store the user's message
        st.session_state.chat_history.append(f"🧑 You: {user_input}")

        # Generate AI Response
        formatted_history = "\n".join(
            st.session_state.chat_history[-5:]
        )  # Keep last 5 messages for context
        prompt = f"{formatted_history}\nUser: {user_input}\nAI:"

        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False,
            },
        }

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            json=payload,
            headers=headers,
        )

        ai_response = (
            response.json()[0]["generated_text"]
            if response.status_code == 200
            else f"Error: {response.status_code} - {response.text}"
        )

        # Store AI Response
        st.session_state.chat_history.append(f"🤖 AI: {ai_response}")

        # Refresh UI to update chat history
        st.rerun()


# Initialize Chat History & Stored Data
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mpg_prediction" not in st.session_state:
    st.session_state.mpg_prediction = None
if "ai_suggestions" not in st.session_state:
    st.session_state.ai_suggestions = None


# Custom CSS for fonts, colors, and stylish input fields
def set_custom_css():
    custom_css = """
    <style>
        /* Set global font to Times New Roman and text color to off-white */
        html, body, [class*="st-"] {
            font-family: "Times New Roman", serif;
            color: #f8f8ff;
        }

        /* Background Styling */
        .stApp {
            background-color: #2b2b2b; /* Dark background */
        }

        /* Titles & Headers */
        .stTitle, .stHeader {
            font-size: 28px;
            font-weight: bold;
            color: #f8f8ff;
        }

        /* Streamlit widget labels */
        label {
            font-size: 18px;
            color: #f8f8ff;
        }

        /* Buttons */
        .stButton>button {
            font-size: 16px;
            font-weight: bold;
            background-color: #4a4a4a;
            color: #f8f8ff;
            border-radius: 8px;
            padding: 10px 20px;
            transition: 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #6a6a6a;
            box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.4);
        }

        /* Input Fields - Black Background with Smooth Borders */
        input, textarea, select {
            background-color: black !important;
            color: #f8f8ff !important;
            border: 2px solid #666 !important;
            font-family: "Times New Roman", serif;
            font-size: 16px;
            padding: 10px;
            border-radius: 8px;
            transition: all 0.3s ease-in-out;
        }

        /* Glow Effect on Focus */
        input:focus, textarea:focus, select:focus {
            border-color: #ffcc00 !important;
            box-shadow: 0px 0px 10px rgba(255, 204, 0, 0.6);
            outline: none;
        }

        /* Sidebar Styling */
        .stSidebar {
            background-color: #222;
        }

        /* Modify Gauge Chart Text */
        .gauge-text {
            color: #f8f8ff !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# Call function to apply custom styling
set_custom_css()


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
    # Custom CSS for fonts and colors


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
cylinders = st.slider("Cylinders", 2, 8, 6)


# Predict Button
if st.button("Predict MPG & Get Suggestions"):
    # Prepare input data
    input_data = np.array([[acceleration, displacement, weight, horsepower, cylinders]])

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    st.session_state.mpg_prediction = ridge_model.predict(input_scaled)[0]

    # Get LLM suggestions
    llm_response = suggest_car_modifications(
        acceleration, displacement, weight, horsepower, cylinders
    )

    st.session_state.ai_suggestions = llm_response


# **Ensure Prediction & AI Suggestions Remain Visible**
if st.session_state.mpg_prediction is not None:
    st.subheader("📊 Predicted Fuel Efficiency")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=st.session_state.mpg_prediction,
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

    st.success(f"✅ Predicted MPG: {st.session_state.mpg_prediction:.2f} MPG")


if st.session_state.ai_suggestions:
    st.subheader("💡 AI Suggestions for Better Fuel Efficiency")
    st.write(st.session_state.ai_suggestions)


st.subheader("💬 AI Car Expert Chatbot")
st.write("Ask me anything about cars, engines, fuel efficiency, and maintenance!")

for message in st.session_state.chat_history:
    st.write(message)

chat_with_bot()
