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


HF_MODEL2 = (
    "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Choose a model suitable for text analysis
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

    ### **🔹 Goal: Optimize Fuel Efficiency**
    - Suggest **new values** for each attribute (acceleration, displacement, weight, horsepower, cylinders).
    - Explain **why each change** will improve fuel economy.
    - Provide **a numerical comparison** (e.g., "Reducing weight by 10% can improve fuel efficiency by ~5%").
    - Suggest **real-world solutions** (e.g., using aluminum body panels to reduce weight).
    

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
            "max_new_tokens": 800,  # Ensure enough space for output
            "temperature": 0.5,  # Higher creativity for better suggestions
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
        prompt = f"""You are a **car expert chatbot**.  
- Answer **only car-related questions**.  
- Do **not** ask questions or start conversations.  
- Do **not include "User:" in responses**.  
- If a question is **not about cars**, reply: "I only answer car-related questions."  
- Keep responses **concise, accurate, and professional**.  

User: {user_input}  
AI:"""

        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.5,
                "do_sample": True,
                "return_full_text": False,
            },
        }

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL2}",
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
import streamlit as st


def set_custom_css():
    custom_css = """
    <style>
        /* Global Font and Darker Background */
        html, body, [class*="st-"] {
            font-family: "Times New Roman", serif;
            color: #f8f8ff;
            font-size: 19px !important; /* Decreased from 22px */
        }
        .stApp {
            background-color: #0d0d0d !important;
        }

        /* Animated Welcome Message */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0px); }
        }
        .welcome-message {
            font-size: 37px; /* Decreased from 40px */
            font-weight: bold;
            color: #ffaa00;
            text-align: center;
            animation: fadeIn 2s ease-in-out;
            margin-bottom: 20px;
        }

        /* Titles & Headers */
        .stTitle, .stHeader, h1, h2, h3 {
            font-size: 35px !important; /* Decreased from 38px */
            font-weight: bold;
            color: #f8f8ff;
        }

        /* 🚀 AI Response - Darker Background */
        .ai-response {
            background: linear-gradient(135deg, rgba(5, 5, 5, 0.98), rgba(0, 0, 0, 1));
            border: 2px solid rgba(255, 204, 0, 1);
            color: #ffcc00;
            font-size: 21px; /* Decreased from 24px */
            line-height: 1.5;
            padding: 14px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(255, 204, 0, 0.5);
            transition: all 0.3s ease-in-out;
            animation: slideUp 0.8s ease-in-out;
        }
        .ai-response:hover {
            box-shadow: 0px 4px 20px rgba(255, 204, 0, 0.7);
            transform: scale(1.03);
        }

        /* 🚀 Input Label Size */
        div[data-testid="stTextInput"] label,  
        div[data-testid="stNumberInput"] label,  
        div[data-testid="stSelectbox"] label,  
        div[data-testid="stSlider"] label {
            font-size: 29px !important; /* Decreased from 32px */
            font-weight: bold !important;
            background: linear-gradient(90deg, #ffaa00, #ff5500);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
            animation: fadeIn 1.5s ease-in-out, pulse 2s infinite;
        }

        /* Input Field Styling */
        input, textarea, select {
            background-color: #050505 !important;
            color: #f8f8ff !important;
            border: 2px solid #777 !important;
            font-family: "Times New Roman", serif;
            font-size: 19px !important; /* Decreased from 22px */
            padding: 10px;
            border-radius: 12px;
            transition: all 0.3s ease-in-out;
        }

        /* Hover Glow Effect */
        input:focus, textarea:focus, select:focus {
            border-color: #ffaa00 !important;
            box-shadow: 0px 0px 12px rgba(255, 170, 0, 0.8);
            outline: none;
        }

        /* Gradient Buttons */
        .stButton>button {
            font-size: 19px !important; /* Decreased from 22px */
            font-weight: bold;
            background: linear-gradient(90deg, #ff6600, #ffcc00);
            color: #1a1a1a !important;
            border: none;
            border-radius: 12px;
            padding: 10px 22px;
            transition: 0.3s ease-in-out;
            box-shadow: 0px 5px 10px rgba(255, 102, 0, 0.4);
        }

        /* Button Hover Effect */
        .stButton>button:hover {
            background: linear-gradient(90deg, #ffcc00, #ff6600);
            box-shadow: 0px 5px 20px rgba(255, 255, 255, 0.6);
            transform: scale(1.07);
        }

        /* Sidebar Styling */
        .stSidebar {
            background-color: #050505 !important;
        }

        /* Modify Gauge Chart Text */
        .gauge-text {
            color: #f8f8ff !important;
            font-size: 25px !important; /* Decreased from 28px */
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0px); }
        }
        @keyframes pulse {
            0% { text-shadow: 0 0 6px #ff6600; }
            50% { text-shadow: 0 0 20px #ffaa00; }
            100% { text-shadow: 0 0 6px #ff6600; }
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0px); }
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Display the Animated Welcome Message
    st.markdown(
        '<h1 class="welcome-message"> || Welcome Fuel Efficiency Predictor || </h1>',
        unsafe_allow_html=True,
    )


# Call function to apply custom styling and welcome message
set_custom_css()


# Set background image with a dark overlay for better visibility
def set_bg_from_url(image_url):
    bg_css = f"""
    <style>
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),  
                        url("{image_url}") no-repeat center center fixed;
            background-size: cover;
        }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)


# Set darkened animated background (Replace with your car image URL)
car_image_url = (
    "https://i.pinimg.com/736x/28/a9/a2/28a9a25559512b25e9b1264543ddcf6b.jpg"
)
set_bg_from_url(car_image_url)

# Streamlit App Title

st.write(
    "Enter vehicle details to predict fuel efficiency (KMPL) and get tips to improve it."
)

# Input fields
acceleration = st.number_input("Acceleration (0-60 mph in sec)",min_value=0)
displacement = st.number_input("Displacement",min_value=0)
weight = st.number_input("Weight",min_value=0)
horsepower = st.number_input("Horsepower",min_value=0)
cylinders = st.number_input("Cylinders",min_value=0,max_value=12)


# Predict Button
if st.button("Predict KMPL"):
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
            title={"text": "KILOMETERS PER LITER (KMPL)"},
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

    st.success(f"✅ Predicted KMPL: {st.session_state.mpg_prediction:.2f} KMPL")


if st.session_state.ai_suggestions:
    st.subheader("💡 AI Suggestions for Better Fuel Efficiency")
    st.write(st.session_state.ai_suggestions)


st.subheader("💬 AI Car Expert Chatbot")
st.write("Ask me anything about cars, engines, fuel efficiency, and maintenance!")

for message in st.session_state.chat_history:
    st.write(message)

chat_with_bot()
