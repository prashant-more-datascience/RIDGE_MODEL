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

# Function to Check if Question is Car-Related
def is_car_related(user_input):
    keywords = [
        "car",
        "engine",
        "fuel",
        "horsepower",
        "mpg",
        "mileage",
        "brake",
        "transmission",
        "tires",
        "battery",
        "speed",
        "torque",
        "oil",
        "hybrid",
        "electric",
        "vehicle",
        "diesel",
        "gasoline",
    ]
    return any(word in user_input.lower() for word in keywords)


# Function to Get Chatbot Responses
def chat_with_bot(user_input):
    if not is_car_related(user_input):
        return "🚫 Sorry, I can only answer car-related questions."

    prompt = f"You are a car expert assistant. Only answer car-related questions. If the question is not related to cars, respond with an apology. \n\nUser: {user_input}\nAI:"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
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

    return (
        response.json()[0]["generated_text"]
        if response.status_code == 200
        else f"Error: {response.status_code} - {response.text}"
    )
# Function to create a Static Gauge Chart
def create_gauge_chart(mpg_value):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=mpg_value,
            title={"text": "Miles Per Gallon (MPG)", "font": {"size": 24}},
            gauge={
                "axis": {"range": [0, 50], "tickwidth": 1, "tickcolor": "white"},
                "bar": {"color": "purple"},
                "steps": [
                    {"range": [0, 15], "color": "#ff4d4d"},
                    {"range": [15, 30], "color": "#ffd633"},
                    {"range": [30, 50], "color": "#33cc33"},
                ],
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 2,
                "bordercolor": "gray",
            },
            number={"font": {"size": 40}},
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        font=dict(color="white"),
        template="plotly_dark",
    )
    return fig


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
cylinders = st.slider("Cylinders", 2, 12, 6)

# Placeholder for Gauge Chart
chart_placeholder = st.empty()

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

  

    # Get LLM suggestions
    llm_response = suggest_car_modifications(
        acceleration, displacement, weight, horsepower, cylinders
    )

    # Display results
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.subheader("🔍 Prediction Results")
    st.write(f"**Predicted MPG (Miles Per Gallon):** {mpg_prediction:.2f}")

    # Display Final Gauge Chart
   chart_placeholder.plotly_chart(
        create_gauge_chart(mpg_prediction), use_container_width=True, key="gauge_final"
    )

    st.subheader("💡 AI Suggestions for Better Fuel Efficiency")
    st.write(llm_response)
    st.markdown("</div>", unsafe_allow_html=True)


st.subheader("💬 AI Car Expert Chatbot")
st.write("Ask me anything about cars, engines, fuel efficiency, and maintenance!")

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat Input
user_input = st.text_input(
    "You:", key="chat_input", placeholder="Ask a car-related question..."
)

# Ask Button
if st.button("Ask AI"):
    if user_input:
        # Add User Message to Chat History
        st.session_state.chat_history.append(f"🧑 You: {user_input}")

        # Get AI Response
        ai_response = chat_with_bot(user_input)

        # Add AI Response to Chat History
        st.session_state.chat_history.append(f"🤖 AI: {ai_response}")

# Display Chat History
if st.session_state.chat_history:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.chat_history[-5:]:  # Show last 5 messages
        st.write(message)
    st.markdown("</div>", unsafe_allow_html=True)

