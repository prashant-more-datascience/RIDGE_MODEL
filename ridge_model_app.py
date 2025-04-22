import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

# ---------------- Custom Styling Function ----------------
def set_custom_css():
    custom_css = """
    <style>
        html, body, [class*="st-"] {
            font-family: "Times New Roman", serif;
            color: #f8f8ff;
            font-size: 19px !important;
        }

        .stApp {
            background-color: #2E0854 !important;
        }

        .center-text {
            text-align: center;
            font-size: 35px;
            font-weight: bold;
            background: linear-gradient(90deg, #00FFFF, #9D00FF, #FF69B4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0px); }
        }

        .welcome-message {
            font-size: 37px;
            font-weight: bold;
            background: linear-gradient(90deg, #00FFFF, #9D00FF, #FF69B4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            animation: fadeIn 2s ease-in-out;
            margin-bottom: 20px;
        }

        .stTitle, .stHeader, h1, h2, h3 {
            font-size: 35px !important;
            font-weight: bold;
            background: linear-gradient(90deg, #00FFFF, #9D00FF, #FF69B4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            color: transparent;
        }

        div[data-testid="stTextInput"] label,  
        div[data-testid="stNumberInput"] label,  
        div[data-testid="stSelectbox"] label,  
        div[data-testid="stSlider"] label {
            font-size: 29px !important;
            font-weight: bold !important;
            background: linear-gradient(90deg, #00FFFF, #9D00FF, #FF69B4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
            animation: fadeIn 1.5s ease-in-out, pulse 2s infinite;
        }

        input, textarea, select {
            background-color: #050505 !important;
            color: #f8f8ff !important;
            border: 2px solid #777 !important;
            font-family: "Times New Roman", serif;
            font-size: 19px !important;
            padding: 10px;
            border-radius: 12px;
            transition: all 0.3s ease-in-out;
        }

        input:focus, textarea:focus, select:focus {
            border-color: #FF69B4 !important;
            box-shadow: 0px 0px 12px rgba(255, 105, 180, 0.8);
            outline: none;
        }

        .stButton>button {
            font-size: 19px !important;
            font-weight: bold;
            background: linear-gradient(90deg, #00FFFF, #9D00FF, #FF69B4);
            color: #1a1a1a !important;
            border: none;
            border-radius: 12px;
            padding: 10px 22px;
            transition: 0.3s ease-in-out;
            box-shadow: 0px 5px 10px rgba(157, 0, 255, 0.4);
        }

        .stButton>button:hover {
            background: linear-gradient(90deg, #FF69B4, #9D00FF, #00FFFF);
            box-shadow: 0px 5px 20px rgba(255, 255, 255, 0.6);
            transform: scale(1.07);
        }

        .stSidebar {
            background-color: #050505 !important;
        }

        .gauge-text {
            color: #f8f8ff !important;
            font-size: 25px !important;
        }

        @keyframes pulse {
            0% { text-shadow: 0 0 6px #00FFFF; }
            50% { text-shadow: 0 0 20px #9D00FF; }
            100% { text-shadow: 0 0 6px #FF69B4; }
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0px); }
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown('<h1 class="welcome-message"> ⛽ Welcome Fuel Efficiency Predictor ⛽ </h1>', unsafe_allow_html=True)

# Apply custom CSS and display welcome message
set_custom_css()

# Set background image with dark overlay
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

# Background image
car_image_url = "https://i.pinimg.com/736x/13/a6/e7/13a6e7c7214158c4f676084788520266.jpg"
set_bg_from_url(car_image_url)

# ---------------- Load Model ----------------
with open("xgb_model.pkl", "rb") as model_file:
    ridge_model = pickle.load(model_file)
with open("ridgescaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# ---------------- Input Form ----------------
st.markdown('<div class="center-text">Enter vehicle details to predict fuel efficiency (KM/L).</div>', unsafe_allow_html=True)

acceleration = st.number_input("Acceleration (0-100 mph in sec)", min_value=0, placeholder="Acceleration")
displacement = st.number_input("Displacement", min_value=0, placeholder="Displacement")
weight = st.number_input("Weight", min_value=0, placeholder="Weight")
horsepower = st.number_input("Horsepower", min_value=0, placeholder="Horsepower")
cylinders = st.number_input("Cylinders", min_value=0, max_value=12, placeholder="Cylinders")

# ---------------- Predict ----------------
if st.button("PREDICT"):
   
    # Combine all inputs including fuel type
    input_data = np.array([[cylinders, displacement, weight, horsepower, acceleration]])
    input_scaled = scaler.transform(input_data)
    st.session_state.mpg_prediction = ridge_model.predict(input_scaled)[0]

# ---------------- Output ----------------
if "mpg_prediction" in st.session_state:
    st.subheader("\U0001F4CA Predicted Fuel Efficiency")
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
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 2,
                "bordercolor": "gray",
            },
        )
    )
    fig.update_layout(
        transition_duration=500,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        template="plotly_dark",
    )
    st.plotly_chart(fig)
    st.success(f"\u2705 Predicted KM/L: {st.session_state.mpg_prediction:.2f} KM/L")
