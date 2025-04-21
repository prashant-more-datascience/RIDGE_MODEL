import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

# ---------------- Custom Styling Function ----------------
def set_custom_css():
    custom_css = """
    <style>
        /* Global Font and Darker Background */
        html, body, [class*="st-"] {
            font-family: "Times New Roman", serif;
            color: #f8f8ff; /* White text for contrast */
            font-size: 19px !important; /* Decreased from 22px */
        }
        .stApp {
            background-color: #800080 !important; /* Purple Background */
        }

        /* Centering the text */
        .center-text {
            text-align: center;
            font-size: 35px;
            font-weight: bold;
            background: linear-gradient(90deg, #FFD700, #ffcc00, #ff6600, #ff0000); /* Gold to Yellow to Orange to Red Gradient for text */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
        }

        /* Animated Welcome Message */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0px); }
        }
        .welcome-message {
            font-size: 37px; /* Decreased from 40px */
            font-weight: bold;
            background: linear-gradient(90deg, #FFD700, #ffcc00, #ff6600, #ff0000); /* Gold to Yellow to Orange to Red Gradient */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            animation: fadeIn 2s ease-in-out;
            margin-bottom: 20px;
        }

        /* Titles & Headers */
        .stTitle, .stHeader, h1, h2, h3 {
            font-size: 35px !important; /* Decreased from 38px */
            font-weight: bold;
            background: linear-gradient(90deg, #FFD700, #ffcc00, #ff6600, #ff0000); /* Gold to Yellow to Orange to Red Gradient for headers */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            color: transparent;
        }

        /* 🚀 Input Label Size */
        div[data-testid="stTextInput"] label,  
        div[data-testid="stNumberInput"] label,  
        div[data-testid="stSelectbox"] label,  
        div[data-testid="stSlider"] label {
            font-size: 29px !important; /* Decreased from 32px */
            font-weight: bold !important;
            background: linear-gradient(90deg, #FFD700, #ffcc00, #ff6600); /* Gold to Yellow to Orange Gradient */
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
            border-color: #ff6600 !important;
            box-shadow: 0px 0px 12px rgba(255, 102, 0, 0.8);
            outline: none;
        }

        /* Gradient Buttons */
        .stButton>button {
            font-size: 19px !important; /* Decreased from 22px */
            font-weight: bold;
            background: linear-gradient(90deg, #FFD700, #ff6600, #ff0000); /* Gold to Orange to Red Gradient for buttons */
            color: #1a1a1a !important;
            border: none;
            border-radius: 12px;
            padding: 10px 22px;
            transition: 0.3s ease-in-out;
            box-shadow: 0px 5px 10px rgba(255, 102, 0, 0.4);
        }

        /* Button Hover Effect */
        .stButton>button:hover {
            background: linear-gradient(90deg, #ff6600, #ff0000, #FFD700);
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
            0% { text-shadow: 0 0 6px #FFD700; }
            50% { text-shadow: 0 0 20px #ff6600; }
            100% { text-shadow: 0 0 6px #FFD700; }
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0px); }
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown('<h1 class="welcome-message"> || Welcome Fuel Efficiency Predictor || </h1>', unsafe_allow_html=True)

# Apply custom CSS and display welcome message
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
    "https://i.pinimg.com/736x/13/a6/e7/13a6e7c7214158c4f676084788520266.jpg"
)
set_bg_from_url(car_image_url)


# ---------------- Load Model ----------------
with open("ridge_model.pkl", "rb") as model_file:
    ridge_model = pickle.load(model_file)
with open("ridgescaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# ---------------- Input Form ----------------
st.markdown('<div class="center-text">Enter vehicle details to predict fuel efficiency (KM/L) .</div>', unsafe_allow_html=True)

acceleration = st.number_input("Acceleration (0-100 mph in sec)", min_value=0,placeholder="Acceleration")
displacement = st.number_input("Displacement", min_value=0,placeholder="Displacement")
weight = st.number_input("Weight", min_value=0,placeholder="Weight")
horsepower = st.number_input("Horsepower", min_value=0,placeholder="Horsepower")
cylinders = st.number_input("Cylinders", min_value=0, max_value=12,placeholder="Cylinders")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Predict ----------------
if st.button("Predict KMPL"):
    input_data = np.array([[acceleration, displacement, weight, horsepower, cylinders]])
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
    st.success(f"\u2705 Predicted KMPL: {st.session_state.mpg_prediction:.2f} KMPL")
