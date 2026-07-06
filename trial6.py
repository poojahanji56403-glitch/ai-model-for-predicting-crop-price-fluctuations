import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import base64

st.set_page_config(
    page_title="🌾 Crop Dashboard",
    layout="wide"
)

def set_bg(image_file):
    with open(image_file, "rb") as f:
        img_bytes = f.read()
    encoded = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function
set_bg("images/Gemini_Generated_Image_7j3bv67j3bv67j3b.png")

cola,colb,colc =  st.columns(3)
# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


with cola:
# ---------------- LOGIN FUNCTION ----------------
    def login():
        st.title("🔐 Crop Dashboard Login")
        st.title("")
        #st.title("")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            # Hardcoded credentials
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.success("Login Successful ✅")
                st.rerun()
            else:
                st.error("Invalid Username or Password ❌")

    # ---------------- LOGOUT FUNCTION ----------------
    def logout():
        st.session_state.logged_in = False
        st.rerun()

    # ---------------- LOGIN CHECK ----------------
    if not st.session_state.logged_in:
        login()
        st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌾 Navigation")
page = st.sidebar.radio("Go to", ["Analysis", "Prediction"])

if st.sidebar.button("Logout"):
    logout()




if page == "Analysis":
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f4f1de 0%, #e8f5e8 50%, #fff5ee 100%);
            color: #2d5016;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📊 Crop Analysis")

    df = pd.read_csv("crop_price_dataset_100.csv")

    top_crops = df.groupby('Crop Type')['Crop Price'].sum().sort_values(ascending=False)
    top_demand_crops = df.groupby('Crop Type')['Market Demand Index'].mean()
    top_temp_crops = df.groupby('Crop Type')['Temperature (°C)'].mean()
    top_humidity_crops = df.groupby('Crop Type')['Humidity (%)'].mean()
    top_lessfertilizer_crops = df.groupby('Crop Type')['Fertilizer Usage'].mean()

    # ---------- FIRST ROW ----------
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.pie(top_crops, labels=top_crops.index, autopct='%1.1f%%')
        ax1.set_title("Crop Price Distribution")
        ax1.axis('equal')
        st.pyplot(fig1)

    with col2:
        st.header("This analysis calculates the total accumulated crop price for each crop type across the dataset.")
        st.subheader("Useful for:")
        st.subheader("1. High revenue-generating crops")
        st.subheader("2. Market dominance by crop type")
        st.subheader("3. Strategic crop prioritization")

    col3,col4 = st.columns(2)
    with col3:
        st.header("Calculates the average demand index for each crop type.")
        st.subheader("Useful for:")
        st.subheader("1. Select crops with stable demand")
        st.subheader("2. Reduce risk of unsold produce")

        
    
    with col4:
        fig2, ax2 = plt.subplots()
        ax2.plot(top_demand_crops.index, top_demand_crops.values, marker='o')
        ax2.set_title("Market Demand")
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)
    
        
    # ---------- SECOND ROW ----------
    col5, col6 = st.columns(2)

    with col5:
        fig3, ax3 = plt.subplots()
        ax3.pie(top_temp_crops.values, labels=top_temp_crops.index, autopct='%1.1f%%')
        ax3.set_title("Temperature Distribution")
        ax3.axis('equal')
        st.pyplot(fig3)

    with col6:
        st.header("Determines the average temperature conditions under which each crop is cultivated.")
        st.subheader("Useful for:")
        st.subheader("1. Climate-based crop selection")
        st.subheader("2. Region-specific farming decisions")
        st.subheader("3. Predicting crop performance under climate change")    
    
    col7, col8 = st.columns(2)

    with col7:
        st.header("Calculates the average humidity levels associated with each crop type.")
        st.subheader("Useful for:")
        st.subheader("1. Plan irrigation strategies")
        st.subheader("2. Understand crop resilience in dry vs humid regions")

    with col8:
        fig4, ax4 = plt.subplots()
        ax4.bar(top_humidity_crops.index, top_humidity_crops.values)
        ax4.set_title("Humidity by Crop")
        ax4.tick_params(axis='x', rotation=45)
        st.pyplot(fig4)

    col9, col10 = st.columns(2)

    with col9:
        

        fig5, ax5 = plt.subplots()
        ax5.plot(top_lessfertilizer_crops.index, top_lessfertilizer_crops.values, marker='o')
        ax5.set_title("Fertilizer Usage")
        ax5.tick_params(axis='x', rotation=45)
        st.pyplot(fig5)

    with col10:
        st.header("Computes the average fertilizer input required per crop type.")
        st.subheader("Useful for:")
        st.subheader("1. Cost optimization")
        st.subheader("2. Sustainable agriculture planning")
# ======================================================
# ===================== PREDICTION PAGE =================
# ======================================================

elif page == "Prediction":
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f4f1de 0%, #e8f5e8 50%, #fff5ee 100%);
            color: #2d5016;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🤖 Crop Price Prediction System")
    st.markdown("Predict crop price based on environmental and market factors")

    model = joblib.load('rf_regressor.pkl')

    crop_mapping = {
        "Almond": 0,
        "Amaranth": 1,
        "Apple": 2,
        "Arecanut": 3,
        "Ash Gourd": 4,
        "Bajra": 5,
        "Banana": 6,
        "Barley": 7,
        "Beetroot": 8,
        "Bitter Gourd": 9,
        "Black Pepper": 10,
        "Blueberry": 11,
        "Bottle Gourd": 12,
        "Brinjal": 13,
        "Buckwheat": 14,
        "Cabbage": 15,
        "Capsicum": 16,
        "Cardamom": 17,
        "Carrot": 18,
        "Cashew": 19,
        "Castor": 20,
        "Cauliflower": 21,
        "Chana": 22,
        "Cherry": 23,
        "Chilli": 24,
        "Clove": 25,
        "Coconut": 26,
        "Coffee": 27,
        "Coriander": 28,
        "Cotton": 29,
        "Cucumber": 30,
        "Cumin": 31,
        "Custard Apple": 32,
        "Date": 33,
        "Drumstick": 34,
        "Fig": 35,
        "Flaxseed": 36,
        "Garlic": 37,
        "Ginger": 38,
        "Grapes": 39,
        "Groundnut": 40,
        "Guava": 41,
        "Hemp": 42,
        "Horse Gram": 43,
        "Jackfruit": 44,
        "Jowar": 45,
        "Jute": 46,
        "Kidney Beans": 47,
        "Lentils": 48,
        "Lettuce": 49,
        "Litchi": 50,
        "Maize": 51,
        "Mango": 52,
        "Masoor Dal": 53,
        "Millet": 54,
        "Moong Dal": 55,
        "Muskmelon": 56,
        "Mustard": 57,
        "Oats": 58,
        "Okra": 59,
        "Onion": 60,
        "Orange": 61,
        "Papaya": 62,
        "Pear": 63,
        "Peas": 64,
        "Pineapple": 65,
        "Pistachio": 66,
        "Plum": 67,
        "Pomegranate": 68,
        "Potato": 69,
        "Pumpkin": 70,
        "Quinoa": 71,
        "Ragi": 72,
        "Rice": 73,
        "Rubber": 74,
        "Saffron": 75,
        "Sesame": 76,
        "Sorghum": 77,
        "Soybean": 78,
        "Spinach": 79,
        "Strawberry": 80,
        "Sugarcane": 81,
        "Sunflower": 82,
        "Tea": 83,
        "Tobacco": 84,
        "Tomato": 85,
        "Tur Dal": 86,
        "Turmeric": 87,
        "Urad Dal": 88,
        "Walnut": 89,
        "Watermelon": 90,
        "Wheat": 91
    }

    season_mapping = {
        "Kharif": 0,
        "Rabi": 1,
        "Zaid": 2
    }

    st.subheader("🔧 Input Features")

    crop_name = st.selectbox("Select Crop", list(crop_mapping.keys()))
    season_name = st.selectbox("Select Season", list(season_mapping.keys()))

    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=1.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=1.0)
    soil_fertility = st.slider("Soil Fertility Index", 0.0, 1.0, step=0.01)
    fertilizer_usage = st.number_input("Fertilizer Usage (kg/ha)", min_value=0.0, step=1.0)
    market_demand = st.slider("Market Demand Index", 0.0, 1.0, step=0.01)
    transport_cost = st.number_input("Transportation Cost (₹)", min_value=0.0, step=10.0)

    crop_encoded = crop_mapping[crop_name]
    season_encoded = season_mapping[season_name]

    input_data = np.array([[ 
        crop_encoded,
        season_encoded,
        rainfall,
        temperature,
        humidity,
        soil_fertility,
        fertilizer_usage,
        market_demand,
        transport_cost
    ]])

    if st.button("📊 Predict Crop Price"):
        prediction = model.predict(input_data)[0]

        st.success(f"💰 Predicted Crop Price: ₹ {prediction:,.2f}/quintal")
        st.markdown("---")
        st.caption("Model: Voting Regressor | Deployment: Streamlit")