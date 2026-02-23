import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from gtts import gTTS
import base64
import os
from languages import get_sign_text

# --- PAGE SETUP & CUSTOM CSS ---
st.set_page_config(page_title="Road Sign AI", layout="centered", page_icon="🚦")

# Injecting Custom CSS to make it look like a real mobile app
st.markdown("""
    <style>
    /* Hide the Streamlit top menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* App Background and Font */
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Style the Title */
    h1 {
        text-align: center;
        color: #00E676;
        font-weight: 700;
        margin-bottom: 0px;
    }
    
    /* Style the subtext */
    .subtext {
        text-align: center;
        color: #A0A0A0;
        font-size: 16px;
        margin-bottom: 20px;
    }
    
    /* Style the Success/Warning Boxes */
    .stAlert {
        border-radius: 15px !important;
    }
    </style>
    
    <h1>🚦 AI Road Sign Scanner</h1>
    <p class="subtext">Scan a traffic sign for real-time multilingual alerts.</p>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_ai_model():
    try:
        return load_model('models/road_sign_model_final.h5', compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_ai_model()

# --- PREPROCESSING ---
def preprocess_image(image):
    img_array = np.array(image)
    resized = cv2.resize(img_array, (30, 30))
    img_final = np.expand_dims(resized, axis=0) / 255.0
    return img_final

# --- AUDIO GENERATOR ---
def play_audio(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
        tts.save("alert.mp3")
        
        audio_file = open("alert.mp3", "rb")
        audio_bytes = audio_file.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        # Hidden audio player that auto-plays
        audio_html = f"""
            <audio autoplay style="display:none;">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error("Audio generation failed.")

# --- USER INTERFACE ---
st.markdown("### 🌐 Select Language")
lang_map = {'English': 'en', 'Hindi': 'hi', 'Kannada': 'kn', 'Tamil': 'ta', 'Telugu': 'te'}
selected_lang = st.selectbox("", list(lang_map.keys()), label_visibility="collapsed")
lang_code = lang_map[selected_lang]

st.markdown("### 📷 Camera Scanner")
img_file_buffer = st.camera_input("", label_visibility="collapsed")

if img_file_buffer is not None and model is not None:
    image = Image.open(img_file_buffer)
    
    with st.spinner('🔍 Analyzing AI patterns...'):
        img_final = preprocess_image(image)
        prediction = model.predict(img_final, verbose=0)
        
        confidence = np.max(prediction)
        class_id = np.argmax(prediction)

    st.markdown("---")
    
    if confidence > 0.50: 
        translated_text = get_sign_text(class_id, lang_code)
        
        st.success(f"**Detected:** {translated_text}")
        st.info(f"**AI Confidence:** {confidence*100:.1f}%")
        
        play_audio(translated_text, lang_code)
    else:
        st.warning(f"Sign not recognized clearly. (Highest guess: {confidence*100:.1f}%). Please retake the photo.")