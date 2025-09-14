import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from gtts import gTTS
import base64
from io import BytesIO

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EmotionAI - Real-Time Emotion Recognition",
    page_icon="🙂",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- 2. CUSTOM CSS STYLING ---
# We inject custom CSS to override Streamlit's default styles and create a modern look.
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# You would create a style.css file with this content
# For simplicity, we'll define it here as a string.
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

body {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background-image: url("https://images.unsplash.com/photo-1513151233558-d860c5398176?q=80&w=2070&auto=format&fit=crop");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

.main-container {
    background-color: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 10px 20px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: #FFFFFF;
    color: #000000;
}

.stButton>button {
    background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    font-weight: 700;
    font-size: 16px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px 0 rgba(254, 107, 139, 0.75);
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 20px 0 rgba(254, 107, 139, 0.85);
}

h1, h2, h3, p, label {
    color: white !important;
}

.stSpinner > div > div {
    border-top-color: #FE6B8B !important;
}

"""
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)


# --- 3. MODEL AND RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    """Loads the pre-trained model and face cascade classifier."""
    model = load_model('5_class_model.h5')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, face_cascade

model, face_cascade = load_resources()
CLASS_LABELS = ['angry', 'happy', 'neutral', 'sad', 'surprise']


# --- 4. HELPER FUNCTIONS ---
def text_to_audio_autoplay(text):
    """Generates audio from text and creates an auto-playing HTML audio tag."""
    try:
        tts = gTTS(text=text, lang='en')
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        b64 = base64.b64encode(mp3_fp.read()).decode()
        audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not generate audio feedback: {e}")

def predict_emotion(image):
    """Detects a face, preprocesses it, and predicts the emotion."""
    img_copy = image.copy()
    gray_image = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    (x, y, w, h) = faces[0]
    face_roi = gray_image[y:y+h, x:x+w]
    roi_resized = cv2.resize(face_roi, (48, 48))
    roi_normalized = roi_resized / 255.0
    roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

    prediction = model.predict(roi_reshaped)
    label_index = np.argmax(prediction)
    emotion = CLASS_LABELS[label_index]
    
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(img_copy, emotion.capitalize(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return emotion, img_copy

# --- 5. MAIN APP INTERFACE ---
with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>🙂 EmotionAI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Uncover emotions in real-time. Choose your method below.</p>", unsafe_allow_html=True)
    
    st.divider()

    tab1, tab2 = st.tabs(["🖼️ Upload an Image", "📷 Live Webcam"])

    # --- IMAGE UPLOAD TAB ---
    with tab1:
        st.subheader("Analyze an Image from Your Device")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file:
            pil_image = Image.open(uploaded_file)
            opencv_image = np.array(pil_image.convert('RGB'))[:, :, ::-1]
            
            st.image(pil_image, caption='Your Uploaded Image', use_container_width=True)
            
            if st.button("Analyze Image", use_container_width=True):
                with st.spinner("Detecting emotion..."):
                    emotion, result_img = predict_emotion(opencv_image)
                    if emotion:
                        st.success(f"**Predicted Emotion: {emotion.capitalize()}**")
                        st.image(result_img, caption="Analysis Result", use_container_width=True)
                        text_to_audio_autoplay(f"The detected emotion is {emotion}")
                    else:
                        st.error("No face detected. Please try another image.")

    # --- WEBCAM TAB ---
    with tab2:
        st.subheader("Analyze Emotion from Your Webcam")
        img_file_buffer = st.camera_input("Take a picture to analyze", label_visibility="collapsed")
        
        if img_file_buffer:
            pil_image = Image.open(img_file_buffer)
            opencv_image = np.array(pil_image.convert('RGB'))[:, :, ::-1]
            
            with st.spinner("Detecting emotion..."):
                emotion, result_img = predict_emotion(opencv_image)
                if emotion:
                    st.success(f"**Predicted Emotion: {emotion.capitalize()}**")
                    st.image(result_img, caption="Analysis Result", use_container_width=True)
                    text_to_audio_autoplay(f"The detected emotion is {emotion}")
                else:
                    st.error("No face detected. Please try again.")

    st.markdown("</div>", unsafe_allow_html=True)

# --- 6. FOOTER ---
st.markdown("<div style='text-align: center; color: white; margin-top: 2rem; font-weight: bold;'>Created by Ishani Jindal & Mehar Bhanwra</div>", unsafe_allow_html=True)