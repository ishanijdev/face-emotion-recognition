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
    page_title="EmotionAI | Portfolio Project",
    page_icon="🙂",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS FOR A PROFESSIONAL LOOK ---
# This CSS transforms the default Streamlit appearance into a modern, polished web app.
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F0F4F8; /* A clean, light blue-gray background */
}

/* Hide Streamlit's default hamburger menu and footer */
header, footer {
    visibility: hidden !important;
}

/* Custom header with a professional look */
.custom-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 3rem;
    background-color: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid #E6EAF1;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    position: sticky;
    top: 0;
    left: 0;
    right: 0;
    z-index: 1000;
}
.logo-text {
    font-weight: 700;
    font-size: 1.8rem;
    color: #2c3e50;
}

/* Main content area styling */
.main-content {
    padding: 2rem 3rem;
}

/* Card-like containers for sections */
.section-container {
    background-color: white;
    border-radius: 1rem;
    padding: 2.5rem;
    margin-top: 2rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.05);
}

/* Feature boxes inside the "About" section */
.feature-box {
    background-color: #E6F0F8;
    border: 1px solid #D0E0F0;
    border-radius: 0.75rem;
    padding: 1.2rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
.feature-box:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 15px rgba(0,0,0,0.08);
}
.feature-box span {
    font-size: 1.8rem;
}

/* Button Styling */
.stButton>button {
    background: linear-gradient(45deg, #4B8BBE, #3A6A94);
    color: white;
    border-radius: 0.5rem;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    border: none;
    transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    width: 100%;
}
.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 15px rgba(0,0,0,0.15);
}

/* Custom footer */
.custom-footer {
    text-align: center;
    padding: 2rem 0;
    color: #888;
}
"""
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)


# --- 3. MODEL AND RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    model = load_model('5_class_model.h5')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, face_cascade

model, face_cascade = load_resources()
CLASS_LABELS = ['angry', 'happy', 'neutral', 'sad', 'surprise']


# --- 4. HELPER FUNCTIONS ---
def text_to_audio_autoplay(text):
    try:
        tts = gTTS(text=text, lang='en')
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        b64 = base64.b64encode(mp3_fp.read()).decode()
        audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        print(f"Error generating audio: {e}")

def predict_emotion(image):
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
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (75, 139, 190), 3)
    cv2.putText(img_copy, emotion.capitalize(), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (75, 139, 190), 2)
    return emotion, img_copy

# --- 5. UI RENDERING ---

# Custom Header
st.markdown('<div class="custom-header"><div class="logo-text">🙂 EmotionAI</div></div>', unsafe_allow_html=True)

# Main Content Area
with st.container():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    # Hero Section
    st.title("Real-Time Face Emotion Recognition")
    st.markdown("Experience real-time emotion recognition using deep learning. Interact with your webcam or upload an image and explore how AI interprets human emotions.")
    st.divider()

    # Main Interaction Tabs
    tab1, tab2 = st.tabs(["🖼️ **Upload Image**", "📷 **Live Webcam**"])

    with tab1:
        st.header("Analyze an Image")
        uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        audio_feedback_upload = st.toggle("🔈 Audio Feedback", value=True, key="audio_upload")

        if uploaded_file:
            pil_image = Image.open(uploaded_file)
            opencv_image = np.array(pil_image.convert('RGB'))[:, :, ::-1]
            st.image(pil_image, caption='Your Uploaded Image', use_container_width=True)
            if st.button("Analyze Image", use_container_width=True):
                with st.spinner("Analyzing..."):
                    emotion, result_img = predict_emotion(opencv_image)
                    if emotion:
                        st.success(f"**Predicted Emotion: {emotion.capitalize()}**")
                        st.image(result_img, use_container_width=True)
                        if audio_feedback_upload: text_to_audio_autoplay(f"The detected emotion is {emotion}")
                    else:
                        st.error("No face detected.")

    with tab2:
        st.header("Real-Time Webcam Feed")
        st.write("Click the button below to start your webcam and see live emotion predictions.")
        # Removed the webrtc component as it requires more complex setup and can be less reliable
        # st.camera_input is the most stable built-in option
        img_file_buffer = st.camera_input("Take a picture", label_visibility="collapsed")
        audio_feedback_webcam = st.toggle("🔈 Audio Feedback", value=True, key="audio_webcam")

        if img_file_buffer:
            pil_image = Image.open(img_file_buffer)
            opencv_image = np.array(pil_image.convert('RGB'))[:, :, ::-1]
            with st.spinner("Analyzing..."):
                emotion, result_img = predict_emotion(opencv_image)
                if emotion:
                    st.success(f"**Predicted Emotion: {emotion.capitalize()}**")
                    st.image(result_img, use_container_width=True)
                    if audio_feedback_webcam: text_to_audio_autoplay(f"The detected emotion is {emotion}")
                else:
                    st.error("No face detected.")
    
    # About Section
    with st.container():
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.header("💡 About the Project")
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.write("""
            This project is a smart system that detects and classifies human emotions from facial expressions in real-time. It uses the **FER2013 dataset** to detect 5 emotions: **Happy, Sad, Angry, Surprise**, and **Neutral**.
            Whether for mental health, user experience, or just curiosity, this tool makes emotion-aware interaction simple and insightful!
            """)
        with col2:
            st.markdown("""
            <div class="feature-box"><span>📷</span> <div><strong>Real-Time</strong> — Works instantly with your webcam.</div></div>
            <div class="feature-box"><span>📈</span> <div><strong>Accuracy</strong> — Achieves 73.67% on FER2013.</div></div>
            <div class="feature-box"><span>🔊</span> <div><strong>Audio Feedback</strong> — Spoken output of your emotion.</div></div>
            <div class="feature-box"><span>🖼️</span> <div><strong>Upload Photo</strong> — Detect emotion from an uploaded image.</div></div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Emotions Gallery
    with st.container():
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.header("🔍 Emotions Recognized")
        
        # Create a 'static' folder in your repo and place these images inside.
        image_files = {
            "Angry": "static/angry.jpg", "Happy": "static/happy.jpg", "Sad": "static/sad.jpg",
            "Neutral": "static/neutral.jpg", "Surprise": "static/surprise.jpg"
        }
        
        cols = st.columns(5)
        for i, (emotion, path) in enumerate(image_files.items()):
            try:
                with cols[i]:
                    st.image(path, caption=emotion, use_container_width=True)
            except FileNotFoundError:
                with cols[i]:
                    st.warning(f"Image not found: {path}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Contact Section
    with st.container():
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.header("📬 Contact Us")
        st.write("Email: [ijindal2005@gmail.com](mailto:ijindal2005@gmail.com) | [meharbhanwra1004@gmail.com](mailto:meharbhanwra1004@gmail.com)")
        st.write("GitHub: [github.com/ishanijdev](https://github.com/ishanijdev) | [github.com/meharbhanwra](https://github.com/meharbhanwra)")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True) # Close main-content

# Custom Footer
st.markdown('<div class="custom-footer">Created by Ishani Jindal & Mehar Bhanwra | Powered by FER2013</div>', unsafe_allow_html=True)