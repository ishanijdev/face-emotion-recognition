from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64

app = Flask(__name__)

# Load the model
model = load_model('5_class_model.h5')
class_labels = ['angry', 'happy', 'neutral', 'sad', 'surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']

        # Decode base64 image
        encoded_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(encoded_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return jsonify({'emotion': 'No face detected'})

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi_normalized = roi_resized / 255.0
            roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

            prediction = model.predict(roi_reshaped)
            label_index = np.argmax(prediction)
            label = class_labels[label_index]

            return jsonify({'emotion': label})

        return jsonify({'emotion': 'No face detected'})

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'emotion': 'Error processing image'})

if __name__ == '__main__':
    app.run(debug=True)
