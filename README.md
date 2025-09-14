# Real-Time Face Emotion Recognition System

A deep learning-powered web application that detects and classifies human emotions from a live webcam feed or user-uploaded images in real-time.

## About The Project

This project aims to bridge the gap between human-computer interaction by creating an intelligent system that can understand human emotions through facial expressions. By leveraging the power of computer vision and deep learning, we have developed a responsive and interactive web application that provides real-time emotion analysis.

### Key Features

* *Real-Time Webcam Analysis*: Captures video frames to detect and classify emotions instantly.
* *Image Upload Functionality*: Allows users to upload an image to get an emotion prediction.
* *High Performance*: The system is optimized for a balance between accuracy and real-time usability, ensuring a smooth user experience.
* *Interactive UI*: A clean and responsive web interface built with HTML, CSS, and JavaScript, featuring a dark mode toggle for user comfort.
* *Audio Feedback*: The system utilizes the Web Speech API to provide spoken output of the detected emotion, enhancing accessibility.

## Tech Stack

* Python, Flask
* TensorFlow, Keras
* OpenCV, NumPy
* HTML, CSS, JavaScript

## Model Performance and Results

We developed and evaluated three different models to find the optimal balance between accuracy and real-time performance. The final deployed model is a Convolutional Neural Network (CNN) trained on a refined subset of the *FER2013 dataset. This model was specifically trained on 5 clearly distinguishable emotions (angry, happy, neutral, sad, and surprise*) to ensure high stability and confidence in live predictions.

The model architecture, built using TensorFlow and Keras, consists of multiple convolutional blocks, batch normalization, and dropout layers to ensure robust feature extraction and prevent overfitting.

The final model achieved the following performance on the test dataset:

| Metric    | Score    |
| :-------- | :------- |
| Accuracy  | 73.67%   |
| Precision | 0.7864   |
| Recall    | 0.6768   |
| Loss      | 0.6954   |

These results were achieved on the test partition of the FER2013 dataset.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.x
* pip

### Installation

1.  Clone the repo
    git clone https://github.com/ishanijdev/face-emotion-recognition.git
2.  Install the required packages
    pip install -r requirements.txt
3.  Run the application
    python app.py

## Contributors

This project was a collaborative effort by:

* *Mehar Bhanwra*
    * Dataset Collection, Preprocessing, and Data Augmentation.
    * CNN Model Architecture Design, Training, and Optimization.
    * Model Evaluation and Performance Analysis.

* *Ishani Jindal*
    * CNN Model Architecture Design, Training, and Optimization.
    * Full-Stack Web Application Development (Flask Backend).
    * Responsive User Interface Design (HTML, CSS, JavaScript).
    * Integration of the CNN Model with the Web Interface using OpenCV.

## Acknowledgments

We would like to express our sincere gratitude to our faculty guide, *Dr. Abhishek Singhal, for his invaluable guidance and unwavering support throughout this project. This project was submitted in partial fulfillment of the requirements for the Bachelor of Technology degree at **Amity University Uttar Pradesh*.

## License

Distributed under the MIT License. See LICENSE for more information.
