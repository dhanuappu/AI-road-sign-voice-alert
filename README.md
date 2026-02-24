# 🚦 Real-Time Road Sign Recognition & Voice Alert System

A computer vision system that detects traffic signs in real-time, translates them into 5 Indian languages (English, Hindi, Kannada, Tamil, Telugu), and provides audio alerts for driver safety.

## 🌟 Features
* **Real-Time Detection:** Uses a custom CNN model trained on the GTSRB dataset.
* **Multi-Language Support:** Translates sign meanings instantly.
* **Voice Alerts:** online text-to-speech engine (`gtts`) for driver warnings.
* **Safety First:** Ignores detection if a face is obstructing the view.

## 🛠️ Tech Stack
* **Core:** Python 3.9+
* **AI/ML:** TensorFlow (CPU), OpenCV, Keras
* **Web:** Flask, HTML5, Bootstrap
* **Audio:** Pygame, gtts

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python app.py`

## dataset used 
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
