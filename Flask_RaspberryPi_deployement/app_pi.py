import os
import cv2
import numpy as np
import time
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
model = load_model(r'C:\Users\djeri\OneDrive\Desktop\Deploying-deep-learning-model-using-flask-API-main\TILDA_model_efficientNet-B0.h5')

# Define the camera stream URL
CAMERA_URL = 'http://150.250.219.136/webcam/?action=stream'
CLASSES = ["good", "hole", "objects", "oil spot", "thread error"]

# Function to get a frame from the camera stream
def get_frame():
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame.")
        return None

    return frame

# Function to preprocess the frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))  # Resize to model input shape
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    return frame

# Function to generate video frames with predictions
def predict_frame():
    last_time = 0
    fps_interval = 1 / 2  # 2 FPS (one frame every 0.5 seconds)

    while True:
        current_time = time.time()
        if current_time - last_time < fps_interval:
            continue  # Limit frame rate to 2 FPS

        frame = get_frame()
        if frame is None:
            continue  # Skip iteration if no frame

        processed_frame = preprocess_frame(frame)
        prediction = model.predict(processed_frame)
        class_index = np.argmax(prediction, axis=1)[0]
        prediction_text = CLASSES[class_index]

        # Overlay prediction text on the frame
        cv2.putText(frame, f"Prediction: {prediction_text}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        last_time = current_time  # Update time

# Route to display the HTML page
@app.route('/')
def index():
    return render_template('stream.html')

# Route to stream the video with predictions
@app.route('/video_feed')
def video_feed():
    return Response(predict_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)
