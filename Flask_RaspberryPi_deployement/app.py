import os
import cv2
import numpy as np
import time
from flask import Flask, render_template, Response, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import torch
from ultralytics import YOLO

app = Flask(__name__)

# Load models
classification_model = load_model(r'C:\Users\djeri\OneDrive\Desktop\Deploying-deep-learning-model-using-flask-API-main\TILDA_model_efficientNet-B0.h5')
object_detection_model = YOLO(r"C:\Users\djeri\OneDrive\Desktop\stream\best.pt")

# Classification classes
CLASSIFICATION_CLASSES = ["good", "hole", "objects", "oil spot", "thread error"]

latest_prediction = {"mode": "classification", "result": "", "image": None}

# Open laptop camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Could not open camera.")

def classify_frame(frame):
    frame_resized = cv2.resize(frame, (64, 64))
    frame_array = img_to_array(frame_resized)
    frame_array = np.expand_dims(frame_array, axis=0)
    frame_array = preprocess_input(frame_array)
    prediction = classification_model.predict(frame_array)
    class_index = np.argmax(prediction, axis=1)[0]
    return CLASSIFICATION_CLASSES[class_index]

def detect_objects(frame):
    results = object_detection_model(frame)
    detected_objects = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Get the class name directly from the YOLO model
            class_name = object_detection_model.names[class_id]
            
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            detected_objects.append(f"{class_name} ({confidence:.2f})")
    return frame, detected_objects

def predict_frame(mode):
    global latest_prediction
    last_time = 0
    fps_interval = 1 / 2  # 2 FPS

    while True:
        current_time = time.time()
        if current_time - last_time < fps_interval:
            continue

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            continue

        if mode == "classification":
            prediction_text = classify_frame(frame)
            latest_prediction = {"mode": "classification", "result": prediction_text}
            cv2.putText(frame, f"Prediction: {prediction_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            frame, detected_objects = detect_objects(frame)
            latest_prediction = {"mode": "object_detection", "result": detected_objects}

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        latest_prediction["image"] = frame_bytes

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        last_time = current_time

@app.route('/')
def index():
    return render_template('stream.html')

@app.route('/video_feed')
def video_feed():
    mode = request.args.get("mode", "classification")  # Get the mode from the query parameters
    return Response(predict_frame(mode), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    return jsonify({"mode": latest_prediction["mode"], "result": latest_prediction["result"]})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8880)