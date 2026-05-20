import os
import pickle
import cv2
import numpy as np
import streamlit as st

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "ASL_Model", "asl_classifier.pkl")
label_map_path = os.path.join(BASE_DIR, "ASL_Model", "label_map.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(label_map_path, "rb") as f:
        label_map = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found! Please run train_model.py first.")
    st.stop()

# Initialize HandLandmarker
hl_model_path = os.path.join(BASE_DIR, "ASL_Model", "hand_landmarker.task")
base_options = python.BaseOptions(model_asset_path=hl_model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
hands = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

st.title("Real-Time Local ASL Recognition")
run_app = st.checkbox("Turn on Webcam", value=False)

FRAME_WINDOW = st.image([])

if run_app:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab hardware camera frame.")
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = hands.detect(mp_image)

        prediction_text = "Show Hand"
        box_color = (255, 0, 0)

        if results.hand_landmarks:
            box_color = (0, 255, 0)
            for hand_landmarks in results.hand_landmarks:
                h, w, _ = img_rgb.shape

                # Draw connections
                for connection in HAND_CONNECTIONS:
                    start = hand_landmarks[connection.start]
                    end = hand_landmarks[connection.end]
                    sx, sy = int(start.x * w), int(start.y * h)
                    ex, ey = int(end.x * w), int(end.y * h)
                    cv2.line(img_rgb, (sx, sy), (ex, ey), (0, 255, 0), 2)

                # Draw landmarks
                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(img_rgb, (cx, cy), 5, (0, 0, 255), -1)

                # Vector extraction
                wrist_x = hand_landmarks[0].x
                wrist_y = hand_landmarks[0].y
                landmark_vector = []
                for landmark in hand_landmarks:
                    landmark_vector.extend(
                        [landmark.x - wrist_x, landmark.y - wrist_y]
                    )

                try:
                    prediction = model.predict([landmark_vector])
                    prediction_text = label_map.get(prediction[0], "Unknown")
                except Exception:
                    prediction_text = "Prediction Error"

        cv2.putText(
            img_rgb,
            f"Prediction: {prediction_text}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            box_color,
            2,
            cv2.LINE_AA,
        )

        FRAME_WINDOW.image(img_rgb)

    cap.release()
    hands.close()
else:
    st.write("Webcam stopped.")
