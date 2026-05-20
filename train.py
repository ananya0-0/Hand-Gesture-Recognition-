import os
import cv2
import numpy as np
import pickle
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

zip_path = os.path.join(BASE_DIR, 'asl_alphabet_train.zip')
dataset_dir = os.path.join(BASE_DIR, 'asl_alphabet_train')

if not os.path.exists(dataset_dir):
    print("Extracting dataset archive locally...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)
    print("Extraction complete!")
else:
    print("Dataset folder already exists. Skipping extraction.")

model_save_dir = os.path.join(BASE_DIR, 'ASL_Model')
os.makedirs(model_save_dir, exist_ok=True)
model_path = os.path.join(model_save_dir, 'asl_classifier.pkl')
label_map_path = os.path.join(model_save_dir, 'label_map.pkl')

# Initialize HandLandmarker
hl_model_path = os.path.join(model_save_dir, 'hand_landmarker.task')
base_options = python.BaseOptions(model_asset_path=hl_model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7
)
hands = vision.HandLandmarker.create_from_options(options)

def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = hands.detect(mp_image)
    if results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]
        landmark_vector = []
        wrist_x = hand_landmarks[0].x
        wrist_y = hand_landmarks[0].y
        for landmark in hand_landmarks:
            landmark_vector.extend([landmark.x - wrist_x, landmark.y - wrist_y])
        return np.array(landmark_vector)
    return None

def train_model():
    features = []
    labels = []
    label_map = {}
    current_label_id = 0

    all_class_dirs = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

    print(f"Processing {len(all_class_dirs)} classes...")

    for class_name in all_class_dirs:
        if class_name in ['nothing', 'space', 'del'] or len(class_name) > 1:
            continue

        if class_name not in label_map:
            label_map[class_name] = current_label_id
            current_label_id += 1

        class_path = os.path.join(dataset_dir, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for image_file in image_files[:30]:
            feature_vector = extract_features(os.path.join(class_path, image_file))
            if feature_vector is not None:
                features.append(feature_vector)
                labels.append(label_map[class_name])

    hands.close()

    if len(features) == 0:
        print("Error: No features extracted. Something is wrong with the images.")
        return

    print(f"Extracted features from {len(features)} images.")

    X = np.array(features)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
    mlp.fit(X_train, y_train)

    with open(model_path, 'wb') as f:
        pickle.dump(mlp, f)
    with open(label_map_path, 'wb') as f:
        pickle.dump({v: k for k, v in label_map.items()}, f)

    print(f"SUCCESS! Model saved to: {model_path}")
    print("You can now run your Streamlit app.")

if __name__ == "__main__":
    train_model()
