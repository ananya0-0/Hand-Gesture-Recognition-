# 🖐️ Real-Time ASL Hand Gesture Recognition

An interactive, low-latency Computer Vision and Machine Learning application that recognizes American Sign Language (ASL) alphabet gestures in real time using a local webcam. 

This project was originally developed in Google Colab and has been optimized to run locally via VS Code using **OpenCV**, **MediaPipe**, and an **MLP (Multi-Layer Perceptron) Classifier**.

---

## 🚀 Features
* **Real-Time Detection:** Smooth, native webcam integration replacing laggy cloud-based WebRTC streaming.
* **Efficient Feature Extraction:** Tracks 21 hand landmarks using MediaPipe and converts them into normalized coordinate vectors relative to the wrist.
* **Lightweight ML Architecture:** Scikit-Learn's MLPClassifier delivers fast inference speeds on standard consumer CPUs without requiring heavy deep-learning dependencies.
* **Clean UI:** Interactive Streamlit dashboard to control webcam states easily.

---

## 🛠️ Tech Stack & Dependencies
* **Core Language:** Python 3.10+
* **Framework:** Streamlit (Web UI)
* **Computer Vision:** OpenCV-Python, MediaPipe
* **Machine Learning:** Scikit-Learn, NumPy
* **Serialization:** Pickle
