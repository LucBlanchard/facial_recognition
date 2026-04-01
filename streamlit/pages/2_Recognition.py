import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import tempfile
import requests
from io import BytesIO
import os


# Get API endpoint from environment variable or use default
api_endpoint=os.getenv("API_URL","http://localhost:5000")
API_URL=api_endpoint.rstrip('/') + '/recognize'

st.set_page_config(page_title="Live Video Feed", layout="centered")
st.title("Live Video Feed")

# ── Source selector ──────────────────────────────────────────────
source = st.radio("Source", ["Webcam", "Video File"], horizontal=True)

# Fix #1: "yolo26n.pt" does not exist — correct name is "yolov8n.pt"
# Fix #2: model is unused anyway since recognition is handled by the Flask API
# Removed YOLO import and model loading entirely

video_path = None
if source == "Video File":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        tmp.close()
        video_path = tmp.name

# ── Controls ───────────────────────────────────────────────────────
col1, col2 = st.columns(2)
start = col1.button("▶  Start", use_container_width=True)
stop  = col2.button("⏹  Stop",  use_container_width=True)

if stop:
    st.session_state["running"] = False
if start:
    st.session_state["running"] = True

# ── Display placeholder ────────────────────────────────────────────
frame_window = st.empty()
status       = st.empty()

# Fix #3: Haar cascade and detect() function are unused since
# face detection is handled server-side — removed entirely

# ── Capture loop ───────────────────────────────────────────────────
if st.session_state.get("running"):

    cap_source = 0 if source == "Webcam" else video_path

    if cap_source is None:
        st.warning("Upload a video file first.")
        st.stop()

    cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        st.error("Cannot open source. Check your webcam or file.")
        st.stop()

    while st.session_state.get("running"):
        ret, frame = cap.read()

        if not ret:
            status.info("End of video / no frame received.")
            break

        # Fix #4: can't open a numpy array with open() — encode frame as JPEG bytes first
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_bytes = buffer.tobytes()

        try:
            response = requests.post(
                API_URL,
                files={"image": ("frame.jpg", jpg_bytes, "image/jpeg")}
            )
            print(f"Status code: {response.status_code}")

            if response.content:
                image = Image.open(BytesIO(response.content))
                # Fix #5: "stretch" is not a valid width value — use use_container_width instead
                frame_window.image(image, channels="RGB", use_container_width=True)
            else:
                print("Server returned an empty response — is the Flask server running?")

        # Fix #6: network errors not handled — would crash the whole app
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the Flask server at localhost:5000.")
            break

    cap.release()
    status.success("Stream stopped.")