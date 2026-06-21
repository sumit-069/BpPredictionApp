import streamlit as st
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import tempfile
import pickle

# Page config
st.set_page_config(page_title="BP Detector", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
.metric {
    font-size: 28px;
    font-weight: bold;
    color: #00FFAA;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🩺 BP Detection from Video</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a face video to estimate Heart Rate & Blood Pressure</p>", unsafe_allow_html=True)

# Load model
model = pickle.load(open("bpmodel.pkl", "rb"))

# Upload
uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4", "avi", "mov"])

# Haar cascade (face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Filters
def bandpass_filter(signal, fs=30, low=0.7, high=4):
    nyquist = 0.5 * fs
    low /= nyquist
    high /= nyquist
    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, signal)

def calculate_bpm(signal, fs=30):
    filtered = bandpass_filter(signal, fs)
    peaks, _ = find_peaks(filtered, distance=fs/2)
    bpm = len(peaks) * (60 / (len(signal) / fs))
    return bpm, filtered

def extract_features(signal):
    return [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.ptp(signal)
    ]

# Main logic
if uploaded_file is not None:

    st.subheader("🎥 Processing Video...")
    progress = st.progress(0)

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    green_signal = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stframe = st.empty()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = frame[y:y+int(h*0.3), x:x+w]

            if roi.size != 0:
                avg_color = np.mean(roi, axis=(0,1))
                green_signal.append(avg_color[1])

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        stframe.image(frame, channels="BGR")

        frame_count += 1
        if total_frames > 0:
            progress.progress(frame_count / total_frames)

    cap.release()

    st.success("✅ Processing Complete!")

    if len(green_signal) > 100:

        bpm, filtered = calculate_bpm(np.array(green_signal), fs=fps)

        features = np.array(extract_features(filtered)).reshape(1, -1)
        prediction = model.predict(features)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="card">
                <h3>❤️ Heart Rate</h3>
                <div class="metric">{int(bpm)} BPM</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            try:
                systolic, diastolic = prediction[0]
                bp_text = f"{int(systolic)}/{int(diastolic)} mmHg"
            except:
                bp_text = str(prediction)

            st.markdown(f"""
            <div class="card">
                <h3>🩸 Blood Pressure</h3>
                <div class="metric">{bp_text}</div>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("📈 PPG Signal")
        st.line_chart(filtered)

    else:
        st.error("❌ Not enough data. Upload a longer, stable video.")