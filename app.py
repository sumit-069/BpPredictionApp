import streamlit as st
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import tempfile

st.set_page_config(page_title="Finger PPG Detector", layout="wide")

st.title("🩺 Finger PPG - Heart Rate Detection")

uploaded_file = st.file_uploader("📤 Upload Finger Video", type=["mp4", "avi", "mov"])

# Bandpass filter
def bandpass_filter(signal, fs=30, low=0.7, high=4):
    nyquist = 0.5 * fs
    low /= nyquist
    high /= nyquist
    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Normalize signal
def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

# BPM calculation (improved)
def calculate_bpm(signal, fs=30):
    filtered = bandpass_filter(signal, fs)
    filtered = normalize(filtered)

    peaks, _ = find_peaks(filtered, distance=fs*0.5, height=0)

    if len(peaks) < 2:
        return 0, filtered

    peak_intervals = np.diff(peaks) / fs
    avg_interval = np.mean(peak_intervals)

    bpm = 60 / avg_interval

    return bpm, filtered

if uploaded_file is not None:

    st.subheader("🎥 Processing Finger Video...")

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    red_signal = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract RED channel
        avg_color = np.mean(frame, axis=(0,1))
        red_signal.append(avg_color[2])

        frame_count += 1
        if total_frames > 0:
            progress.progress(frame_count / total_frames)

    cap.release()

    st.success("✅ Processing Complete!")

    if len(red_signal) > 100:

        signal = np.array(red_signal)

        bpm, filtered = calculate_bpm(signal, fs=fps)

        if bpm == 0:
            st.warning("⚠️ Could not detect stable heart rate")
        else:
            st.success(f"❤️ Heart Rate: {int(bpm)} BPM")

        st.subheader("📈 PPG Signal")
        st.line_chart(filtered)

    else:
        st.error("❌ Not enough data. Upload longer video (20–30 sec)")