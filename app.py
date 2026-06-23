import streamlit as st
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import tempfile
import joblib
import os

# ---------------- UI ----------------
st.set_page_config(page_title="Finger PPG + BP Detector", layout="wide")
st.title("🩺 Finger PPG - Heart Rate & BP Detection")

uploaded_file = st.file_uploader("📤 Upload Finger Video", type=["mp4", "avi", "mov"])

# ---------------- Load Model ----------------
model = None
if os.path.exists("bpmodel.pkl"):
    try:
        model = joblib.load("bpmodel.pkl")
    except:
        st.warning("⚠️ Model could not be loaded")

# ---------------- Signal Processing ----------------
def bandpass_filter(signal, fs=30, low=0.7, high=4):
    nyquist = 0.5 * fs
    low /= nyquist
    high /= nyquist
    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, signal)

def normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

def calculate_bpm(signal, fs=30):
    filtered = bandpass_filter(signal, fs)
    filtered = normalize(filtered)

    peaks, _ = find_peaks(filtered, distance=fs*0.5)

    if len(peaks) < 2:
        return 0, filtered

    peak_intervals = np.diff(peaks) / fs
    bpm = 60 / np.mean(peak_intervals)

    return bpm, filtered

# ---------------- Feature Extraction (MATCHED) ----------------
def extract_features(signal, fs):

    signal = bandpass_filter(signal, fs)

    peaks, _ = find_peaks(signal, distance=fs//2)

    if len(peaks) < 2:
        return None

    peak_intervals = np.diff(peaks)

    heart_rate = 60 / (np.mean(peak_intervals) / fs)
    pulse_width = np.mean(peak_intervals) / fs

    rise_times = []
    for p in peaks:
        start = max(0, p - int(0.2 * fs))
        rise_times.append((p - start) / fs)
    rise_time = np.mean(rise_times)

    amplitude = np.max(signal) - np.min(signal)
    area = np.trapz(signal)

    slopes = []
    for p in peaks:
        start = max(0, p - int(0.2 * fs))
        slope = (signal[p] - signal[start]) / (p - start + 1e-6)
        slopes.append(slope)

    systolic_slope = np.mean(slopes)

    d_slopes = []
    for p in peaks:
        end = min(len(signal)-1, p + int(0.2 * fs))
        slope = (signal[end] - signal[p]) / (end - p + 1e-6)
        d_slopes.append(slope)

    diastolic_slope = np.mean(d_slopes)

    vpg = np.gradient(signal)
    apg = np.gradient(vpg)

    apg_peaks, _ = find_peaks(apg)

    if len(apg_peaks) >= 4:
        a, b, c, d = apg[apg_peaks[:4]]
        b_a = b / (a + 1e-6)
        c_a = c / (a + 1e-6)
        d_a = d / (a + 1e-6)
    else:
        b_a, c_a, d_a = 0, 0, 0

    return [
        heart_rate,
        pulse_width,
        rise_time,
        amplitude,
        area,
        systolic_slope,
        diastolic_slope,
        b_a,
        c_a,
        d_a
    ]

# ---------------- Main Processing ----------------
if uploaded_file is not None:

    st.subheader("🎥 Processing Finger Video...")
    progress = st.progress(0)

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    red_signal = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        avg_color = np.mean(frame, axis=(0,1))
        red_signal.append(avg_color[2])  # RED channel

        frame_count += 1
        if total_frames > 0:
            progress.progress(frame_count / total_frames)

    cap.release()

    st.success("✅ Processing Complete!")

    if len(red_signal) > 100:

        signal = np.array(red_signal)

        # ❤️ BPM
        bpm, filtered = calculate_bpm(signal, fs=fps)

        if bpm == 0:
            st.warning("⚠️ Could not detect stable heart rate")
        else:
            st.success(f"❤️ Heart Rate: {int(bpm)} BPM")

        # 📈 Signal
        st.subheader("📈 PPG Signal")
        st.line_chart(filtered)

        # 🩸 BP Prediction
        if model is not None:

            features = extract_features(filtered, fps)

            if features is None:
                st.error("❌ Not enough peaks for BP prediction")
            else:
                features = np.array(features).reshape(1, -1)

                try:
                    prediction = model.predict(features)

                    systolic, diastolic = prediction[0]

                    st.success(f"🩸 BP: {int(systolic)}/{int(diastolic)} mmHg")

                except Exception as e:
                    st.error(f"❌ BP prediction failed: {e}")

        else:
            st.info("ℹ️ BP model not available")

    else:
        st.error("❌ Not enough data. Upload longer video (20–30 sec)")