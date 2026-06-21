# 🩺 BP Detection from Video using rPPG

This project is an AI-powered web application that estimates **Heart Rate (BPM)** and **Blood Pressure (BP)** from a facial video using **remote Photoplethysmography (rPPG)**.

Built using **Streamlit, OpenCV, MediaPipe, and Machine Learning**, this project demonstrates how subtle skin color changes can be used to extract physiological signals.

---

## 🚀 Features

* 📤 Upload video instead of live camera
* 👤 Automatic face detection using MediaPipe
* 🧠 Extract rPPG signal from forehead region
* ❤️ Heart Rate (BPM) estimation
* 🩸 Blood Pressure prediction using trained ML model (`bpmodel.pkl`)
* 📈 Real-time PPG waveform visualization
* 🎨 Clean and interactive Streamlit UI

---

## 🧠 How It Works

1. Video is uploaded by the user
2. Face is detected frame-by-frame
3. Forehead region is extracted (ROI)
4. Green channel signal is captured over time
5. Signal is filtered using bandpass filter
6. Peaks are detected → Heart Rate is calculated
7. Features are extracted from signal
8. ML model predicts Blood Pressure

---

## 🛠️ Tech Stack

* **Frontend/UI**: Streamlit
* **Computer Vision**: OpenCV, MediaPipe
* **Signal Processing**: SciPy
* **Machine Learning**: Scikit-learn / XGBoost
* **Language**: Python

---

## 📂 Project Structure

```
├── app.py               # Main Streamlit app
├── bpmodel.pkl         # Trained BP prediction model
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/bp-detection-app.git
cd bp-detection-app
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📌 Usage Instructions

* Upload a **clear face video (10–30 seconds)**
* Ensure:

  * Good lighting 💡
  * Minimal motion 🚫
  * Face clearly visible 👤
* Wait for processing
* View:

  * ❤️ Heart Rate
  * 🩸 Blood Pressure
  * 📈 PPG Signal

---

## ⚠️ Limitations

* Not medically accurate (research-level system)
* Sensitive to:

  * Lighting conditions
  * Motion
  * Video quality
* BP prediction depends on model quality and training data

---

## 🔥 Future Improvements

* Use advanced rPPG algorithms (POS, CHROM)
* Deep Learning models (LSTM/CNN)
* Multi-region signal extraction (forehead + cheeks)
* Real-time deployment (web/mobile)
* Better accuracy with large datasets (e.g., MIMIC)

---

## 🙌 Acknowledgements

* MediaPipe for face detection
* OpenCV for video processing
* SciPy for signal filtering

---

## 📜 License

This project is for educational and research purposes only.

---

## 👨‍💻 Author

**Sumit Kumar Bajpai**
GitHub: https://github.com/sumit-069

---

⭐ If you like this project, don’t forget to star the repository!
