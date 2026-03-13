import streamlit as st
import cv2
import numpy as np
import tempfile
from moviepy import VideoFileClip


def apply_quality_cc(frame):
    # 1. Kontrastı ve Keskinliği Artır (CLAHE)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 2. Renk Doygunluğunu (Saturation) Yükselt
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype("float32")
    hsv[:, :, 1] = hsv[:, :, 1] * 1.4  # Doygunluğu %40 artırır
    hsv[:, :, 2] = hsv[:, :, 2] * 1.1  # Parlaklığı %10 artırır
    hsv = np.clip(hsv, 0, 255).astype("uint8")
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3. Hafif Keskinleştirme (Sharpening)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    frame = cv2.filter2D(frame, -1, kernel)
    
    return frame

st.title("⚽ Football Edit CC Maker")
uploaded_file = st.file_uploader("Editlenecek videoyu seç...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    if st.button("CC Uygula ve Renderla"):
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        output_path = "cc_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        progress_bar = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = apply_quality_cc(frame)
            out.write(processed_frame)
            
            count += 1
            progress_bar.progress(count / frame_count)

        cap.release()
        out.release()
        
        st.success("İşlem tamamlandı!")
        with open(output_path, "rb") as file:
            st.download_button("Videoyu İndir", file, "kaliteli_edit.mp4")
