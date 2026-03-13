import streamlit as st
import cv2
import numpy as np
import tempfile
from moviepy import VideoFileClip

def apply_pro_football_cc(frame):
    # 1. Renkleri Canlandır (Saturasyon & Parlaklık)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype("float32")
    hsv[:, :, 1] *= 1.35  # Renk doygunluğu %35 artış (İdeal seviye)
    hsv[:, :, 2] *= 1.05  # Hafif parlaklık
    hsv = np.clip(hsv, 0, 255).astype("uint8")
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 2. Detayları Çıkar (Daha yumuşak bir CLAHE)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8)) # Limit 3.0'dan 1.8'e düştü (Çamurlaşmayı önler)
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 3. Profesyonel Kontrast (Gamma Correction)
    # Bu adım videonun o "premium" koyu tonlarını sağlar
    invGamma = 1.0 / 1.1
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    frame = cv2.LUT(frame, table)
    
    return frame

st.set_page_config(page_title="EAGLE22 CC Maker", page_icon="⚽")
st.title("⚽ Premium Football CC")

uploaded_file = st.file_uploader("Videonu Yükle", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    if st.button("CC Bas ve Render Al"):
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        output_path = "vibrant_edit.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        bar = st.progress(0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for i in range(frames):
            ret, frame = cap.read()
            if not ret: break
            
            # CC Uygula
            processed = apply_pro_football_cc(frame)
            out.write(processed)
            bar.progress((i + 1) / frames)

        cap.release()
        out.release()
        
        st.success("CC Hazır! Aşağıdan indir.")
        with open(output_path, "rb") as f:
            st.download_button("⚡ Videoyu İndir", f, "eagle22_edit.mp4")
