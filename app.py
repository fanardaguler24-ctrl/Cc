import streamlit as st
import cv2
import numpy as np
import tempfile
from moviepy import VideoFileClip

def apply_ae_like_cc(frame):
    # 1. Profesyonel Kontrast (CLAHE) - Daha Hafif ve Doğal
    # Beyaz haleleri engellemek için clipLimit'i düşürdük
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8,8)) # 1.8'den 1.6'ya düştü
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 2. Tok Doygunluk ve Derin Siyahlar (Saturasyon & Gamma Correction)
    # Bu adım,ae cc'deki o zengin renk hissini verir
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype("float32")
    
    # Saturasyonu (Doygunluğu) tok hale getirir
    hsv[:, :, 1] *= 1.45  # Tok doygunluk
    # Siyahları derinleştirir (AE'deki "Levels" veya "Curves" gibi)
    hsv[:, :, 2] *= 0.92  # Derin siyahlar
    
    hsv = np.clip(hsv, 0, 255).astype("uint8")
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3. Profesyonel Keskinleştirme (Unsharp Mask)
    # Gürültüyü (noise) tetikleyen "hard sharpening" yerine
    # daha kaliteli "Unsharp Mask" kullanıyoruz.
    gaussian_blur = cv2.GaussianBlur(frame, (7,7), 1.5)
    frame = cv2.addWeighted(frame, 1.8, gaussian_blur, -0.8, 0)
    
    return frame

st.set_page_config(page_title="Pro-Edit CC Maker", page_icon="⚽")
st.title("⚽ AE CC Stilinde Vibrant Dark")

uploaded_file = st.file_uploader("Editlenecek Videoyu Seç", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    if st.button("AE CC Bas ve Renderle"):
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        output_path = "vibrant_dark_edit.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        bar = st.progress(0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for i in range(frames):
            ret, frame = cap.read()
            if not ret: break
            
            # AE-Like CC Uygula
            processed = apply_ae_like_cc(frame)
            out.write(processed)
            bar.progress((i + 1) / frames)

        cap.release()
        out.release()
        
        st.success("AE CC Hazır! İndirebilirsin.")
        with open(output_path, "rb") as f:
            st.download_button("⚡ Videoyu İndir", f, "ae_cc_edit.mp4")
