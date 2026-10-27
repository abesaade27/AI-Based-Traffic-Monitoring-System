import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
from datetime import datetime

from fines import calculate_fine
from plate_ocr import extract_plate_text

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Pakistan E-Challan System",
    layout="wide"
)

MODEL_PATH = os.path.join("final_model", "best.pt")
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align:center;'>E-Challan</h1>
    <h4 style='text-align:center;'>Electronic Traffic Violation Management System</h4>
    <hr>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload Traffic Violation Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # Detection
    results = model(img_np, conf=0.35)
    annotated = results[0].plot()

    detected_classes = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        detected_classes.append(CLASS_NAMES[cls_id])

    # OCR
    plate_text = extract_plate_text(img_np, results)
    plate_text = plate_text if plate_text else "Not Detected"

    # Fines
    violations, total_fine = calculate_fine(detected_classes)

    # Date & Time
    now = datetime.now().strftime("%d-%m-%Y  %H:%M")

    # ---------------- E-CHALLAN LAYOUT ----------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Violation / Vehicle Details")

        st.markdown(f"""
        **Date & Time of Offence:** {now}  
        **Violation Type:** {", ".join(violations) if violations else "No Violation"}  
        **Number Plate:** {plate_text}  
        **City:** Lahore  
        **Province:** Punjab  
        """)

    with col2:
        st.subheader("Penalty Details")
        st.markdown(
            f"""
            <div style="border:2px solid black; padding:15px; font-size:18px;">
            <b>Total Fine:</b><br>
            Rs. {total_fine}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ---------------- IMAGE EVIDENCE ----------------
    st.subheader("Violation Evidence")
    st.image(
        annotated,
        caption="Captured Traffic Violation Image",
        use_container_width=True
    )

    # ---------------- FOOTER ----------------
    st.markdown(
        """
        <hr>
        <center>
        Punjab Safe Cities Authority – Demo AI System<br>
        This E-Challan is system generated for academic purposes only.
        </center>
        """,
        unsafe_allow_html=True
    )
