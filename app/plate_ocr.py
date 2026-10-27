import cv2
import pytesseract
from ocr_utils import preprocess_plate
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_plate_text(image, yolo_results):
    plates = []

    for box in yolo_results[0].boxes:
        cls_id = int(box.cls[0])
        if yolo_results[0].names[cls_id] == "license_plate":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = image[y1:y2, x1:x2]

            processed = preprocess_plate(plate_img)
            text = pytesseract.image_to_string(
                processed,
                config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            text = "".join(c for c in text if c.isalnum())
            if len(text) >= 4:
                plates.append(text)

    return plates[0] if plates else None
