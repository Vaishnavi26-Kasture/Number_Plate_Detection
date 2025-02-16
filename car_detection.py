import cv2
import pytesseract
import os
import streamlit as st
from PIL import Image
import tempfile

# Streamlit UI
st.title("License Plate Detection and OCR")
st.sidebar.header("Upload and Settings")

# Input for cascade file path
cascade_file_path = st.sidebar.text_input(
    "Cascade File Path", "C:/Users/Vaishnavi/Documents/Naresh IT/OPENCV/OCR/haarcascade_license_plate_rus_16stages.xml"
)
tesseract_path = st.sidebar.text_input(
    "Tesseract Path", "C:/Program Files/Tesseract-OCR/tesseract.exe"
)
video_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

# Check if paths are valid
if not os.path.exists(cascade_file_path):
    st.error("Error: Cascade file not found. Please provide a valid path.")
    st.stop()

if not os.path.exists(tesseract_path):
    st.error("Error: Tesseract executable not found. Please provide a valid path.")
    st.stop()

if video_file is None:
    st.info("Upload a video file to start processing.")
    st.stop()

# Save uploaded video to a temporary file
temp_video = tempfile.NamedTemporaryFile(delete=False)
temp_video.write(video_file.read())
video_path = temp_video.name

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Load the cascade classifier
plate_cascade = cv2.CascadeClassifier(cascade_file_path)

# Open video
video_capture = cv2.VideoCapture(video_path)
frame_window = st.image([])  # Placeholder for frames
detected_text = st.empty()  # Placeholder for detected text

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        st.warning("End of video or error in reading frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cropped_plate = gray[y:y + h, x:x + w]
        text = pytesseract.image_to_string(cropped_plate, config="--psm 7").strip()

        if text:
            detected_text.write(f"Detected License Plate: {text}")
        else:
            detected_text.write("License plate detected but could not recognize text.")
        
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)

video_capture.release()
os.remove(video_path)
cv2.destroyAllWindows()
