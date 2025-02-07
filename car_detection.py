import cv2
import pytesseract
import os
import streamlit as st
from PIL import Image

# Streamlit interface
st.title("License Plate Detection and OCR")
st.sidebar.header("Upload and Settings")

# Input for cascade file path
cascade_file_path = st.sidebar.text_input(
    "Cascade File Path", r"C:\Users\Vaishnavi\Documents\Naresh IT\OPENCV\OCR\haarcascade_license_plate_rus_16stages.xml"
)
tesseract_path = st.sidebar.text_input(
    "Tesseract Path", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
video_path = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

# Check if paths are valid
if not os.path.exists(cascade_file_path):
    st.error("Error: Cascade file not found. Please provide a valid path.")
    st.stop()

if not os.path.exists(tesseract_path):
    st.error("Error: Tesseract executable not found. Please provide a valid path.")
    st.stop()

if video_path is None:
    st.info("Upload a video file to start processing.")
    st.stop()

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Load the cascade classifier
no_cascade = cv2.CascadeClassifier(cascade_file_path)

# Streamlit video display
video_capture = cv2.VideoCapture(video_path.name)

frame_window = st.image([])  # Placeholder for frames
detected_text = st.empty()  # Placeholder for detected text

while video_capture.isOpened():
    # Read frames from the video
    ret, frame = video_capture.read()
    if not ret:
        st.info("End of video or error in reading frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect license plates
    plates = no_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Process each detected license plate
    for (x, y, w, h) in plates:
        # Draw a rectangle around the license plate
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Crop the license plate region
        cropped_plate = gray[y:y + h, x:x + w]

        # Perform OCR on the cropped plate
        text = pytesseract.image_to_string(cropped_plate, config="--psm 7")  # Single line OCR
        detected_text.write(f"Detected License Plate Text: {text.strip()}")

        # Display the detected text on the frame
        cv2.putText(frame, text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convert frame to RGB for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Update the Streamlit frame display
    frame_window.image(frame)

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()
