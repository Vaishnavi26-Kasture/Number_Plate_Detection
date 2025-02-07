import cv2
import pytesseract
import os
import streamlit as st

# Streamlit app setup
st.title("License Plate Detection and OCR")
st.text("Upload a video file for license plate detection.")

# Upload video file
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    video_path = os.path.join("temp_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    # Check if cascade classifier file exists
    cascade_file_path = r"C:\Users\Vaishnavi\Documents\Naresh IT\OPENCV\OCR\haarcascade_license_plate_rus_16stages.xml"
    if not os.path.exists(cascade_file_path):
        st.error("Error: Cascade classifier file not found!")
        st.stop()

    # Load the cascade classifier
    no_cascade = cv2.CascadeClassifier(cascade_file_path)

    # Set the Tesseract executable path
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        st.error("Error: Tesseract executable not found at the specified path!")
        st.stop()

    # Initialize video capture
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        st.error("Error: Unable to open the video file!")
        st.stop()

    stframe = st.empty()  # Placeholder for video frames
    plate_list = []  # List to store detected plate numbers

    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.write("End of video or error in reading frame.")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect license plates
        plates = no_cascade.detectMultiScale(gray, 1.1, 5)  # Adjusted minNeighbors for better detection

        if len(plates) == 0:
            st.sidebar.write("No license plates detected in this frame.")

        # Draw rectangles around detected plates and perform OCR
        for (x, y, w, h) in plates:
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # Crop the number plate region
            cropped_plate = gray[y:y + h, x:x + w]

            # Perform OCR on the cropped plate
            text = pytesseract.image_to_string(cropped_plate, config='--psm 7').strip()

            if text and text not in plate_list:
                plate_list.append(text)

            # Display detected text on the frame
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update the list of detected plates in the sidebar
        st.sidebar.title("Detected License Plates")
        st.sidebar.write(plate_list)

        # Convert the frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        # Add a delay for visualization
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release the video capture
    video_capture.release()

    st.write("Video processing completed.")
