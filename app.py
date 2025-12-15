import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Face Detection Streamlit App")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file
    image = Image.open(uploaded_file)
    # Convert PIL Image to OpenCV format (numpy array)
    img_array = np.array(image.convert('RGB'))
    img_opencv = img_array[:, :, ::-1].copy() # Convert RGB to BGR

    # Convert to grayscale for detection
    gray = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained face cascade classifier
    # You would need the 'haarcascade_frontalface_default.xml' file in your project directory
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img_opencv, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Convert back to RGB for Streamlit display
    final_img = img_opencv[:, :, ::-1]

    st.image(final_img, caption=f"Detected {len(faces)} face(s)", use_column_width=True)
