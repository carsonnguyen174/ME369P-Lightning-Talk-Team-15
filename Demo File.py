# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:21:01 2024

@author: Carson
"""

# import argparse
 
# import numpy as np
# import cv2 as cv

# #NOTE: Some people may not have the file for the haarcascade_frontalface_default installed. 
# # These models are included when you download the OpenCV library but need to be accessed
# # with the following line ( this goes to the directory of OpenCV where the models are located )
# face_model = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Capture video from webcam
# cam = cv.VideoCapture(0)

# try:
#     assert cam.isOpened() == True
# except:
#     print("Error: could not open video")



# Code from CHATGPT:
# Note this code does not have an exit button, needs one lol
import cv2
# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video from the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Loop to continuously get frames from the camera
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert frame to grayscale as face detection requires grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame with rectangles around faces
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

    
    
