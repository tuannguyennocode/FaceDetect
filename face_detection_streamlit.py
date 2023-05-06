import streamlit as st
import cv2

st.title("Face Detection")

# load OpenCV's Haar cascade for face detection from disk
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# initialize the video stream
"[INFO] starting video stream..."


@st.cache(allow_output_mutation=True)
def get_cap():
    return cv2.VideoCapture(0)

cap = get_cap()

frameST = st.empty()

# loop over the frames from the video stream
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    orig = frame.copy()

    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30))

    # loop over the face detections and draw them on the frame
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the output frame
    frameST.image(frame, channels="BGR")
    
        
    
        
