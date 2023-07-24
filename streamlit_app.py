import cv2
import dlib
import time
from playsound import playsound
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading

# Load the pre-trained face detector and landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Audio file for alert
audio_file = "wake_up.wav"

# Variables for drowsiness detection
eyes_open = True
eyes_closed_start_time = None

# Streamlit App
st.set_page_config(page_title="Real-Time Drowsiness Detection", page_icon="😴")

st.sidebar.title('Drowsiness Detection Settings')
eye_aspect_ratio_threshold = st.sidebar.slider("Eye Aspect Ratio Threshold", 0.1, 0.5, 0.25, 0.01)
eyes_closed_threshold = st.sidebar.slider("Eyes Closed Threshold (Seconds)", 0.5, 5.0, 1.0, 0.1)

st.title('Real-Time Drowsiness Detection 😴')

# Lock to synchronize access to model variables
model_lock = threading.Lock()

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = detector(gray)

        with model_lock:
            global eyes_open, eyes_closed_start_time

            for face in faces:
                # Get facial landmarks for each detected face
                landmarks = predictor(gray, face)

                # Extract the eye region coordinates
                left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
                right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
                
                # Draw the eyes' contours on the frame
                for eye in [left_eye, right_eye]:
                    for i in range(len(eye) - 1):
                        cv2.line(img, eye[i], eye[i + 1], (0, 255, 0), 1)

                # Calculate the eye aspect ratio for each eye
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)

                # Calculate the average eye aspect ratio
                avg_ear = (left_ear + right_ear) / 2.0

                # Check if the eyes are closed
                if avg_ear < eye_aspect_ratio_threshold:
                    if eyes_open:
                        eyes_closed_start_time = time.time()
                        eyes_open = False

                    # Calculate the duration of closed eyes
                    eyes_closed_duration = time.time() - eyes_closed_start_time

                    # Display the number of seconds the eyes are closed
                    cv2.putText(img, f"Eyes Closed: {eyes_closed_duration:.1f} seconds", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Check if the eyes have been closed for the specified threshold duration
                    if eyes_closed_duration >= eyes_closed_threshold:
                        # Play the audio alert
                        playsound(audio_file)

                        # Display the text as WAKE UP!!!
                        cv2.putText(img, "WAKE UP!!!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # Reset the eyes_open flag to True, indicating that the eyes are open
                        # eyes_open = True
                else:
                    # Reset the eyes_open flag to True, indicating that the eyes are open
                    eyes_open = True

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def eye_aspect_ratio(eye):
    # Calculate the distance between the vertical eye landmarks
    A = euclidean_distance(eye[1], eye[5])
    B = euclidean_distance(eye[2], eye[4])

    # Calculate the distance between the horizontal eye landmarks
    C = euclidean_distance(eye[0], eye[3])

    # Calculate the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def euclidean_distance(point1, point2):
    # Calculate the Euclidean distance between two points
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
)
