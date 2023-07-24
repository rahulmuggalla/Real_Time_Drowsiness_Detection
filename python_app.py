import cv2
import dlib
import time
from playsound import playsound

# Load the pre-trained face detector and landmark predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

audio_file = "wake_up.wav"

last_alert_time = time.time()

eyes_open = True
eyes_closed_start_time = None

eyes_closed_threshold = 1  # in seconds

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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Get facial landmarks for each detected face
        landmarks = predictor(gray, face)

        # Extract the eye region coordinates
        left_eye = []
        right_eye = []
        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x, y))

        # Calculate the eye aspect ratio for each eye
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Calculate the average eye aspect ratio
        avg_ear = (left_ear + right_ear) / 2.0
                
    # Check if the eyes are closed
        if avg_ear < 0.25:
            if eyes_open:
                eyes_closed_start_time = time.time()
                eyes_open = False
                
            # Calculate the duration of closed eyes
            eyes_closed_duration = time.time() - eyes_closed_start_time
            
            # Display the number of seconds the eyes are closed
            cv2.putText(frame, f"Eyes Closed: {eyes_closed_duration:.1f} seconds", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Check if the eyes have been closed for the specified threshold duration
            if eyes_closed_duration >= eyes_closed_threshold:
                # Play the audio alert
                playsound(audio_file)
                
                # Display the text as WAKE UP!!!
                cv2.putText(frame, "WAKE UP!!!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                
                # Reset the eyes_open flag to True, indicating that the eyes are open
                #eyes_open = True
        else:
            # Reset the eyes_open flag to True, indicating that the eyes are open
            eyes_open = True
        
        '''# Draw the eyes' contours on the frame
        for eye in [left_eye, right_eye]:
            for i in range(len(eye) - 1):
                cv2.line(frame, eye[i], eye[i + 1], (0, 255, 0), 1)'''
    
    cv2.imshow("Real Time Drowsiness Detection", frame)

    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()