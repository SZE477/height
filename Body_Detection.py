import cv2 as cv
import mediapipe as mp
import time
import math
import warnings
import os

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set TensorFlow logging level to ERROR to suppress INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Initializing MediaPipe Pose module
mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils

# Initializing video capture
capture = cv.VideoCapture(0)

# Scaling factor for height calculation
scaling_factor = 1.9  # Adjust as needed

while True:
    isTrue, img = capture.read()
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Processing the image to detect pose landmarks
    result = pose.process(img_rgb)
    
    height = None  # Initialize height variable
    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)

        h, w, c = img.shape
        landmarks = result.pose_landmarks.landmark

        # Assuming points 31 (right shoulder) and 32 (left shoulder) for height calculation
        shoulder_left = landmarks[32]
        shoulder_right = landmarks[31]

        cx1, cy1 = int(shoulder_left.x * w), int(shoulder_left.y * h)
        cx2, cy2 = int(shoulder_right.x * w), int(shoulder_right.y * h)

        # Calculate the distance between the two points
        d = math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
        height = round(d * scaling_factor)  # Scale the height

        # Draw circles at shoulder positions
        cv.circle(img, (cx1, cy1), 15, (0, 0, 255), cv.FILLED)
        cv.circle(img, (cx2, cy2), 15, (0, 0, 255), cv.FILLED)

    # Resize image for display
    img = cv.resize(img, (700, 500))

    # Display the height if calculated
    if height is not None:
        height_color = (0, 255, 0)  # Change the color to green
        cv.putText(img, "Height: ", (40, 70), cv.FONT_HERSHEY_COMPLEX, 1, height_color, thickness=2)
        cv.putText(img, str(height), (180, 70), cv.FONT_HERSHEY_DUPLEX, 1, height_color, thickness=2)
        cv.putText(img, "cms", (240, 70), cv.FONT_HERSHEY_PLAIN, 2, height_color, thickness=2)
    
    cv.putText(img, "Stand at least 3 meters away", (40, 450), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    # Display the frame
    cv.imshow("Task", img)
    
    # Break the loop if 'q' is pressed
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

# Release resources
capture.release()
cv.destroyAllWindows()
