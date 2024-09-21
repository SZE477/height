# Required Libraries
import cv2
import os

# Constants for distance and face size (in centimeters)
Known_distance = 60.96
Known_width = 14.3

# Colors for drawing on frames
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)

# Font for displaying text
fonts = cv2.FONT_HERSHEY_COMPLEX

# Face detector using pre-trained model
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to calculate the focal length
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

# Function to calculate distance based on focal length
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length) / face_width_in_frame
    return distance

# Function to detect face and return its width in pixels
def face_data(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)
        return w  # Return the face width in pixels
    return 0  # Return 0 if no face is detected

# Load reference image to calculate focal length
ref_image = cv2.imread("Ref_image.jpg")
ref_image_face_width = face_data(ref_image)

# Calculate focal length based on the reference image
Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, ref_image_face_width)
print(f"Focal Length: {Focal_length_found}")

# Show the reference image
cv2.imshow("Reference Image", ref_image)

# Initialize webcam feed
cap = cv2.VideoCapture(0)

# Main loop for processing video frames
while True:
    _, frame = cap.read()  # Capture frame from webcam
    face_width_in_frame = face_data(frame)  # Detect face in the current frame

    if face_width_in_frame != 0:
        Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)
        Distance = round(Distance)

        # Display appropriate messages based on distance
        if 330 <= Distance <= 360:
            os.startfile(r"C:\Users\Easwar\Desktop\Height-Detection-main\Body_Detection.py")
            break
        elif Distance < 330:
            print("Step back")
        else:
            print("Come a little closer")

        # Draw distance information on the frame
        cv2.line(frame, (30, 30), (230, 30), RED, 32)
        cv2.line(frame, (30, 30), (230, 30), BLACK, 28)
        cv2.putText(frame, f"Distance: {Distance} cm", (30, 35), fonts, 0.6, GREEN, 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
