import cv2
import numpy as np
from mtcnn import MTCNN
import threading

# Load the MTCNN detector model
detector = MTCNN()

# Function to apply the funny eyes filter
def apply_funny_eyes(image, left_eye, right_eye):
    left_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
    right_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)

    radius = left_eye[3] // 2
    cv2.circle(image, left_center, radius, (255, 255, 255), -1)
    cv2.circle(image, right_center, radius, (255, 255, 255), -1)

    pupil_radius = radius // 2
    cv2.circle(image, left_center, pupil_radius, (0, 0, 0), -1)
    cv2.circle(image, right_center, pupil_radius, (0, 0, 0), -1)

    return image


# Function to process a single frame
def process_frame(frame):
    # Convert the frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the frame to a smaller size
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Detect faces in the small frame using the MTCNN detector
    faces = detector.detect_faces(small_frame)

    # Iterate over the detected faces and apply the funny eyes filter
    for face in faces:
        left_eye = face['keypoints']['left_eye']
        right_eye = face['keypoints']['right_eye']

        # Scale up the eye coordinates to the original size
        left_eye = [coord * 2 for coord in left_eye]
        right_eye = [coord * 2 for coord in right_eye]

        # Calculate the bounding boxes for the eyes
        eye_width = int(face['box'][2] * 0.3)
        eye_height = int(face['box'][3] * 0.3)
        left_eye_bbox = (left_eye[0] - eye_width // 2, left_eye[1] - eye_height // 2, eye_width, eye_height)
        right_eye_bbox = (right_eye[0] - eye_width // 2, right_eye[1] - eye_height // 2, eye_width, eye_height)

        # Apply the funny eyes filter to the original frame
        frame = apply_funny_eyes(frame, left_eye_bbox, right_eye_bbox)

    # Convert the frame back to BGR format for display
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame


cap = cv2.VideoCapture(0)

while True:
    # Capture a frame
    ret, frame = cap.read()


    # Check if the frame was successfully captured
    if not ret:
        print("Error capturing frame.")
        break

    # Process the frame (e.g., apply a filter or perform other operations)
    # In this example, we simply convert the frame to grayscale.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #print(gray_frame.shape)

    process_frame(frame)
    cv2.imshow("Processed Frame", process_frame)

    # Display the processed frame
    # cv2.imshow("Processed Frame", gray_frame)
    cv2.waitKey(20)

    # Wait for the user to press the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
