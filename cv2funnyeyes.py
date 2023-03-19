import cv2
# import dlib
import numpy as np


from mtcnn import MTCNN

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


# def image_with_funny_eyes(img):
#     # Load the pre-trained facial landmark detector model (download the model from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
#     predictor_path = "shape_predictor_68_face_landmarks.dat"
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor(predictor_path)

#     # Load an image containing a face
#     image_path = "input.jpg"
#     image = cv2.imread(image_path)

#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the image
#     faces = detector(gray)

#     for face in faces:
#         # Get the facial landmarks
#         landmarks = predictor(gray, face)

#         # Convert the landmarks to a NumPy array
#         landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])

#         # Apply the funny eyes filter
#         image = apply_funny_eyes(image, landmarks)

#     # Save the output image
#     cv2.imwrite("output.jpg", image)

#     # Show the output image
#     cv2.imshow("Funny Eyes Filter", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# Open the default camera (usually the built-in webcam). You can change the index (0) to use other cameras.
cap = cv2.VideoCapture(0)

# while True:
#     # Capture a frame
#     ret, frame = cap.read()

#     # Check if the frame was successfully captured
#     if not ret:
#         print("Error capturing frame.")
#         break

#     # Process the frame (e.g., apply a filter or perform other operations)
#     # In this example, we simply convert the frame to grayscale.
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Display the processed frame
#     cv2.imshow("Processed Frame", gray_frame)

#     # Wait for the user to press the 'q' key to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


while True:
    # Capture a frame
    ret, frame = cap.read()

    def image_with_funny_eyes(image):
        # Load the pre-trained facial landmark detector model (download the model from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
        detector = MTCNN()

        # Load an image containing a face
        # image_path = "input.jpg"
        # image = cv2.imread(image_path)

        # Convert the image to grayscale
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = detector.detect_faces(image)

        for face in faces:
            # Get the coordinates of the left and right eyes
            left_eye = face['keypoints']['left_eye']
            right_eye = face['keypoints']['right_eye']

            # Calculate the bounding boxes for the eyes
            eye_width = int(face['box'][2] * 0.3)
            eye_height = int(face['box'][3] * 0.3)
            left_eye_bbox = (left_eye[0] - eye_width // 2, left_eye[1] - eye_height // 2, eye_width, eye_height)
            right_eye_bbox = (right_eye[0] - eye_width // 2, right_eye[1] - eye_height // 2, eye_width, eye_height)

            # Apply the funny eyes filter
            image = apply_funny_eyes(image, left_eye_bbox, right_eye_bbox)

        # Save the output image
        # cv2.imwrite("output.jpg", image)

        # Show the output image
        cv2.imshow("Funny Eyes Filter", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    #############

    # Check if the frame was successfully captured
    if not ret:
        print("Error capturing frame.")
        break

    # Process the frame (e.g., apply a filter or perform other operations)
    # In this example, we simply convert the frame to grayscale.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(gray_frame.shape)

    image_with_funny_eyes(gray_frame)

    # Display the processed frame
    # cv2.imshow("Processed Frame", gray_frame)
    cv2.waitKey(20)

    # Wait for the user to press the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
