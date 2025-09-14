# ----------- Pattern recognition: object detection and tracking -----------------

'''

Ідентифікація очей обличчя на потоковому відео

алгоритм ідентифікації "каскади Хаара":
 - ідентифікація обличчя;
 - кідентифікація очей на обличчі.

Prateek Joshi Artificial Intelligence applications with Python, Part 13:
https://mazz.keybase.pub/ebooks/ai/9781786464392-ARTIFICIAL_INTELLIGENCE_WITH_PYTHON.pdf
Scripts:
https://github.com/PacktPublishing/Artificial-Intelligence-with-Python

Package            Version
------------------ -----------
opencv-python      3.4.18.65

'''

import cv2

# Load the Haar cascade files for face and eye
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Check if the face cascade file has been loaded correctly
if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')

# Check if the eye cascade file has been loaded correctly
if eye_cascade.empty():
	raise IOError('Unable to load the eye cascade classifier xml file')

# Initialize the video capture object from the webcam
# cap = cv2.VideoCapture(0)

# Initialize the video capture object
cap = cv2.VideoCapture('Video_1.mp4')

# Define the scaling factor
ds_factor = 0.5

# Iterate until the user hits the 'Esc' key
while True:
    # Capture the current frame
    _, frame = cap.read()

    # Resize the frame
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run the face detector on the grayscale image
    '''
     Ідентифікація облич на потоковому відео з каскадами Хаара     
    '''
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each face that's detected, run the eye detector
    for (x,y,w,h) in faces:
        # Extract the grayscale face ROI
        roi_gray = gray[y:y+h, x:x+w]

        # Extract the color face ROI
        roi_color = frame[y:y+h, x:x+w]

        '''
         Ідентифікація очей на обличчі на потоковому відео з каскадами Хаара     
        '''

        # Run the eye detector on the grayscale ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Draw circles around the eyes
        for (x_eye,y_eye,w_eye,h_eye) in eyes:
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            radius = int(0.3 * (w_eye + h_eye))
            color = (0, 255, 0)
            thickness = 3
            cv2.circle(roi_color, center, radius, color, thickness)

    # Display the output
    cv2.imshow('Eye Detector', frame)

    # Check if the user hit the 'Esc' key
    c = cv2.waitKey(1)
    if c == 27:
        break

# Release the video capture object
cap.release()

# Close all the windows
cv2.destroyAllWindows()
