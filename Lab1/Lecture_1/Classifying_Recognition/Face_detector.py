# ----------- Pattern recognition: object detection and tracking -----------------

'''

Ідентифікація облич на потоковому відео

алгоритм ідентифікації "каскади Хаара":
 - класифікатор Хаара - ієрархічна множина простих вирішувачів - як сума та різниця підобластей зображення;
 - надмірність даних - потокове відео, класифікатор Хаара - вважається вже сформованим;
 - каскади Хаара складаються з множини "цифрових автоматів" що побудовані в ієрархію структур "дерево";
 - класифікатор на кінцевому прошарку  має образ (образи) із часткових ознак;
 - зображення описується множиною частковимих ознак;
 - процес ідентифікації: порівняння множини ознак на прошарках для віднесення обєкту до найближчого образа.

Prateek Joshi Artificial Intelligence applications with Python, Part 13:
https://mazz.keybase.pub/ebooks/ai/9781786464392-ARTIFICIAL_INTELLIGENCE_WITH_PYTHON.pdf
Scripts:
https://github.com/PacktPublishing/Artificial-Intelligence-with-Python

Package            Version
------------------ -----------
opencv-python      3.4.18.65


'''

import cv2

# Load the Haar cascade file
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Check if the cascade file has been loaded correctly
if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')

# Initialize the video capture object from the webcam
# cap = cv2.VideoCapture(0)

# Initialize the video capture object
cap = cv2.VideoCapture('Video_1.mp4')

# Define the scaling factor
scaling_factor = 0.5

# Iterate until the user hits the 'Esc' key
while True:
    # Capture the current frame
    _, frame = cap.read()

    # Resize the frame
    frame = cv2.resize(frame, None, 
            fx=scaling_factor, fy=scaling_factor, 
            interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    '''
    Ідентифікація облич на потоковому відео з каскадами Хаара     
    '''

    # Run the face detector on the grayscale image
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around the face
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

    # Display the output
    cv2.imshow('Face Detector', frame)

    # Check if the user hit the 'Esc' key
    c = cv2.waitKey(1)
    if c == 27:
        break

# Release the video capture object
cap.release()

# Close all the windows
cv2.destroyAllWindows()
