import os
import cv2
import numpy as np
import face_recognition

path = 'Images for live'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # Calculate the Euclidean distances between the pairs of landmarks
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# ...



def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


encodelistKnown = findEncodings(images)
print('Encoding complete')

cap = cv2.VideoCapture(0)

# Initialize variables for blink detection
blink_frames = 0  # Number of consecutive frames with low EAR
blink_threshold = 0.20  # Adjust this threshold based on experimentation
consecutive_frames_threshold = 3  # Number of consecutive frames to detect a blink


while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)


    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        landmarks = face_recognition.face_landmarks(imgS, [faceLoc])  # Use a list with a single rectangle

        # Extract eye landmarks
        left_eye = landmarks[0]['left_eye']
        right_eye = landmarks[0]['right_eye']

        # Calculate Eye Aspect Ratio (EAR) for left and right eyes
        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)

        # Average EAR for both eyes
        ear_avg = (ear_left + ear_right) / 2.0

        # Display EAR for debugging
        print(f'Eye Aspect Ratio: {ear_avg}')

        # Check for blink based on EAR threshold
        if ear_avg < blink_threshold:
            blink_frames += 1
        else:
            blink_frames = 0

        # If consecutive frames with low EAR reach the threshold, consider it as a blink
        if blink_frames >= consecutive_frames_threshold:
            print('Blink Detected!')

        # Rest of the code for face recognition (draw rectangles, display names, etc.)
        matches = face_recognition.compare_faces(encodelistKnown, encodeFace)



        facedis = face_recognition.face_distance(encodelistKnown, encodeFace)
        print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        if facedis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
        else:
            name = 'Unknown: ACCESS DENIED'
        print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()