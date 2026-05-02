import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone

CONFIDENCE_THRESHOLD = 0.5  # Reject weak matches

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load background ONCE outside the loop (performance fix)
ImageBackground = cv2.imread('Resources/background.png')
ImageBackground = cv2.resize(ImageBackground, (1000, 700))

# Load mode images
folderModepath = 'Resources/Modes'
modepathlist = os.listdir(folderModepath)
imgModeList = []
for path in modepathlist:
    imgModeList.append(cv2.imread(os.path.join(folderModepath, path)))

# Load encoding file
print("Loading Encoded File ...")
with open('EncodeFile.p', 'rb') as file:
    encodeListKnownwithIds = pickle.load(file)
encodeListKnown, studentIds = encodeListKnownwithIds
print("Encode File Loaded")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    # Reset background each frame from the pre-loaded copy
    frameBG = ImageBackground.copy()

    # Place webcam feed
    frameBG[220:520, 69:524] = cv2.resize(img, (455, 300))

    # Display mode image safely
    if len(imgModeList) > 0:
        modeIdx = min(3, len(imgModeList) - 1)
        modeImg = cv2.resize(imgModeList[modeIdx], (347, 581))
        frameBG[56:637, 620:967] = modeImg

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        # Only accept match if confidence is above threshold
        if matches[matchIndex] and faceDis[matchIndex] < CONFIDENCE_THRESHOLD:
            print(f"Known Face: {studentIds[matchIndex]} (distance: {faceDis[matchIndex]:.2f})")

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            scale_x = 455 / 640
            scale_y = 300 / 480
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)

            bbox = 69 + x1, 220 + y1, x2 - x1, y2 - y1
            frameBG = cvzone.cornerRect(frameBG, bbox, rt=0)

    cv2.imshow("Face Attendance", frameBG)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
