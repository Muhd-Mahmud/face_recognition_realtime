import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone



from EncodeGenerator import encodeListKnown, encodeListKnownwithIds

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

ImageBackground = cv2.imread('Resources/background.png')

# importing the mode images into a list
folderModepath = 'Resources/Modes'
modepathlist = os.listdir(folderModepath)
imgModeList = []
for path in modepathlist:
    imgModeList.append(cv2.imread(os.path.join(folderModepath, path)))
# print(len(imageModeList))

# Load the encoding file
print("Loading Encoded File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownwithIds = pickle.load(file)
file.close()
encodeListKnown,studentIds = encodeListKnownwithIds
#print(studentIds)
print("Encode File Loaded")

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Also BGR2RGB, not BGRA2RGB

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    # Resize background to be smaller
    ImageBackground = cv2.imread('Resources/background.png')
    ImageBackground = cv2.resize(ImageBackground, (1000, 700))

    # Place webcam feed
    ImageBackground[220:520, 69:524] = cv2.resize(img, (455, 300))

    # Display mode image on the right panel
    # Width = 967 - 620 = 347, Height = 637 - 56 = 581
    if len(imgModeList) > 0:
        modeImg = cv2.resize(imgModeList[3], (347, 581))
        ImageBackground[56:637, 620:967] = modeImg

        for encodeFace, faceLoc in  zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            #print("matches", matches)
            #print("faceDis",faceDis)

            matchIndex = np.argmin(faceDis)
            #print("Match Index", matchIndex)

            if matches[matchIndex]:
                # print("known Face Detector")
                # print(studentIds[matchIndex])
                y1, x2, y2, x1 = faceLoc

                # Scale back from the small image (0.25 scale)
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                # Now scale to match the resized webcam feed (640x480 -> 455x300)
                scale_x = 455 / 640
                scale_y = 300 / 480

                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y)
                y2 = int(y2 * scale_y)

                # Add offset for position in background (69, 220)
                bbox = 69 + x1, 220 + y1, x2 - x1, y2 - y1
                ImageBackground = cvzone.cornerRect(ImageBackground, bbox, rt=0)




    #cv2.imshow("webcam", img)
    cv2.imshow("Face Attendance", ImageBackground)
    cv2.waitKey(1)