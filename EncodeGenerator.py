import cv2
import face_recognition
import pickle
import os

# importing student images
folderpath = 'images'
pathlist = os.listdir(folderpath)
print(pathlist)
imgList = []
studentIds = []
for path in pathlist:
    imgList.append(cv2.imread(os.path.join(folderpath, path)))
    studentIds.append(os.path.splitext(path)[0])
    #print(path)
    #print(os.path.splitext(path)[0])
print(studentIds)


def findEncodings(imageslist):
    encodeList = []
    for img in imageslist:
        img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

print("Encoding Started...")
encodeListKnown = findEncodings(imgList)
encodeListKnownwithIds = [encodeListKnown,studentIds]
print("Encoding Complete")

file = open("EncodeFile.p", 'wb', )
pickle.dump(encodeListKnownwithIds,file)
file.close()
print("File Saved")
