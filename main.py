import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
from encodeGenerator import peopleFaceListWithId
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://realtimerecognizer-ac6e4-default-rtdb.firebaseio.com/',
    'storageBucket': 'realtimerecognizer-ac6e4.appspot.com'

})

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# imgBg = cv2.imread('library/halah.png')
folderModePath = 'Library/Modes'
modePathList = os.listdir(folderModePath)
imageModeList = []
for path in modePathList:
    imageModeList.append(cv2.imread(os.path.join(folderModePath, path)))
    print(len(imageModeList))

# load process the encoding file
file = open('EncodeFile.p', 'rb')
peopleFaceListWithId = pickle.load(file)
file.close()
peopleFaceList, peopleID = peopleFaceListWithId
print(peopleID)

while True:
    _, img = cam.read()

    smallerImage = cv2.resize(img, (0, 0), None, 0.6, 0.6)
    smallerImage = cv2.cvtColor(smallerImage, cv2.COLOR_BGR2RGB)

    faceCurrentFrame = face_recognition.face_locations(smallerImage)
    encodeCurrentFrame = face_recognition.face_encodings(
        smallerImage, faceCurrentFrame)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
    cv2.imshow('Face Detector(webcam)', img)

    # Compare image and camera
    for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame):
        # matches = face_recognition.compare_faces(peopleFaceList, encodeFace)

        matches = face_recognition.compare_faces(
            peopleFaceList, encodeFace, tolerance=0.7)
        distanceComparison = face_recognition.face_distance(
            peopleFaceList, encodeFace)
        matches = [bool(match) for match in matches]
        # print("matches", matches)
        # print("distanceComparison", distanceComparison)

        matchIndex = np.argmin(distanceComparison)
        # print("Match Index", matchIndex)

        if matches[matchIndex]:
            # print("Known Face Detected")
            print(peopleID[matchIndex])

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cam.release()
