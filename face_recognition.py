# pylint:disable=no-member

import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
color_palette = {
    'Ben Afflek': (0, 255, 0),
    'Elton John': (0, 0, 255),
    'Jerry Seinfield': (255, 0, 0),
    'Madonna': (255, 255, 0),
    'Mindy Kaling': (0, 255, 255)
}
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'Faces/val/ben_afflek/5.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 13)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y + h, x:x + w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, color_palette[people[label]], thickness=1)
    cv.rectangle(img, (x, y), (x + w, y + h), color_palette[people[label]], thickness=2)
cv.imshow('Detected Face', img)

cv.waitKey(0)

# READ A VIDEO
# capture = cv.VideoCapture(0)
#
# haar_cascade = cv.CascadeClassifier('haar_face.xml')
#
# people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
# color_palette = {
#     'Ben Afflek': (0, 255, 0),
#     'Elton John': (0, 0, 255),
#     'Jerry Seinfield': (255, 0, 0),
#     'Madonna': (255, 255, 0),
#     'Mindy Kaling': (0, 255, 255)
# }
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')
#
# face_recognizer = cv.face.LBPHFaceRecognizer_create()
# face_recognizer.read('face_trained.yml')
#
# while True:
#     isTrue, frame = capture.read()
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # Detect the face in the image
#     faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 3)
#     for (x, y, w, h) in faces_rect:
#         faces_roi = gray[y:y + h, x:x + w]
#
#         label, confidence = face_recognizer.predict(faces_roi)
#         print(f'Label = {people[label]} with a confidence of {confidence}')
#
#         cv.putText(frame, str(people[label]), (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, color_palette[people[label]], thickness=1)
#         cv.rectangle(frame, (x, y), (x + w, y + h), color_palette[people[label]], thickness=2)
#
#     cv.imshow('Video', frame)
#
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break
#
# capture.release()
# cv.destroyAllWindows()
