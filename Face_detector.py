import os

import cv2

import imutils


person_name = 'marco'

if not os.path.exists(f'data/{person_name}'):
    print(f'Carpeta Creada: {person_name}')
    os.makedirs(f'data/{person_name}')



#Capturar un video de la webcam

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('No se puede abrir la camara')
    exit()





face_classif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_frame = frame.copy()


    faces = face_classif.detectMultiScale(gray, 1, 3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x +  w, y + h), (0, 255, 0), 2)
        rostro = aux_frame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f'data/{person_name}/rostro_{count}.jpg', rostro)