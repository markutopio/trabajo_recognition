import cv2

# Cargar la cascada

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Capturar video de una webcam

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('No se puede abrir la camara')
    exit()


# Usar archivo de video como input

while True:
    #Leer un frame
    _, img = cap.read()


    #Transformar a escala de grises

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    #Detectar los rostros

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    #Aplicar el detector sobre la imagen (el rectangulo que rodea la cara)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


    #Mostrar la imagen dada en una ventana

    cv2.imshow('img', img)

    #Detecta si se presiona la tecla esc para salir y sale del bucle

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break



#Cierra el archivo de video o capturadora

cap.release()





