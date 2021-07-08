import cv2

#load the Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Capture Video From Webcam
cap = cv2.VideoCapture(0)
#TO Use Video File as Input
#cap = cv2.VideoCapture('video.mp4')


while True:
    #Read the Frame
    _, img = cap.read()

    #Convert to Gray Scale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Draw the Rectangle around Each Faces.
    faces = face_cascade.detectMultiScale(gray,1.1,4)


    for (x,y,w,h) in faces:

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)

    #Diplay of Face Detection
    cv2.imshow('Face Detection', img)

    #Stop if ESC Button Pressed.
    k = cv2.waitKey(20) & 0xff
    if k==27:
        break

#Release / Run
cap.release()

