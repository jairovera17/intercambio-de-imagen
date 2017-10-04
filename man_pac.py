import sys
import numpy as np
import cv2


print(str(sys.argv))
if(len(sys.argv)!=3):
    print ('INPUT REQUIRED: <FACE-CASCADE CLASSIFIER> <IMAGE>')
    exit()

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(sys.argv[1])
pacman_img = cv2.imread(sys.argv[2])

while (True):
   
    ret, img = cap.read()

    #Operaciones en el frame 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
   
    for (x,y,w,h) in faces:
        temp_pac = cv2.resize(pacman_img,(w,h), 0)     
        roi_color = img[y:y+h, x:x+w]
        img[y:y+h, x:x+w] = temp_pac
       
    cv2.imshow('jairo-vera',img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


