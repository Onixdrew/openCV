import cv2

camera = cv2.VideoCapture(0)

haar = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

while True:

    ret, frame = camera.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = haar.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        pt1=(x,y)
        pt2=(x+w,y+h)
        
        cv2.rectangle(frame, pt1 ,pt2 ,(255, 0, 0), 3)
        cv2.rectangle(frame, pt1 ,(x+100,y+40) , (255, 0, 0),-1)
        cv2.putText(frame,'Face',(x+10,y+30),font,0.9,(255,255,255),2)
                  
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()
