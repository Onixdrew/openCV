#  ///////////////  Reconocimiento Facial(GPT guia) //////////////


import cv2
from pymongo import MongoClient

# Cargar el clasificador en cascada para la detección de rostros
                                    #  módulo en OpenCV que proporciona
                                    # rutas  predefinidas a los 
                                    # clasificadores en cascada Haar  +  archivo XML que contiene la información del clasificador en cascada para la detección de rostros frontales
faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar el modelo(clase) de reconocimiento facial LBPH de openCV
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Leer el modelo previamente entrenado para el reconocimiento facial
recognizer.read('modelo_vision_artificial.xml')
# read(): Este método(funcion) del objeto recognizer se utiliza para cargar un modelo de reconocimiento facial desde un archivo XML

# Conectarse a la base de datos MongoDB Atlas
client = MongoClient("mongodb+srv://Andrew:6yRZzkGdCsFPGPs0@cluster0.qj0gkdd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# Seleccionar la base de datos y la colección
db = client["vision_artificial"]
usuarios = db["usuarios"]

camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # Leer un fotograma del video
    ret, frame = camara.read()
    
    if ret == False: break
    
    # Voltear horizontalmente la imagen
    frameNormal = cv2.flip(frame, 1)
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frameNormal, cv2.COLOR_BGR2GRAY)
    auxFrame=gray.copy()
    
    # Detectar rostros en la imagen
    faces = faceClassifier.detectMultiScale(gray, 1.3, 5)
    
    # Para cada rostro detectado, realizar el reconocimiento facial
    for (x, y, w, h) in faces:
        # Dibujar un rectángulo alrededor del rostro detectado
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        rostro= auxFrame[y:y+h, x:x+w]
        rostro= cv2.resize(rostro,(150,150), interpolation= cv2.INTER_CUBIC)
        result=recognizer.predict(rostro)
        
        cv2.putText(frameNormal, '{}'.format(result), (x, y-5),1,1.3,(255,255,0),1, cv2.LINE_AA )
        
        consultaCodigo=usuarios.find_one({"codigo":result[0]})
        #  se comprueba la confianza de la prediccion para asignar el nombre de la persona
        # result[1] es la confianza de prediccion, ya que predict devuelve dos variables(id, conf)
        
        if result[1] < 98:
            
            cv2.putText(frameNormal,'{}'.format(consultaCodigo["Nombre"]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frameNormal,(x,y),(x+w, y+h),(0,255,0),2)
            
        else:
            cv2.putText(frameNormal,'Desconocido',(x,y-20), 2, 0.8, (0,0,255),1 , cv2.LINE_AA)
            cv2.rectangle(frameNormal,(x,y),(x+w, y+h),(0,0,255),2)

        
        # Realizar el reconocimiento facial en la región del rostro
                                        #  Esto extrae la región de la imagen que contiene el rostro detectado.
        # id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
        # Esta función toma una imagen(o una región de una imagen) como entrada y devuelve dos valores: id_ y conf.
        
        # id_: Este es el ID asociado a la persona reconocida. En el proceso de entrenamiento, cada persona tiene un ID único asociado con su rostro.
        
        # conf: Esta es la confianza o certeza de la predicción. Indica cuán seguro está el modelo en su predicción. Generalmente, los valores más bajos indican mayor confianza.
        # Un valor bajo significa que el modelo encontró una buena coincidencia entre el rostro detectado y las caras en el conjunto de entrenamiento.
        
        
        # Si la confianza es alta, mostrar el nombre predicho
        # if conf < 70:
            # Aquí deberías tener un diccionario o base de datos que asocie los IDs con los nombres de las personas
        #     nombre = "Persona " + str(id)
        # else:
        #     nombre = "Desconocido"
        
        # Mostrar el nombre predicho sobre el rostro
        # cv2.putText(frame, nombre, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
    
  
    # Mostrar el fotograma resultante
    cv2.imshow('RECONOCIMENTO FACIAL', frameNormal)
    
    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar todas las ventanas
camara.release()
cv2.destroyAllWindows()