#  ///////////////  Reconocimiento Facial(GPT guia) //////////////


import cv2
# from pymongo import MongoClient
import os

# Cargar el clasificador en cascada para la detección de rostros
                                    #  módulo en OpenCV que proporciona
                                    # rutas  predefinidas a los 
                                    # clasificadores en cascada Haar  +  archivo XML que contiene la información del clasificador en cascada para la detección de rostros frontales
faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")


# Cargar el modelo(clase) de reconocimiento facial LBPH de openCV
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Leer el modelo previamente entrenado para el reconocimiento facial
recognizer.read('modelo_vision_artificial.xml')
# read(): Este método(funcion) del objeto recognizer se utiliza para cargar un modelo de reconocimiento facial desde un archivo XML

# Conectarse a la base de datos MongoDB Atlas
# client = MongoClient("mongodb+srv://Andrew:BEyKKt0ai4ArRqBQ@cluster0.qj0gkdd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# Seleccionar la base de datos y la colección
# db = client["vision_artificial"]
# usuarios = db["usuarios"]

# dataPath= r'c:\Users\SENA_Aprendiz\Documents\Onixdrew\OpenCV\data'
dataPath= r'C:\Users\Andrew\Documents\Proyecto_Vision_Artificial\OpenCV\data'
listaPersonas= os.listdir(dataPath)
print('lista de usuarios: ', listaPersonas)


camara = cv2.VideoCapture(0, cv2.CAP_ANY)

while True:
    # Leer un fotograma del video
    ret, frame = camara.read()
    
    if not ret:
        print("No se pudo capturar el fotograma")
        break
        
    # Voltear horizontalmente la imagen
    frame= frameNormal = cv2.flip(frame, 1)
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    
        
        cv2.putText(frame, '{}'.format(result), (x, y-5),1,1.3,(255,255,0),1, cv2.LINE_AA )
        
        # consultaCodigo=usuarios.find_one({"codigo":result[0]})
        #  se comprueba la confianza de la prediccion para asignar el nombre de la persona
        # result[1] es la confianza de prediccion, ya que predict devuelve dos variables(id, conf)
        
        if result[1] < 80:
            
            # /////////////////// Mejorar esta linea, aqui se presenta el error
            
            cv2.putText(frame,'{}'.format(listaPersonas[result[0]]), (x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0),2)
            
        else:
            cv2.putText(frame,'Desconocido',(x,y-20), 2, 0.8, (0,0,255),1 , cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w, y+h),(0,0,255),2)

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
    cv2.imshow('RECONOCIMENTO FACIAL', frame)
    
    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar todas las ventanas
camara.release()
cv2.destroyAllWindows()






# //////////////////////////////////////////////////////////////////////////////////////////////////



# import cv2
# from pymongo import MongoClient

# # Cargar el clasificador en cascada para la detección de rostros
# faceClassifier = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')

# # Cargar el modelo de reconocimiento facial LBPH de OpenCV
# recognizer = cv2.face.LBPHFaceRecognizer_create()

# # Leer el modelo previamente entrenado para el reconocimiento facial
# recognizer.read('modelo_vision_artificial.xml')

# # Conectarse a la base de datos MongoDB Atlas
# client = MongoClient("mongodb+srv://Andrew:BEyKKt0ai4ArRqBQ@cluster0.qj0gkdd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# # Seleccionar la base de datos y la colección
# db = client["vision_artificial"]
# usuarios = db["usuarios"]

# camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# # Contador de fotogramas procesados
# frame_count = 0

# while True:
#     # Leer un fotograma del video
#     ret, frame = camara.read()
    
#     if not ret:
#         break
    
#     # Incrementar el contador de fotogramas
#     frame_count += 1
    
#     # Procesar solo cada segundo fotograma
#     if frame_count % 2 != 0:
#         continue
    
#     # Voltear horizontalmente la imagen
#     frameNormal = cv2.flip(frame, 1)
    
#     # Convertir la imagen a escala de grises
#     gray = cv2.cvtColor(frameNormal, cv2.COLOR_BGR2GRAY)
    
#     # Detectar rostros en la imagen
#     faces = faceClassifier.detectMultiScale(gray, 1.3, 5)
    
#     # Para cada rostro detectado, realizar el reconocimiento facial
#     for (x, y, w, h) in faces:
#         # Extraer la región del rostro
#         rostro = gray[y:y+h, x:x+w]
#         # Cambiar el tamaño del rostro
#         rostro = cv2.resize(rostro, (150, 150))
        
#         # Realizar el reconocimiento facial en el rostro
#         id_, conf = recognizer.predict(rostro)
        
#         # Obtener la información del usuario desde la base de datos
#         consultaCodigo = usuarios.find_one({"codigo": id_})
        
#         # Dibujar un rectángulo alrededor del rostro detectado
#         cv2.rectangle(frameNormal, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
#         # Mostrar el nombre predicho sobre el rostro
#         if conf < 65:
#             nombre = consultaCodigo["Nombre"]
#         else:
#             nombre = "Desconocido"
#         cv2.putText(frameNormal, nombre, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
    
#     # Mostrar el fotograma resultante
#     cv2.imshow('RECONOCIMIENTO FACIAL', frameNormal)
    
#     # Salir del bucle si se presiona 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Liberar la captura y cerrar todas las ventanas
# camara.release()
# cv2.destroyAllWindows()






# //////////////////////////////// 2 clasificadores (frontal, perfil derecho)//////////////////////////////////////////////


# import cv2
# from pymongo import MongoClient

# # Cargar el clasificador en cascada para la detección de rostros frontal
# faceClassifierFrontal = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')

# # Cargar el clasificador en cascada para la detección de rostros lateral (perfil)
# faceClassifierProfile = cv2.CascadeClassifier('lbpcascade_profileface.xml')

# # Cargar el modelo de reconocimiento facial LBPH de OpenCV
# recognizer = cv2.face.LBPHFaceRecognizer_create()

# # Leer el modelo previamente entrenado para el reconocimiento facial
# recognizer.read('modelo_vision_artificial.xml')

# # Conectarse a la base de datos MongoDB Atlas
# client = MongoClient("mongodb+srv://Andrew:BEyKKt0ai4ArRqBQ@cluster0.qj0gkdd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# # Seleccionar la base de datos y la colección
# db = client["vision_artificial"]
# usuarios = db["usuarios"]

# camara = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# # Contador de fotogramas procesados
# frame_count = 0

# while True:
#     # Leer un fotograma del video
#     ret, frame = camara.read()
    
#     if not ret:
#         break
    
#     # Incrementar el contador de fotogramas
#     frame_count += 1
    
#     # Procesar solo cada segundo fotograma
#     if frame_count % 3 != 0:
#         continue
    
#     # Voltear horizontalmente la imagen
#     frameNormal = cv2.flip(frame, 1)
    
#     # Convertir la imagen a escala de grises
#     gray = cv2.cvtColor(frameNormal, cv2.COLOR_BGR2GRAY)
    
#     # Detectar rostros frontales en la imagen
#     facesFrontal = faceClassifierFrontal.detectMultiScale(gray, 1.3, 5)
    
#     # Detectar rostros de perfil en la imagen
#     facesProfile = faceClassifierProfile.detectMultiScale(gray, 1.3, 5)
    
#     # Combinar las detecciones de rostros frontales y de perfil
#     faces = list(facesFrontal) + list(facesProfile)
    
#     # Para cada rostro detectado, realizar el reconocimiento facial
#     for (x, y, w, h) in facesFrontal:
#         # Extraer la región del rostro
#         rostro = gray[y:y+h, x:x+w]
#         # Cambiar el tamaño del rostro
#         rostro = cv2.resize(rostro, (200, 200))
        
#         # Realizar el reconocimiento facial en el rostro
#         id_, conf = recognizer.predict(rostro)
        
#         # Obtener la información del usuario desde la base de datos
#         consultaCodigo = usuarios.find_one({"codigo": id_})
        
#         # Dibujar un rectángulo alrededor del rostro detectado
#         cv2.rectangle(frameNormal, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
#         # Mostrar el nombre predicho sobre el rostro
#         if conf < 70:
#             nombre = consultaCodigo["Nombre"]
#         else:
#             nombre = "Desconocido"
#         cv2.putText(frameNormal, nombre, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
    
#     # Mostrar el fotograma resultante
#     cv2.imshow('RECONOCIMIENTO FACIAL', frameNormal)
    
#     # Salir del bucle si se presiona 'q'
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break

# # Liberar la captura y cerrar todas las ventanas
# camara.release()
# cv2.destroyAllWindows()
