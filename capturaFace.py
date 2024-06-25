# ///////////////  toma de fotos y almacenamiento en base de datos !!!!!!!! //////////////


import cv2
# from pymongo import MongoClient
import imutils
import os
#se usa para el procesamiento de imágenes de manera más fáci, como Redimensionar imágenes.Rotar imágenes.Transladar imágenes.Recortar regiones de interés (ROI) en imágenes.
# from bson import Binary
# BSON es un formato binario utilizado por MongoDB para almacenar 
# y transferir datos de manera eficiente. La biblioteca bson proporciona
# herramientas para codificar y decodificar datos en formato BSON.

# Conectarse a la base de datos MongoDB Atlas
# client = MongoClient("mongodb+srv://Andrew:BEyKKt0ai4ArRqBQ@cluster0.qj0gkdd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# Seleccionar la base de datos y la colección
# db = client["vision_artificial"]
# usuarios = db["usuarios"]

clasificador= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

# codigo=int(input("Ingrese su ID: "))

# se crean la carpeta donde se va almacenar las fotos

NombreUser=input('Ingresa tu nombre: ')

dataPath= r'c:\Users\SENA_Aprendiz\Documents\Onixdrew\OpenCV\data'
dirPersona= dataPath + '/' + NombreUser



if not os.path.exists(dirPersona):
    os.makedirs(dirPersona)
    print('caperta creada: ', dirPersona)
    
    # user=input("Ingrese su nombre: ")
    # usuarios.insert_one({"codigo":codigo,"Nombre":user,"fotos":[]})

    # Configurar la cámara
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    contador=0


    # listaFotos=[]
    
    while True:
        # Capturar un fotograma
        imgDisponible, frame = camera.read()
        
        if imgDisponible==False:
            break
        
        # Voltear horizontalmente la imagen
        frameNormal = cv2.flip(frame, 1)
        
        frame=imutils.resize(frameNormal, width=750)
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame=frame.copy()
        
        # deteccion rostro
        faces= clasificador.detectMultiScale(gray,1.3,5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w, y+h), (0,255,0), 2)
            rostro= auxFrame[y:y+h, x:x+w]
            rostro= cv2.resize(rostro, (720, 720), interpolation=cv2.INTER_CUBIC)
            
            # Guardar la imagen en en la carpeta local
            cv2.imwrite(dirPersona + '/rostro_{}.jpg'.format(contador), rostro)
            
            
            # Guardar la imagen en MongoDB
            # ret, buffer = cv2.imencode('.jpg', rostro)
            #  La función cv2.imencode() toma una imagen y la codifica en un formato específico en este caso, .jpg . Esta función devuelve dos valores: ret y buffer. ret es una bandera 
            # que indica si la codificación fue exitosa (True) o no (False). buffer es un array de bytes que contiene los datos de la imagen codificada en formato JPEG.
            
            # image_data = buffer.tobytes()
            #  Convertimos el array de bytes buffer en un objeto bytes utilizando el método tobytes(). Esto nos da los datos de la imagen en formato de bytes.

            # image_binary = Binary(image_data)
            # Creamos un objeto Binary de MongoDB utilizando los datos de la imagen en formato de bytes. MongoDB puede almacenar datos binarios como objetos Binary. Este paso es necesario para poder almacenar la imagen en MongoDB.
            # Los bytes son la unidad básica de almacenamiento y transferencia de datos en sistemas informáticos.
            
            # usuarios.update_one({ "Nombre":user }, { "$push": { "fotos": image_binary } })
            contador+=1 
            

        
        # Mostrar el fotograma 
        cv2.imshow("Captura del rostro", frame)
        

        # Esperar la pulsación de la tecla 'q' para salir
        if cv2.waitKey(1) == ord('q') or contador > 100:
            print(f'Proceso finalizado, se almacenó {contador -1} fotos')
            break

    # Liberar la cámara y cerrar la ventana
    camera.release()
    cv2.destroyAllWindows()
    # print(f'el usuario {user} se registró correctamente')
    
else:
    print('ya existe usuario con ese nombre')
    





# //////////////////////////////////////////////////////////////////////////////////////////////////








# import cv2
# from pymongo import MongoClient
# import imutils

# # Conectarse a la base de datos MongoDB Atlas
# client = MongoClient("mongodb://localhost:27017")

# # Seleccionar la base de datos y la colección
# db = client["vision_artificial"]
# usuarios = db["usuarios"]

# clasificador = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')

# def capturar_imagenes(usuario_id, nombre_usuario):
#     # Configurar la cámara
#     camera = cv2.VideoCapture(0)
#     contador = 0
#     while True:
#         # Capturar un fotograma
#         ret, frame = camera.read()
        
#         if not ret:
#             break
        
#         # Voltear horizontalmente la imagen
#         frameNormal = cv2.flip(frame, 1)
        
#         # Redimensionar el fotograma
#         frame = imutils.resize(frameNormal, width=800)
        
#         # Convertir a escala de grises
#         gray = cv2.cvtColor(frameNormal, cv2.COLOR_BGR2GRAY)
        
#         # Detección de rostros
#         faces = clasificador.detectMultiScale(gray, 1.3, 5)
        
#         for (x, y, w, h) in faces:
#             # Dibujar rectángulo alrededor del rostro
#             cv2.rectangle(frameNormal, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             # Recortar el rostro
#             rostro = frameNormal[y:y+h, x:x+w]
#             # Redimensionar el rostro
#             rostro = cv2.resize(rostro, (200, 200))
#             # Guardar la imagen en MongoDB
#             ret, buffer = cv2.imencode('.jpg', rostro)
#             image_data = buffer.tobytes()
#             usuarios.update_one({"codigo": usuario_id}, {"$push": {"fotos": image_data}})
#             contador += 1
        
#         # Mostrar el fotograma
#         cv2.imshow("Captura del rostro", frameNormal)
        
#         # Esperar la pulsación de la tecla 'q' para salir o alcanzar el límite de imágenes
#         if cv2.waitKey(1) == ord('q') or contador >= 80:
#             break

#     # Liberar la cámara y cerrar la ventana
#     camera.release()
#     cv2.destroyAllWindows()
#     if contador > 0:
#         print(f'Se capturaron {contador} imágenes para el usuario {nombre_usuario}')
#     else:
#         print('No se capturaron imágenes')

# def main():
#     codigo = int(input("Ingrese su ID: "))
#     usuario_existente = usuarios.find_one({"codigo": codigo})
#     if not usuario_existente:
#         nombre_usuario = input("Ingrese su nombre: ")
#         usuarios.insert_one({"codigo": codigo, "Nombre": nombre_usuario, "fotos": []})
#         capturar_imagenes(codigo, nombre_usuario)
#     else:
#         print('Ya existe un usuario con ese código')

# if __name__ == "__main__":
#     main()




# ///////////////////// 2 clasificadores (frontal, perfil derecho)//////////////////////////////////77



# import cv2
# from pymongo import MongoClient
# import imutils

# # Conectarse a la base de datos MongoDB Atlas
# client = MongoClient("mongodb+srv://Andrew:BEyKKt0ai4ArRqBQ@cluster0.qj0gkdd.mongodb.net/")

# # Seleccionar la base de datos y la colección
# db = client["vision_artificial"]
# usuarios = db["usuarios"]

# # Cargar el clasificador en cascada para la detección de rostros frontales
# faceClassifierFrontal = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')

# # Cargar el clasificador en cascada para la detección de rostros de perfil
# faceClassifierProfile = cv2.CascadeClassifier('lbpcascade_profileface.xml')

# def capturar_imagenes(usuario_id, nombre_usuario):
#     # Configurar la cámara
#     camera = cv2.VideoCapture(0)
#     contador = 0
#     while True:
#         # Capturar un fotograma
#         ret, frame = camera.read()
        
#         if not ret:
#             break
        
#         # Voltear horizontalmente la imagen
#         frameNormal = cv2.flip(frame, 1)
        
#         # Redimensionar el fotograma
#         frame = imutils.resize(frameNormal, width=800)
        
#         # Convertir a escala de grises
#         gray = cv2.cvtColor(frameNormal, cv2.COLOR_BGR2GRAY)
        
#         # Detectar rostros frontales en la imagen
#         facesFrontal = faceClassifierFrontal.detectMultiScale(gray, 1.3, 5)
        
#         # Detectar rostros de perfil en la imagen
#         facesProfile = faceClassifierProfile.detectMultiScale(gray, 1.3, 5)
        
#         # Combinar las detecciones de rostros frontales y de perfil
#         faces = list(facesFrontal) + list(facesProfile)
        
#         # Iterar sobre los rostros detectados y guardar las imágenes
#         for (x, y, w, h) in faces:
#             # Dibujar rectángulo alrededor del rostro
#             cv2.rectangle(frameNormal, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             # Recortar el rostro
#             rostro = frameNormal[y:y+h, x:x+w]
#             # Redimensionar el rostro
#             rostro = cv2.resize(rostro, (300, 300))
#             # Guardar la imagen en MongoDB
#             ret, buffer = cv2.imencode('.jpg', rostro)
#             image_data = buffer.tobytes()
#             usuarios.update_one({"codigo": usuario_id}, {"$push": {"fotos": image_data}})
#             contador += 1
        
#         # Mostrar el fotograma
#         cv2.imshow("Captura del rostro", frameNormal)
        
#         # Esperar la pulsación de la tecla 'q' para salir o alcanzar el límite de imágenes
#         if cv2.waitKey(1) == ord('q') or contador >= 80:
#             break

#     # Liberar la cámara y cerrar la ventana
#     camera.release()
#     cv2.destroyAllWindows()
#     if contador > 0:
#         print(f'Se capturaron {contador} imágenes para el usuario {nombre_usuario}')
#     else:
#         print('No se capturaron imágenes')

# def main():
#     codigo = int(input("Ingrese su ID: "))
#     usuario_existente = usuarios.find_one({"codigo": codigo})
#     if not usuario_existente:
#         nombre_usuario = input("Ingrese su nombre: ")
#         usuarios.insert_one({"codigo": codigo, "Nombre": nombre_usuario, "fotos": []})
#         capturar_imagenes(codigo, nombre_usuario)
#     else:
#         print('Ya existe un usuario con ese código')

# if __name__ == "__main__":
#     main()
