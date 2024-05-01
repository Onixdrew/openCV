#///////////////////////// entrenamiento generar archivo xml /////////////////////////////////////



# import cv2
# import numpy as np
# from pymongo import MongoClient
# import os # os permite interactuar con funciones del computador

# # Conectarse a la base de datos MongoDB Atlas
# client = MongoClient("mongodb+srv://Andrew:BEyKKt0ai4ArRqBQ@cluster0.qj0gkdd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# # Seleccionar la base de datos y la colección
# db = client["vision_artificial"]
# usuarios = db["usuarios"]





# # Inicializar listas para almacenar caras y etiquetas
# faces = []
# labels = []

# # Recorrer los documentos para obtener las imágenes de las personas
# for user in usuarios.find():
#     # Obtener la etiqueta de la persona
#     label = user["codigo"]
    
#     # Obtener las imágenes de la persona
#     images = user["fotos"]
    
#     # Recorrer las imágenes de la persona
#     for image_binary in images:
#         # Convertir los datos binarios en un array de bytes
#         image_data = np.frombuffer(image_binary, dtype=np.uint8)
#         #  convierte los datos binarios de la imagen en un arreglo NumPy de tipo uint8 (entero sin signo de 8 bits).
#         # np.uint8, que representa números enteros sin signo de 8 bits (es decir, números enteros en el rango de 0 a 255).
        
#         # Decodificar la imagen utilizando OpenCV
#         image = cv2.imdecode(image_data,cv2.IMREAD_GRAYSCALE)
#         # Esta función de OpenCV decodifica una imagen codificada y la carga en la memoria como una matriz NumPy. En este caso, image_data es el arreglo NumPy que contiene los datos binarios de la imagen que deseas decodificar.
#         # cv2.IMREAD_GRAYSCALE: Este es un indicador que se utiliza para especificar que se debe decodificar la imagen en escala de grises.
        
#         # Agregar la imagen decodificada y la etiqueta(_id) a las listas
#         faces.append(image)
#         labels.append(label)




# # # /////// comprobar si se cargo las imagenes
# # for face in faces:
# #     # Mostrar la imagen en una ventana de OpenCV
# #     cv2.imshow('Imagen', face)
    
# #     # Esperar hasta que se presione la tecla 'q' para salir
# #     if cv2.waitKey(0) & 0xFF == ord('q'):
# #         break
# # cv2.destroyAllWindows()


# #///////////// Entrenar el modelo LBPH Face Recognizer

# # Inicializar el modelo entrenador de rostros LBPH
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#                 # El método train espera que las etiquetas sean de tipo numpy.ndarray o cv2.UMat
# face_recognizer.train(faces, np.array(labels))
# # Cada imagen en faces está asociada con su correspondiente etiqueta en labels.
# # Es decir, la primera imagen en faces se asocia con la primera etiqueta en labels,
# # la segunda imagen en faces se asocia con la segunda etiqueta en labels, y así sucesivamente.

# # verifiar si el archivo xml existe, para actulizarlo por el nuevo
# if os.path.exists("modelo_vision_artificial.xml"):
#     # Si el archivo XML existe, eliminarlo para escribir el nuevo modelo
#     os.remove("modelo_vision_artificial.xml")

# face_recognizer.write("modelo_vision_artificial.xml")


# print("Modelo entrenado y guardado exitosamente.")








# //////////////////////////////////////////////////////////////////////////////////////////////////








import cv2
import numpy as np
from pymongo import MongoClient
import os

# Conectarse a la base de datos MongoDB Atlas
client = MongoClient("mongodb+srv://Andrew:BEyKKt0ai4ArRqBQ@cluster0.qj0gkdd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# Seleccionar la base de datos y la colección
db = client["vision_artificial"]
usuarios = db["usuarios"]

def cargar_imagenes_y_etiquetas():
    faces = []
    labels = []

    # Recorrer los documentos para obtener las imágenes de las personas
    for user in usuarios.find():
        # Obtener la etiqueta de la persona
        label = user["codigo"]
        
        # Obtener las imágenes de la persona
        images = user["fotos"]
        
        # Recorrer las imágenes de la persona
        for image_binary in images:
            # Convertir los datos binarios en un array de bytes
            image_data = np.frombuffer(image_binary, dtype=np.uint8)
            
            # Decodificar la imagen utilizando OpenCV
            image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
            
            # Agregar la imagen decodificada y la etiqueta a las listas
            faces.append(image)
            labels.append(label)
    
    return faces, labels

# # /////// comprobar si se cargo las imagenes

# faces,labels=cargar_imagenes_y_etiquetas()
# for face in faces:
#     # Mostrar la imagen en una ventana de OpenCV
#     cv2.imshow('Imagen', face)
    
#     # Esperar hasta que se presione la tecla 'q' para salir
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()
# /////////////////////////////////////////////////////

def entrenar_modelo(faces, labels):
    # Inicializar el modelo entrenador de rostros LBPH
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Entrenar el modelo
    face_recognizer.train(faces, np.array(labels))
    
    # Verificar si el archivo XML existe y eliminarlo si es necesario
    if os.path.exists("modelo_vision_artificial.xml"):
        os.remove("modelo_vision_artificial.xml")
    
    # Guardar el modelo entrenado
    face_recognizer.write("modelo_vision_artificial.xml")
    
    print("Modelo entrenado y guardado exitosamente.")

def main():
    # Cargar las imágenes y las etiquetas
    faces, labels = cargar_imagenes_y_etiquetas()
    
    # Entrenar el modelo
    entrenar_modelo(faces, labels)

if __name__ == "__main__":
    main()
