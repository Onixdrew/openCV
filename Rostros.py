# Importamos las librerias
import cv2

from pymongo import MongoClient
import imutils

# Realizamos VideoCaptura
camera = cv2.VideoCapture(0)

# Leemos el modelo
net = cv2.dnn.readNetFromCaffe("opencv_face_detector.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Parametros del modelo
# Tamaño
anchonet = 300
altonet = 300
# Valores medios de los canales de color
media = [104, 117, 123]
umbral = 0.7

# Conectarse a la base de datos MongoDB Atlas
client = MongoClient("mongodb+srv://Andrew:BEyKKt0ai4ArRqBQ@cluster0.qj0gkdd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# Seleccionar la base de datos y la colección
db = client["vision_artificial"]
usuarios = db["usuarios"]

def capturar_imagenes(usuario_id, nombre_usuario):
    contador = 0
    # Empezamos
    while True:
        
        # Leemos los frames
        ret, frame = camera.read()
    
        # Si hay error
        if not ret:
            break
    
        # Realizamos conversion de forma
        frame = cv2.flip(frame, 1)
    
        # Extraemos info de los frames
        altoframe = frame.shape[0]
        anchoframe = frame.shape[1]
    
        # Preprocesamos la imagen
        # Images - Factor de escala - tamaño - media de color - Formato de color(BGR-RGB) - Recorte
        blob = cv2.dnn.blobFromImage(frame, 1.0, (anchonet, altonet), media, swapRB = False, crop = False)
    
        # Corremos el modelo
        net.setInput(blob)
        detecciones = net.forward()
    
        # Iteramos
        for i in range(detecciones.shape[2]):
            # Extraemos la confianza de esa deteccion
            conf_detect = detecciones[0,0,i,2]
            # Si superamos el umbral (70% de probabilidad de que sea un rostro)
            if conf_detect > umbral:
                # Extraemos las coordenadas
                xmin = int(detecciones[0, 0, i, 3] * anchoframe)
                ymin = int(detecciones[0, 0, i, 4] * altoframe)
                xmax = int(detecciones[0, 0, i, 5] * anchoframe)
                ymax = int(detecciones[0, 0, i, 6] * altoframe)
    
                # Dibujamos el rectangulo
                rostro= cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                # Texto que vamos a mostrar
                label = "Confianza de deteccion: %.4f" % conf_detect
                # Tamaño del fondo del label
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                # Colocamos fondo al texto
                cv2.rectangle(frame, (xmin, ymin - label_size[1]), (xmin + label_size[0], ymin + base_line),
                              (0,0,0), cv2.FILLED)
                # Colocamps el texto
                cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                
                
                # Guardar solo el rostro en MongoDB
                rostro_recortado = frame[ymin:ymax, xmin:xmax]
               
                
                ret, buffer = cv2.imencode('.jpg', rostro_recortado)
                image_data = buffer.tobytes()
                usuarios.update_one({"codigo": usuario_id}, {"$push": {"fotos": image_data}})
                contador += 1
    
        cv2.imshow("DETECCION DE ROSTROS", frame)
    
        # Esperar la pulsación de la tecla 'q' para salir o alcanzar el límite de imágenes
        if cv2.waitKey(1) == ord('q') or contador >= 80:
            break


    # Liberar la cámara y cerrar la ventana
    camera.release()
    cv2.destroyAllWindows()
    if contador > 0:
        print(f'Se capturaron {contador} imágenes para el usuario {nombre_usuario}')
    else:
        print('No se capturaron imágenes')

def main():
    codigo = int(input("Ingrese su ID: "))
    usuario_existente = usuarios.find_one({"codigo": codigo})
    if not usuario_existente:
        nombre_usuario = input("Ingrese su nombre: ")
        usuarios.insert_one({"codigo": codigo, "Nombre": nombre_usuario, "fotos": []})
        capturar_imagenes(codigo, nombre_usuario)
    else:
        print('Ya existe un usuario con ese código')

if __name__ == "__main__":
    main()