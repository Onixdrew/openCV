import cv2

# /////////// abrir camara/////////////

# # Se inicializa la cámara y se elige la cámara a usar: interna (0) o externa (2)
# camara = cv2.VideoCapture(0)
# # camara externa
# camaraExterna = cv2.VideoCapture(1)

# # Inicia un bucle while para capturar y mostrar continuamente los frames del video
# while True:

    ## un frame es una imagen estática que representa un instante específico en el tiempo dentro del video
    
#     # Captura el video por frames. La variable 'frame' es la imagen capturada y 'ret' indica si la cámara está disponible (True o False)
#     ret, frame = camara.read() 
    
#     # externa  
#     ret, frame2 = camaraExterna.read()   
    
#     # Muestra la imagen en una ventana. Los argumentos son ('título ventana', imagen capturada)
#     cv2.imshow('camara', frame) 
    
#     # externa  
#     cv2.imshow('camara', frame2) 
    
#     # Finaliza el programa al presionar una tecla especificada. En este caso, 'x'
#     # 0xFF en hexadecimal es 11111111 en binario. Al realizar la operación & con 0xFF, se asegura de que solo se conserven los últimos 8 bits de la tecla presionada
#     if cv2.waitKey(1) & 0xFF == ord('x'):
#         break

# # Cierra la conexión con la cámara
# camara.release()

# # externa
# camaraExterna.release()

# # Cierra todas las ventanas creadas
# cv2.destroyAllWindows()



# /////////////// Grabar video ///////////

# # Se inicializa la captura de video desde la cámara del dispositivo
# camara = cv2.VideoCapture(0)

# # Se especifica el formato de video para el archivo de salida
# formatoVideo = cv2.VideoWriter_fourcc(*'MP4V')

# # Se crea un objeto para escribir el video de salida        frames por segundo, (dimension/resolucion)
# videoResultado = cv2.VideoWriter('VideoPrueba.mp4', formatoVideo, 20.0, (640, 480))

# # Bucle mientras la camara de video esté abierta
# while (camara.isOpened()):
#     # Se lee un frame del video capturado
#     valorBool, imagen = camara.read()
    
#     # Se verifica si se ha leído correctamente un frame
#     if(valorBool):
#         # Se muestra el frame en una ventana llamada 'Video'
#         cv2.imshow('Video', imagen)
        
#         # Se escribe el frame en el archivo de video de salida
#         videoResultado.write(imagen)
        
#         # Se verifica si se ha presionado la tecla 'x' para detener la grabación
#         if cv2.waitKey(1) & 0xFF == ord('x'):
#             break
#     else:
#         break

# # Se libera la captura de video y se cierra el archivo de video de salida
# camara.release()
# videoResultado.release()

# # Se cierran todas las ventanas abiertas
# cv2.destroyAllWindows()


# ///////////////// tomar fotos///////////////


# # Se inicializa la captura de video desde la cámara
# camara = cv2.VideoCapture(0)

# # Contador para asignar nombres a las imágenes guardadas
# img_counter = 0

# # Bucle para capturar y mostrar continuamente los frames del video
# while True:
#     # Se lee un frame del video capturado
#     valorBool, frame = camara.read()
    
#     # Se muestra el frame en una ventana llamada 'video'
#     cv2.imshow('video', frame)
    
#     # Se verifica si la captura del frame fue exitosa
#     if not valorBool:
#         break
    
#     # Se espera a que el usuario presione una tecla
#     k = cv2.waitKey(1)
    
#     # Se verifica si se ha presionado la tecla 'a' (ASCII 97)
#     if k % 256 == 97:   # equivale a la letra 'a'
        
#         # Se asigna un nombre a la imagen
#         nombreImagen = f'imagen{img_counter}.png'
        
#         # Se guarda la imagen en formato PNG
#         cv2.imwrite(nombreImagen, frame)
        
#         # Se incrementa el contador de imágenes
#         img_counter += 1

# # Se libera la captura de video
# camara.release()

# # Se cierran todas las ventanas abiertas
# cv2.destroyAllWindows()


# ////////////// Deteccion de colores ///////////

# H: 0 a 360 hue: matiz - codifica la tonalidad del color
# S: 0 a 100 saturacion - codifica la intensidad del color
# V: 0 a 100 valor - codifica la luminosidad del color

# //// rangos en HSV en openCV ///

# H: 0-179
# S: 0-255
# V: 0-255

# numpy permite realizar operaciones con vectores
# import numpy

# camara= cv2.VideoCapture(0)

# # rangos de colores a detectar

#                      #  H    S  V
# azulBajo= numpy.array([100,100,20], numpy.uint8)
# azulAlto= numpy.array([125,255,255], numpy.uint8)

# kernel= numpy.ones((5,5), numpy.uint8)
# # El kernel es una matriz que se utiliza como una "ventana deslizante" sobre
# # la imagen original para aplicar operaciones de convolución, como dilatación, erosión, apertura (eliminacion de puntos(ruido) que aparecen al fondo de la imagen ), 
# # cierre (eliminacion de puntos en el objeto detectado ), entre otras. Estas operaciones son comunes en el procesamiento de imágenes para realizar
# # tareas como eliminación de ruido, detección de bordes, segmentación, etc.

# # numpy.ones(): crea una matriz de un tamaño de 5x5 píxeles.(5 filas y 5 columnas.)

# # uint8: significa "entero sin signo de 8 bits". En el contexto de OpenCV, 
# # este tipo de datos es comúnmente utilizado para representar imágenes en escala de grises 
# # o imágenes binarias, donde cada píxel se representa con un único byte (8 bits) y puede tener valores en el rango de 0 a 255.

# # permite escribir texto sobre la imagen de la camara
# writeText= cv2.FONT_HERSHEY_COMPLEX

# while True:
#     # inicializo la captura constante de frames 
#     camDisponible, frame= camara.read()
    
#     if camDisponible:
#         # convertir la imagen a HSV
#                     # imagen capturada, funcion que convierte a hsv 
#         imagenHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
#         # mascara binaria que detecta el color
        
#                     # imagen convertida, rango color bajo, rango color alto
#         mask= cv2.inRange(imagenHSV, azulBajo, azulAlto)
        
#         # 2da mascara binaria que muestra el color detectado 
#         maskOrigin= cv2.bitwise_and(frame,frame, mask=mask)
        
#         # usamos la operacion de apertura de la matriz creada(kernel)
#         opening=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
#         # calculando los momentos de la imagen binaria mask. Los momentos de una imagen
#         # son una serie de valores que describen varias propiedades de la distribución 
#         # de píxeles en la imagen, como su masa, centroide, orientación, entre otros.
#         moments= cv2.moments(mask)
        
        
#         # calcula el area del objeto(color)
#         # se extrae el área del objeto en la imagen binaria. 
#         # El área del objeto se puede obtener a partir de los momentos utilizando 
#         # la clave 'm00', que representa el momento de masa de orden 00, es decir, el total de la masa de la imagen.
#         areaObjeto=moments['m00']
        
#         if (areaObjeto > 2000000):
#         # si areaObjeto es true se hace la deteccion del color
        
#             x,y,w,h= cv2.boundingRect(opening)
#             # calcula el rectángulo delimitador alrededor de la región de interés
#             # en la imagen binaria (opening) y asigna los valores de las coordenadas x, y del
#             # punto superior izquierdo del rectángulo, así como la w (ancho) y h (alto) del 
#             # rectángulo delimitador a las variables correspondientes.

#             # se calcula el punto medio del rectángulo delimitador que se ha obtenido previamente utilizando cv2.boundingRect
#             eje_x = int((x + x+w) / 2)  # Calculando el punto medio en el eje x
#             eje_y = int((y + y+h) / 2)  # Calculando el punto medio en el eje y
#             centro= (eje_x,eje_y)
            
#             # se dibuja un rectángulo delimitador alrededor de un objeto detectado en una imagen.
#             cv2.rectangle(frame,(x,y),(x+w, y+h),(128,0,128),3)
#             # frame: Es la imagen en la que se dibujará el rectángulo delimitador. En este caso, parece ser el fotograma original en el que se detectó el objeto.
#             # (x, y): (inico) Es la coordenada superior izquierda del rectángulo delimitador. Estas coordenadas se obtienen del rectángulo delimitador calculado previamente, probablemente con la función cv2.boundingRect().
#             # (x+w, y+h): (fin) Es la coordenada inferior derecha del rectángulo delimitador. Estas coordenadas se calculan sumando el ancho (w) y el alto (h) del rectángulo delimitador a las coordenadas (x, y).
#             # (128, 0, 128): Es el color del rectángulo delimitador en el formato BGR (azul, verde, rojo). En este caso, parece ser un color púrpura oscuro.
#             # 3: Es el grosor del borde del rectángulo delimitador.

#             # permite escribir texto sobre la camara
#                  # (imagen, texto,inicio texto, variable con funcion que permite escrir texto, tamaño letra, color letra rgb, grosor)
#             cv2.putText(frame,'Azul',(x,y), writeText,0.6, (255,255,255), 3)
            
#             # visualizar los datos del centro del color
#                             # conversion a string, coordenadas de inicio texto
#             cv2.putText(frame,str(centro),(x+60, y-5), writeText,0.6, (255,255,255), 3)
            
            
#         # mostrar ventanas
        
#         # mascara binaria con el color detectado
#         # cv2.imshow('Mask Color Detectado',maskOrigin)
        
#         # mascara binaria por defecto
#                 # titulo, camara
#         cv2.imshow('Mask por defecto',mask)
        
#         # video captura
#         cv2.imshow('Imagen',frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('x'):
#             break

# # detiene el proceso de la camara    
# camara.release()
# cv2.destroyAllWindows()
        



# //////////////////  DETECCIÓN DE ROSTRO //////////////////
 
# camara= cv2.VideoCapture(0)

# # importa el archivo clasificador previamente entrenado para la detección de rostros frontales.
# clasificador= cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# # Los clasificadores en cascada son algoritmos de detección de objetos que funcionan mediante la evaluación de características en ventanas deslizantes de una imagen.

# # El archivo 'haarcascade_frontalface_alt.xml' es uno de los clasificadores en cascada preentrenados que se
# # distribuyen con OpenCV y se utiliza específicamente para la detección de rostros frontales. Ha sido entrenado con una gran cantidad
# # de imágenes positivas (rostros frontales) y negativas (imágenes sin rostros) para poder identificar correctamente los rostros en diferentes condiciones.

# # # permite escribir texto sobre la imagen de la camara
# font = cv2.FONT_HERSHEY_SIMPLEX


# while True:
#     camDisponible, imagen= camara.read()
    
#     # se convierte la imagen a escala de grises
#     # Nota: para la deteccion de objetos siempre se debe canvertir la imagen a escala de grises
#     imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
#     # detectar rostros
#     faces = clasificador.detectMultiScale(imagenGris, 1.3, 5)
#     # la variable faces ahora es una lista que contiene los rostros detectados, Cada elemento es una tupla que contiene las coordenadas (x, y, w, h) del rectángulo delimitador de un rostro detectado
    
#     #  utiliza un clasificador en cascada para detectar rostros en una imagen en escala de grises
#     # imagenGris: Es la imagen en escala de grises en la que se realizará la detección de rostros
    
#     # 1.3: Indica cuánto se reduce el tamaño de la imagen en cada escala del detector. Un valor más pequeño detectará rostros más pequeños,
#     # pero también aumentará el tiempo de procesamiento.
    
#     # 5: Este parámetro especifica el número mínimo de vecinos que se deben encontrar alrededor de una región 
#     # candidata para que esta sea considerada una detección válida. Un valor más alto reduce las detecciones falsas,
#     # pero también puede perder detecciones verdaderas.
    
    
#     # dibuja un rectángulo delimitador alrededor de cada rostro detectado en una imagen. Este proceso se repite para cada detección de rostro 
#     # en la lista faces, permitiendo así la visualización o procesamiento de cada rostro detectado individualmente.
#     for (x, y, w, h) in faces:
#         # se itera sobre la lista de detecciones de rostros faces,
#         # y en cada rostro detectado se extrae las coordenadas y sa asignan a las variables del 
#         # ciclo for para ser usadas
        
#         # se asginan las cordenadas a variables
#         pt1=(x,y)
#         pt2=(x+w,y+h)
#         # Estos puntos pt1 y pt2 pueden ser utilizados para dibujar un rectángulo delimitador alrededor de cada rostro detectado en una imagen
        
#         # dibuja un rectangulo cuando se detecte un rostro
#                         # inicio, fin, color, grosor del rectangulo
#         cv2.rectangle(imagen, pt1 ,pt2 ,(255, 0, 0), 3)
        
#         # dibuja un 2do rectangulo dentro del primero, en la esquina superior izquierda.
#         cv2.rectangle(imagen, pt1 ,(x+100,y+40) , (255, 0, 0),-1)
        
#         # se escribe un texto dentro del 2do rectangulo
#                 # (imagen, texto,coordenadas del texto(centro), variable con funcion que permite escrir texto, tamaño letra, color letra rgb, grosor)
#         cv2.putText(imagen,'Rostro',(x+10,y+30),font,0.9,(255,255,255),2)
    
#     # video captura
#     cv2.imshow('Video', imagen)
        
#     if cv2.waitKey(1) & 0xFF == ord('x'):
#         break

# # detiene el proceso de la camara    
# camara.release()
# cv2.destroyAllWindows()
 
 
#  ///////////////  Reconocimiento Facial //////////////


# # import cv2

# # Cargar el clasificador en cascada para la detección de rostros
#                                     #  módulo en OpenCV que proporciona
#                                     # rutas  predefinidas a los 
#                                     # clasificadores en cascada Haar  +  archivo XML que contiene la información del clasificador en cascada para la detección de rostros frontales
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Cargar el modelo(clase) de reconocimiento facial LBPH de openCV
# recognizer = cv2.face.LBPHFaceRecognizer_create()

# # Leer el modelo previamente entrenado para el reconocimiento facial
# recognizer.read('modelo_reconocimiento_facil.xml')
# # read(): Este método(funcion) del objeto recognizer se utiliza para cargar un modelo de reconocimiento facial desde un archivo XML


# # Capturar video desde la cámara
# cap = cv2.VideoCapture(0)

# while True:
#     # Leer un fotograma del video
#     ret, frame = cap.read()
    
#     # Convertir la imagen a escala de grises
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Detectar rostros en la imagen
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
#     # Para cada rostro detectado, realizar el reconocimiento facial
#     for (x, y, w, h) in faces:
#         # Dibujar un rectángulo alrededor del rostro detectado
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
#         # Realizar el reconocimiento facial en la región del rostro
#                                         #  Esto extrae la región de la imagen que contiene el rostro detectado.
#         id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
#         # Esta función toma una imagen(o una región de una imagen) como entrada y devuelve dos valores: id_ y conf.
        
#         # id_: Este es el ID asociado a la persona reconocida. En el proceso de entrenamiento, cada persona tiene un ID único asociado con su rostro.
        
#         # conf: Esta es la confianza o certeza de la predicción. Indica cuán seguro está el modelo en su predicción. Generalmente, los valores más bajos indican mayor confianza.
#         # Un valor bajo significa que el modelo encontró una buena coincidencia entre el rostro detectado y las caras en el conjunto de entrenamiento.
        
        
#         # Si la confianza es alta, mostrar el nombre predicho
#         if conf < 70:
#             # Aquí deberías tener un diccionario o base de datos que asocie los IDs con los nombres de las personas
#             nombre = "Persona " + str(id_)
#         else:
#             nombre = "Desconocido"
        
#         # Mostrar el nombre predicho sobre el rostro
#         cv2.putText(frame, nombre, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
    
#     # Mostrar el fotograma resultante
#     cv2.imshow('Frame', frame)
    
#     # Salir del bucle si se presiona 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Liberar la captura y cerrar todas las ventanas
# cap.release()
# cv2.destroyAllWindows()





#  ///////////////  Reconocimiento Facial (Real)!!!!!!!! //////////////

                # toma de fotos y almacenamiento en base de datos

import cv2
from pymongo import MongoClient
import imutils
#se usa para el procesamiento de imágenes de manera más fáci, como Redimensionar imágenes.Rotar imágenes.Transladar imágenes.Recortar regiones de interés (ROI) en imágenes.
from bson import Binary
# BSON es un formato binario utilizado por MongoDB para almacenar 
# y transferir datos de manera eficiente. La biblioteca bson proporciona
# herramientas para codificar y decodificar datos en formato BSON.

# Conectarse a la base de datos MongoDB Atlas
client = MongoClient("mongodb+srv://Andrew:6yRZzkGdCsFPGPs0@cluster0.qj0gkdd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# Seleccionar la base de datos y la colección
db = client["vision_artificial"]
usuarios = db["usuarios"]
user=input("Ingrese su nombre: ")

clasificador= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

usuarios.insert_one({"Nombre":user,"fotos":[]})
# Configurar la cámara
camera = cv2.VideoCapture(0)
contador=0
listaFotos=[]
while True:
    # Capturar un fotograma
    imgDisponible, frame = camera.read()
    
    if imgDisponible==False:
        break
    
    frame=imutils.resize(frame, width=320)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame=frame.copy()
    # deteccion rostro
    faces= clasificador.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w, y+h), (0,255,0), 2)
        rostro= auxFrame[y:y+h, x:x+w]
        rostro= cv2.resize(rostro, (720, 720), interpolation=cv2.INTER_CUBIC)
        
        # Guardar la imagen en MongoDB
        ret, buffer = cv2.imencode('.jpg', rostro)
        #  La función cv2.imencode() toma una imagen y la codifica en un formato específico en este caso, .jpg . Esta función devuelve dos valores: ret y buffer. ret es una bandera 
        # que indica si la codificación fue exitosa (True) o no (False). buffer es un array de bytes que contiene los datos de la imagen codificada en formato JPEG.
        
        image_data = buffer.tobytes()
        #  Convertimos el array de bytes buffer en un objeto bytes utilizando el método tobytes(). Esto nos da los datos de la imagen en formato de bytes.

        image_binary = Binary(image_data)
        # Creamos un objeto Binary de MongoDB utilizando los datos de la imagen en formato de bytes. MongoDB puede almacenar datos binarios como objetos Binary. Este paso es necesario para poder almacenar la imagen en MongoDB.
        # Los bytes son la unidad básica de almacenamiento y transferencia de datos en sistemas informáticos.
        
        usuarios.update_one({ "Nombre":user }, { "$push": { "fotos": image_binary } })
        contador+=1 
    
    # Mostrar el fotograma 
    cv2.imshow("Captura de rostros", frame)
    

    # Esperar la pulsación de la tecla 'q' para salir
    if cv2.waitKey(1) == ord('q') or contador >=20:
        break

# Liberar la cámara y cerrar la ventana
camera.release()
cv2.destroyAllWindows()
print(f'{user} se registró correctamente')






# //////////////////////////////////////////////////////////////
           # entrenamiento generar archivo xml:


# import cv2
# import numpy as np
# from pymongo import MongoClient

# # Conectarse a la base de datos MongoDB Atlas
# client = MongoClient("mongodb+srv://Andrew:6yRZzkGdCsFPGPs0@cluster0.qj0gkdd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# # Seleccionar la base de datos y la colección
# db = client["vision_artificial"]
# usuarios = db["usuarios"]

# # Inicializar el reconocedor de rostros LBPH
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# # Inicializar listas para almacenar caras y etiquetas
# faces = []
# labels = []

# # Recorrer los documentos para obtener las imágenes de las personas
# for user in usuarios.find():
#     # Obtener la etiqueta de la persona
#     label = user["_id"]
    
#     # Obtener las imágenes de la persona
#     images = user["fotos"]
    
#     # Recorrer las imágenes de la persona
#     for image_binary in images:
#         # Convertir los datos binarios en un array de bytes
#         image_data = np.frombuffer(image_binary, dtype=np.uint8)
#         #  convierte los datos binarios de la imagen en un arreglo NumPy de tipo uint8 (entero sin signo de 8 bits).
#         # np.uint8, que representa números enteros sin signo de 8 bits (es decir, números enteros en el rango de 0 a 255).
        
#         # Decodificar la imagen utilizando OpenCV
#         image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
#         # Esta función de OpenCV decodifica una imagen codificada y la carga en la memoria como una matriz NumPy. En este caso, image_data es el arreglo NumPy que contiene los datos binarios de la imagen que deseas decodificar.
#         # cv2.IMREAD_GRAYSCALE: Este es un indicador que se utiliza para especificar que se debe cargar la imagen en escala de grises.
        
#         # Agregar la imagen y la etiqueta a las listas
#         faces.append(image)
#         labels.append(label)

# # Entrenar el modelo LBPH Face Recognizer
# face_recognizer.train(faces, np.array(labels))
# # Cada imagen en faces está asociada con su correspondiente etiqueta en labels.
# # Es decir, la primera imagen en faces se asocia con la primera etiqueta en labels,
# # la segunda imagen en faces se asocia con la segunda etiqueta en labels, y así sucesivamente.

# # Guardar el modelo entrenado en un único archivo XML
# face_recognizer.save("modelo_vision_artificial.xml")

# print("Modelo entrenado y guardado exitosamente.")
