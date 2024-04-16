# ///////////////  toma de fotos y almacenamiento en base de datos !!!!!!!! //////////////


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

codigo=int(input("Ingrese su ID: "))
user=input("Ingrese su nombre: ")

clasificador= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
queryUser=usuarios.find_one({"codigo":codigo})

if not queryUser:
    usuarios.insert_one({"codigo":codigo,"Nombre":user,"fotos":[]})

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
        cv2.imshow("Captura del rostro", frame)
        

        # Esperar la pulsación de la tecla 'q' para salir
        if cv2.waitKey(1) == ord('q') or contador >=20:
            break

    # Liberar la cámara y cerrar la ventana
    camera.release()
    cv2.destroyAllWindows()
    print(f'el usuario {user} se registró correctamente')
    
else:
    print('ya existe usuario con ese código')
    

