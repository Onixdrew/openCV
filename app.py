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


# Se inicializa la captura de video desde la cámara
camara = cv2.VideoCapture(0)

# Contador para asignar nombres a las imágenes guardadas
img_counter = 0

# Bucle para capturar y mostrar continuamente los frames del video
while True:
    # Se lee un frame del video capturado
    valorBool, frame = camara.read()
    
    # Se muestra el frame en una ventana llamada 'video'
    cv2.imshow('video', frame)
    
    # Se verifica si la captura del frame fue exitosa
    if not valorBool:
        break
    
    # Se espera a que el usuario presione una tecla
    k = cv2.waitKey(1)
    
    # Se verifica si se ha presionado la tecla 'a' (ASCII 97)
    if k % 256 == 97:   # equivale a la letra 'a'
        
        # Se asigna un nombre a la imagen
        nombreImagen = f'imagen{img_counter}.png'
        
        # Se guarda la imagen en formato PNG
        cv2.imwrite(nombreImagen, frame)
        
        # Se incrementa el contador de imágenes
        img_counter += 1

# Se libera la captura de video
camara.release()

# Se cierran todas las ventanas abiertas
cv2.destroyAllWindows()
