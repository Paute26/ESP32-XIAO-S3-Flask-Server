from flask import Flask, render_template, Response
import cv2
import ffmpeg
import numpy as np
import time

app = Flask(__name__)

# URL del flujo de video HLS
_URL = 'http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg'

#Variables
# Constantes
MAX_FRAMES = 1000
LEARNING_RATE = -1  # -1 para que el sustractor ajuste automáticamente el fondo
fgbg = cv2.createBackgroundSubtractorMOG2()

def apply_filters(image):
    """Aplica diferentes filtros para mejorar el contraste y la iluminación."""
    
    # 1. Ecualización de histograma
    equalized_image = cv2.equalizeHist(image)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image)

    # 3. Filtro de aumento de brillo (cambio de gama)
    gamma = 1.5  # Valor de gama para incrementar el brillo
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0][i] = np.clip((i * gamma), 0, 255)
    gamma_corrected_image = cv2.LUT(image, lookUpTable)

    return equalized_image, clahe_image, gamma_corrected_image

def video_capture():
    # Usamos ffmpeg para leer el video del flujo m3u8
    cap = cv2.VideoCapture(_URL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Si no hay más frames, salir del bucle

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Aplicamos los filtros
        equalized_image, clahe_image, gamma_corrected_image = apply_filters(gray)
        
        # Aplicamos filtro de ruido sal y pimienta
        N = 537
        height, width = gray.shape
        noise = np.full((height, width), 0, dtype=np.uint8)
        random_positions = (np.random.randint(0, height, N), np.random.randint(0, width, N))

        noise[random_positions[0], random_positions[1]] = 255
        noise_image = cv2.bitwise_or(gray, noise)
        
        # Creamos una imagen en color (3 canales)
        # Espacio suficiente para apilar 6 imágenes en 2 filas
        total_image = np.zeros((height * 2, width * 3, 3), dtype=np.uint8)  
        
        # Colocamos la imagen original en color en la primera fila
        total_image[:height, :width, :] = frame
                
        # Colocamos la imagen en escala de grises en la primera fila (segunda columna)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convertimos la escala de grises a BGR para mostrarla
        total_image[:height, width:2*width, :] = gray_bgr

        # Colocamos la imagen con filtro de ruido en la primera fila (tercera columna)
        noise_bgr = cv2.cvtColor(noise_image, cv2.COLOR_GRAY2BGR)  # Aseguramos que esté en 3 canales
        total_image[:height, 2*width:, :] = noise_bgr

        # Colocamos la imagen con ecualización de histograma en la segunda fila (primera columna)
        equalized_bgr = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
        total_image[height:, :width, :] = equalized_bgr

        # Colocamos la imagen con CLAHE en la segunda fila (segunda columna)
        clahe_bgr = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
        total_image[height:, width:2*width, :] = clahe_bgr

        # Colocamos la imagen con ajuste de brillo (gamma correction) en la segunda fila (tercera columna)
        gamma_bgr = cv2.cvtColor(gamma_corrected_image, cv2.COLOR_GRAY2BGR)
        total_image[height:, 2*width:, :] = gamma_bgr
        
        (flag, encodedImage) = cv2.imencode(".jpg", total_image)
        if not flag:
            continue

        yield (b'--frame\r\n' 
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    cap.release()

def fps_deteccion(): 
    cap2 = cv2.VideoCapture(_URL)
    frame_count = 0
    last_time = time.time()
    
    while frame_count < MAX_FRAMES:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        
        # Aplicar el sustractor de fondo MOG2 para obtener la máscara de movimiento
        motion_mask = fgbg.apply(frame2, LEARNING_RATE)
        
        # Obtener la imagen de fondo calculada por MOG2
        background = fgbg.getBackgroundImage()
        
        height, width, canales = frame2.shape
        
        # Espacio suficiente para apilar 6 imágenes en 2 filas
        mod_image = np.zeros((height, width * 2, 3), dtype=np.uint8) 
        
        # Imagen Original en la primera celda
        mod_image[:height, :width, :] = frame2 
        
        # Cálculo de FPS y detección de movimiento
        current_time = time.time()
        fps = int(1 / (current_time - last_time))
        last_time = current_time
        
        # Colocamos los FPS en la imagen original
        cv2.putText(mod_image, f"FPS: {fps}", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Concatenar la máscara de movimiento y el fondo en una sola imagen para visualizar juntos
        if background is not None:
            combined = cv2.hconcat([cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR), background])
        else:
            combined = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

        # Redimensionar la imagen combinada para que tenga el tamaño adecuado
        combined_resized = cv2.resize(combined, (width, height))
        
        # Colocar la imagen combinada (máscara de movimiento + fondo) en la segunda celda
        mod_image[:height, width:2*width, :] = combined_resized
        
        # Codificar la imagen modificada como JPEG
        (flag, encodedImage) = cv2.imencode(".jpg", mod_image)
        if not flag:
            continue

        # Generar cada fotograma como una respuesta para transmitir en Flask
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

        frame_count += 1

    cap2.release()

    
    
    
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_stream")
def video_stream():
    return Response(video_capture(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/fps_movedeteccion")
def fps_movedeteccion():
    return Response(fps_deteccion(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=False)
