from flask import Flask, render_template, Response
import cv2
import ffmpeg
import numpy as np
import time

app = Flask(__name__)

# URL del flujo de video HLS
_URL = 'http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg'

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


def video_capture(show_fps=False):
    # Usamos ffmpeg para leer el video del flujo m3u8
    cap = cv2.VideoCapture(_URL)

    last_time = time.time()
    prev_frame = None  # Inicializa la variable para detección de movimiento

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

        # Cálculo de FPS y detección de movimiento
        if show_fps:
            current_time = time.time()
            fps = int(1 / (current_time - last_time))
            last_time = current_time
            cv2.putText(total_image, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if prev_frame is not None:
                frame_diff = cv2.absdiff(prev_frame, gray)
                _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
                motion_level = np.sum(thresh) / 255

                if motion_level > 500:  # Umbral para detectar movimiento
                    cv2.putText(total_image, "Movimiento Detectado", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            prev_frame = gray

        (flag, encodedImage) = cv2.imencode(".jpg", total_image)
        if not flag:
            continue

        yield (b'--frame\r\n' 
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    cap.release()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_stream")
def video_stream():
    return Response(video_capture(show_fps=True),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=False)
