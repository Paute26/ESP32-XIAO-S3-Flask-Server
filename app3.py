
# Author: vlarobbyk
# Version: 1.0
# Date: 2024-10-20
# Description: A simple example to process video captured by the ESP32-XIAO-S3 or ESP32-CAM-MB in Flask.


from flask import Flask, render_template, Response, stream_with_context, Request
from io import BytesIO
import cv2
import numpy as np
import requests
import time

app = Flask(__name__)

# URL del stream MJPEG
_URL = 'http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg'
stream_url = _URL

# Inicializamos el Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Variable para el cálculo de FPS
prev_frame_time = 0
fps_buffer = []  # Para suavizar la fluctuación de FPS


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
    global prev_frame_time, fps_buffer
    res = requests.get(stream_url, stream=True)
    for chunk in res.iter_content(chunk_size=1024):

        if len(chunk) > 100:
            try:
                # Convertir chunk a imagen
                img_data = BytesIO(chunk)
                cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)

                if cv_img is None:
                    print("Error: Unable to decode image")
                    continue

                # Obtenemos el tiempo actual para FPS
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
                prev_frame_time = new_frame_time

                # Suavizar los FPS usando un promedio móvil
                fps_buffer.append(fps)
                if len(fps_buffer) > 10:  # Tomamos un promedio de los últimos 10 FPS
                    fps_buffer.pop(0)
                avg_fps = np.mean(fps_buffer)

                # Convertimos imagen a escala de grises
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

                # Aplicamos los filtros
                equalized_image, clahe_image, gamma_corrected_image = apply_filters(gray)

                # Detectamos movimiento aplicando background subtractor
                fgmask = fgbg.apply(gray)
                motion_detected = cv2.countNonZero(fgmask) > 500  # Threshold para detección de movimiento

                # Aplicamos filtro de ruido sal y pimienta
                N = 537
                height, width = gray.shape
                noise = np.full((height, width), 0, dtype=np.uint8)
                random_positions = (np.random.randint(0, height, N), np.random.randint(0, width, N))

                noise[random_positions[0], random_positions[1]] = 255
                noise_image = cv2.bitwise_or(gray, noise)

                # Combinamos las seis imágenes:
                # 1. Imagen original a color (video normal)
                # 2. Imagen en escala de grises
                # 3. Imagen con filtro de ruido
                # 4. Imagen con ecualización de histograma
                # 5. Imagen con CLAHE
                # 6. Imagen con ajuste de brillo (gamma correction)

                # Creamos una imagen en color (3 canales)
                total_image = np.zeros((height * 2, width * 3, 3), dtype=np.uint8)  # Espacio suficiente para apilar 6 imágenes en 2 filas

                # Colocamos la imagen original en color en la primera fila
                total_image[:height, :width, :] = cv_img

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

                # Agregamos el texto de FPS en la primera imagen (original a color)
                fps_text = f"FPS: {avg_fps:.2f}"
                cv2.putText(total_image, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Codificar la imagen para transmisión
                flag, encoded_image = cv2.imencode(".jpg", total_image)
                if not flag:
                    continue

                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encoded_image) + b'\r\n')

            except Exception as e:
                print(e)
                continue


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_stream")
def video_stream():
    return Response(video_capture(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=False)
