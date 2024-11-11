from flask import Flask, render_template, Response, stream_with_context, Request
from io import BytesIO

import cv2
import numpy as np
import requests
import time

app = Flask(__name__)
# IP Address
#_URL = 'http://10.0.0.3'
_URL = 'http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg'
# Default Streaming Port
_PORT = '81'
# Default streaming route
_ST = '/stream'
SEP = ':'

#stream_url = ''.join([_URL,SEP,_PORT,_ST])
stream_url =_URL

def video_capture(show_fps=False):
    res = requests.get(stream_url, stream=True)
    last_time = time.time()
    prev_frame = None  # Inicializa la variable para detección de movimiento

    for chunk in res.iter_content(chunk_size=100000):

        if len(chunk) > 100:
            try:
                img_data = BytesIO(chunk)
                cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
               
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                N = 537
                height, width = gray.shape
                noise = np.full((height, width), 0, dtype=np.uint8)
                random_positions = (np.random.randint(0, height, N), np.random.randint(0, width, N))
                
                noise[random_positions[0], random_positions[1]] = 255

                noise_image = cv2.bitwise_or(gray, noise)

                total_image = np.zeros((height, width * 2), dtype=np.uint8)
                total_image[:, :width] = gray
                total_image[:, width:] = noise_image

                # CALCULO DE FPS 
                # ENTRADA = true / false
                if show_fps:
                    # Calcular FPS
                    current_time = time.time()
                    fps = int(1 / (current_time - last_time))
                    last_time = current_time
                    cv2.putText(total_image, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Detección de movimiento
                    if prev_frame is not None:
                        frame_diff = cv2.absdiff(prev_frame, gray)
                        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
                        motion_level = np.sum(thresh) / 255
                        
                        if motion_level > 500:  # Umbral para detectar movimiento
                            cv2.putText(total_image, "Movimiento Detectado", (10, 60), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Actualizar el fotograma previo
                    prev_frame = gray

                (flag, encodedImage) = cv2.imencode(".jpg", total_image)
                if not flag:
                    print("Advertencia: Error en la codificación de la imagen.")
                    continue

                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + b'\r\n')

            except Exception as e:
                print(e)
                continue

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_stream")
def video_stream():
    return Response(video_capture(show_fps=True),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=False)
