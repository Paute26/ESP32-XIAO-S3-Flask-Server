from flask import Flask, render_template, Response
import cv2
import ffmpeg
import numpy as np
import time

app = Flask(__name__)

# URL del flujo de video HLS
_URL = 'http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg'
# IP Address
#_URL = 'http://10.0.0.3'
# Default Streaming Port
_PORT = '81'
# Default streaming route
_ST = '/stream'
SEP = ':'


#Variables
#stream_url = ''.join([_URL,SEP,_PORT,_ST])
# Constantes
MAX_FRAMES = 1000
LEARNING_RATE = -1  # -1 para que el sustractor ajuste automáticamente el fondo
# Definir los tamaños de las máscaras para los filtros
FILTER_SIZES = [3, 5, 7]  # Tamaños de máscaras para los filtros
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

def filters_avance(frame):
    """Aplica los filtros mediana, blur y Gaussiano con diferentes tamaños de máscara."""
    
    # Filtro mediana
    median_filtered = cv2.medianBlur(frame, FILTER_SIZES[0])  # Tamaño 3
    # Filtro blur
    blur_filtered = cv2.blur(frame, (FILTER_SIZES[1], FILTER_SIZES[1]))  # Tamaño 5x5
    # Filtro Gaussiano
    gaussian_filtered = cv2.GaussianBlur(frame, (FILTER_SIZES[2], FILTER_SIZES[2]), 0)  # Tamaño 7x7

    return median_filtered, blur_filtered, gaussian_filtered

def add_filters_avance():
    cap3 = cv2.VideoCapture(_URL)
    frame_count = 0
    MAX_FRAMES = 1000  # O cualquier valor que desees

    while frame_count < MAX_FRAMES:
        ret3, frame3 = cap3.read()
        if not ret3:
            break
        
        height, width, canales = frame3.shape
        
        # Aplicar los filtros
        median_filtered, blur_filtered, gaussian_filtered = filters_avance(frame3)
        
        # Crear un espacio para apilar las imágenes con los filtros
        bor_image = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)  # Espacio para 4 imágenes (original + 3 filtros)

        # Colocar la imagen original en la primera celda
        bor_image[:height, :width, :] = frame3
        
        # Colocar la imagen con filtro mediana en la segunda celda
        bor_image[:height, width:2*width, :] = median_filtered
        
        # Colocar la imagen con filtro blur en la tercera celda
        bor_image[height:2*height,:width, :] = blur_filtered
        
        # Colocar la imagen con filtro gaussiano en la cuarta celda
        bor_image[height:2*height, width:2*width, :] = gaussian_filtered

        # Codificar la imagen combinada como JPEG
        (flag, encodedImage) = cv2.imencode(".jpg", bor_image)
        if not flag:
            continue

        # Generar cada fotograma como una respuesta para transmitir en Flask
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

        frame_count += 1

    cap3.release()


    """Aplica algoritmos de detección de bordes y compara los resultados con y sin filtros de suavizado."""
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_filtered = cv2.GaussianBlur(gray, (5, 5), 0)
    
    canny_edges_original = cv2.Canny(gray, 100, 200)
    canny_edges_filtered = cv2.Canny(gaussian_filtered, 100, 200)
    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges_original = np.uint8(np.clip(cv2.magnitude(sobel_x, sobel_y), 0, 255))
    
    sobel_x_filtered = cv2.Sobel(gaussian_filtered, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_filtered = cv2.Sobel(gaussian_filtered, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges_filtered = np.uint8(np.clip(cv2.magnitude(sobel_x_filtered, sobel_y_filtered), 0, 255))
    
    height, width = gray.shape
    edge_image = np.zeros((height * 3, width * 3, 3), dtype=np.uint8)
    
    edge_image[:height, :width, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edge_image[:height, width:2*width, :] = gray_bgr
    
    canny_bgr = cv2.cvtColor(canny_edges_original, cv2.COLOR_GRAY2BGR)
    edge_image[:height, 2*width:, :] = canny_bgr
    cv2.putText(edge_image, "Canny (sin filtro)", (2*width+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    edge_image[height:, :width, :] = cv2.cvtColor(gaussian_filtered, cv2.COLOR_GRAY2BGR)
    edge_image[height:, width:2*width, :] = cv2.cvtColor(canny_edges_filtered, cv2.COLOR_GRAY2BGR)
    cv2.putText(edge_image, "Canny (con filtro)", (width+10, height+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    edge_image[height:, 2*width:, :] = cv2.cvtColor(sobel_edges_original, cv2.COLOR_GRAY2BGR)
    cv2.putText(edge_image, "Sobel (sin filtro)", (2*width+10, height+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    edge_image[2*height:, :width, :] = cv2.cvtColor(sobel_edges_filtered, cv2.COLOR_GRAY2BGR)
    cv2.putText(total_image, "Sobel (con filtro)", (10, 2*height+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return total_image

def filtro_deteccion_borde(frame):
    """Aplica algoritmos de detección de bordes y compara los resultados con y sin filtros de suavizado."""

    height, width, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Suavizado Gaussiano
    gaussian_filtered = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detección de bordes usando Canny
    canny_edges_original = cv2.Canny(gray, 100, 200)
    canny_edges_filtered = cv2.Canny(gaussian_filtered, 100, 200)

    # Detección de bordes usando Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges_original = np.uint8(np.clip(cv2.magnitude(sobel_x, sobel_y), 0, 255))

    sobel_x_filtered = cv2.Sobel(gaussian_filtered, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_filtered = cv2.Sobel(gaussian_filtered, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges_filtered = np.uint8(np.clip(cv2.magnitude(sobel_x_filtered, sobel_y_filtered), 0, 255))

    # Redimensionar todas las imágenes al tamaño original
    gaussian_filtered = cv2.resize(cv2.cvtColor(gaussian_filtered, cv2.COLOR_GRAY2BGR), (width, height))
    canny_edges_original = cv2.resize(cv2.cvtColor(canny_edges_original, cv2.COLOR_GRAY2BGR), (width, height))
    canny_edges_filtered = cv2.resize(cv2.cvtColor(canny_edges_filtered, cv2.COLOR_GRAY2BGR), (width, height))
    sobel_edges_original = cv2.resize(cv2.cvtColor(sobel_edges_original, cv2.COLOR_GRAY2BGR), (width, height))
    sobel_edges_filtered = cv2.resize(cv2.cvtColor(sobel_edges_filtered, cv2.COLOR_GRAY2BGR), (width, height))

    # Crear un lienzo para mostrar todas las imágenes juntas
    total_image = np.zeros((height * 2, width * 3, 3), dtype=np.uint8)

    # Asignar cada imagen a la posición correspondiente
    total_image[:height, :width] = frame  # Imagen original
    total_image[:height, width:2*width] = canny_edges_original
    total_image[:height, 2*width:] = canny_edges_filtered
    total_image[height:, :width] = gaussian_filtered
    total_image[height:, width:2*width] = sobel_edges_original
    total_image[height:, 2*width:] = sobel_edges_filtered

    # Añadir texto a cada imagen
    cv2.putText(total_image, "Original", (10,  height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(total_image, "Canny (sin filtro)", (width + 10,  height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(total_image, "Canny (con filtro)", (2 * width + 10,  height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(total_image, "Suavizado Gaussiano", (10, height +  height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(total_image, "Sobel (sin filtro)", (width + 10, 2 * height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(total_image, "Sobel (con filtro)", (2 * width + 10, 2 * height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return total_image

def edge_detection():
    cap4 = cv2.VideoCapture(_URL)
    frame_count = 0
    MAX_FRAMES = 10000

    while frame_count < MAX_FRAMES:
        ret4, frame4 = cap4.read()
        if not ret4:
            break

        # Aplicar la detección de bordes y comparación
        processed_frame = filtro_deteccion_borde(frame4)
        
        # Mostrar el resultado
        (flag, encodedImage) = cv2.imencode(".jpg", processed_frame)
        if not flag:
            continue

        # Generar cada fotograma como una respuesta para transmitir en Flask
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

        frame_count += 1

    cap4.release()

###PARTE 2 
###OPEREACIONES MORFOLOGICAS

def operaciones(image):
    """Aplica operaciones morfológicas a una imagen y las devuelve en una imagen combinada."""
    
    kernel = np.ones((37, 37), np.uint8)

    # Operaciones morfológicas
    erosion = cv2.erode(image, kernel, iterations=1)
    dilation = cv2.dilate(image, kernel, iterations=1)
    top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    enhanced_image = cv2.add(image, cv2.subtract(top_hat, black_hat))

    # Preparar el contenedor de resultados con 1 fila y 4 columnas
    height, width = image.shape
    result_image = np.zeros((height, width * 4), dtype=np.uint8)
    
    # Colocar los resultados en una fila con 4 columnas
    result_image[:height, :width] = erosion
    result_image[:height, width:2*width] = dilation
    result_image[:height, 2*width:3*width] = black_hat
    result_image[:height, 3*width:] = enhanced_image

    # Etiquetas
    cv2.putText(result_image, "Erosion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(result_image, "Dilation", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(result_image, "black_hat", (2 * width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.putText(result_image, "Ecualizada", (3 * width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    return result_image


def image_manipulation():
    """Procesa tres imágenes, aplica operaciones morfológicas y devuelve una imagen combinada."""
    # Rutas a las tres imágenes que deseas procesar
    imagen1_path = "static/radiografia.jpg"
    imagen2_path = "static/radiografia2.jpg"
    imagen3_path = "static/radiografia3.jpg"
    
    # Cargar las tres imágenes en escala de grises
    imagen1 = cv2.imread(imagen1_path, 0)
    imagen2 = cv2.imread(imagen2_path, 0)
    imagen3 = cv2.imread(imagen3_path, 0)
    
     # Verifica si las imágenes se cargaron correctamente
    if imagen1 is None or imagen2 is None or imagen3 is None:
        print("Error al cargar una de las imágenes")
        return None

    # Redimensionar las imágenes a un tamaño común (por ejemplo, el tamaño de la primera imagen)
    height, width = imagen1.shape
    imagen2 = cv2.resize(imagen2, (width, height))
    imagen3 = cv2.resize(imagen3, (width, height))

    # Aplicar las operaciones a cada imagen
    resultado1 = operaciones(imagen1)
    resultado2 = operaciones(imagen2)
    resultado3 = operaciones(imagen3)

    # Apilar verticalmente los resultados
    combined_result = np.vstack((resultado1, resultado2, resultado3))

    # Codificar la imagen combinada como JPEG
    (flag, encodedImage) = cv2.imencode(".jpg", combined_result)
    if not flag:
        return None
    return encodedImage

# Llama a la función border_detection para iniciar el procesamiento
    
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
    
@app.route("/filtros_avanzados")
def filtros_avanzados():
    return Response(add_filters_avance(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/deteccion_borde")
def deteccion_borde():
    return Response(edge_detection(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")   
    
@app.route('/operacionesMorfologicas', methods=['GET'])
def operacionesMorfologicas():
    # Llamar a la función image_manipulation
    processed_frame = image_manipulation()
    
    if processed_frame is None:
        return "Error procesando las imágenes", 500
    
    # Devolver la imagen procesada
    return Response(processed_frame.tobytes(), mimetype='image/jpeg')
##Paginas 
@app.route('/division2')
def division2():
    return render_template("division2.html")

@app.route('/division3')
def division3():
    return render_template("division3.html")

@app.route('/procesosMorfologicos')
def procesosMorfologicos():
    return render_template("operaciones.html")

if __name__ == "__main__":
    app.run(debug=False)
