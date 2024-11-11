## Universidad Politecnica Salesiana
Estudiante: Edwin Paute

Asignatura : Vision por Computador

# Practica: 
> Detección de movimiento a través de resta de imágenes, convolución, aplicación de filtros para reducción
de ruido y operaciones morfológicas en la mejora de imágenes médicas

Se despliega un servido de flask usando codigo en python para la captura de video del modulo ESP32 y consecutivamente modificar su saliente

### Nota

> [!WARNING]
> Debido a algunos problemas que se tuvo a nivel personal, se opto por usar un stream online, por lo que se probo de esa manera.
> Para usar este proyecto se debe de modificar dos lineas de codigo en `app.py` que son las que redirecciona la captura de video

> [!TIP]
> Pese a que se desarrollo con recursos en linea su uso se puede dar al modulo que se esta previsto.

# Instalacion / Uso
1. Clonar desde el repositorio
2. En el directorio donde se encuentra el proyecto, abrir un terminal
   
![image](https://github.com/user-attachments/assets/bbf82d7d-3b6f-4dc4-8dc7-4f19f1cb712c)

3. escribir en el terminal ```venv\Scripts\activate```
4. como se encuenra en el directorio podemos ejecutar la aplicacion con ```py app.py```
5. El servidor ya estara en linea y podemos ingresar con la direccion dada
   ![image](https://github.com/user-attachments/assets/ebeab6b6-d2f6-498e-8d9f-afc496b2e7d5)


> [!NOTE]
> En caso de usar con el modulo se debe de hacer lo siguiente
> Descomentar las lineas
> Eliminar la linea 10
> Descomentar la linea 12 (reemplar por el url del modulo por el qeu se nos proporciona en arduino)
> Descomentar la linea 21
> ![image](https://github.com/user-attachments/assets/1566bb33-3120-4050-8af7-689d49a3988f)

# Parte 1

**Deteccion de Movimiento y Visualizacion de FPS**

Se usa las tecnicas de **Mezcla de Gaussianos (MoG)** en el código para defini una función que captura un flujo de video, detecta el movimiento en cada cuadro utilizando un sustractor de fondo, calcula los FPS (frames por segundo), y muestra una imagen unificada que incluye el cuadro original, la máscara de movimiento y el fondo. Luego, transmite cada cuadro modificado como una respuesta de imagen en formato JPEG, que se puede visualizar en tiempo real en un navegador. La función sigue capturando y procesando cuadros hasta alcanzar un número máximo de cuadros especificado.

![image](https://github.com/user-attachments/assets/8bc6f0a8-3708-40ad-9e6a-2161b793c8e4)

` La intuición detrás de tener múltiples gaussianos en lugar de uno es que un píxel puede representar muchos objetos (copos de nieve y un edificio detrás, por ejemplo). Al calcular el histograma de color utilizando fotogramas anteriores, podemos tener una idea de cuál podría ser un objeto de fondo o de primer plano.` 
[Técnicas de detección de movimiento (con código en OpenCV)](https://medium.com/@abbessafa1998/motion-detection-techniques-with-code-on-opencv-18ed2c1acfaf)

**Filtros de mejora para los problemas de iluminación**

Se crea una función donde se aplica varios filtros a una imagen en escala de grises para mejorar el contraste y la iluminación:
-  Ecualización de histograma: Mejora el contraste global de la imagen al distribuir los valores de píxeles de manera más uniforme.
-  CLAHE (Contrast Limited Adaptive Histogram Equalization): Aplica una ecualización adaptativa que mejora el contraste local en diferentes regiones de la imagen.
-  Ajuste de brillo mediante corrección de gamma: Modifica el brillo de la imagen utilizando una transformación de gama, aumentando el brillo en las áreas más oscuras.

La función retorna las imágenes procesadas con estos filtros. Donde `video_capture():` captura un flujo de video para el procesamientos en cada cuadro, y luego transmite los cuadros procesados. Dentro de esta función, se aplican los filtros de `apply_filters()` a la imagen en escala de grises.
Se genera ruido de tipo sal y pimienta (con valores aleatorios de píxeles) y se superpone a la imagen en escala de grises.
Se crea una imagen combinada (de múltiples filtros y transformaciones) en un formato de 6 celdas dispuestas en dos filas:
  -  Fila superior: la imagen original en color, la imagen en escala de grises, y la imagen con ruido.
  -  Fila inferior: la imagen ecualizada, la imagen con CLAHE y la imagen con ajuste de brillo.

![image](https://github.com/user-attachments/assets/5ac544a4-3180-40de-9a6d-243a726278f5)


**Filtros para alterar o mejorar la imagen** 

Se realizo una funcion para que se pueda ir modificando su impacto en os valores.

![image](https://github.com/user-attachments/assets/230ff535-4a6a-4488-b656-afa1c37bef4e)

esta funcion aplica los valores en los siguientes filtros:
-  Filtro Mediana (cv2.medianBlur):
Aplica un filtro de mediana con un tamaño de máscara determinado por el primer valor de la lista FILTER_SIZES[0]. En este caso, se utiliza un tamaño de 3 (máscara de 3x3).
El filtro de mediana reemplaza cada píxel con la mediana de los píxeles vecinos en la ventana de la máscara, ayudando a reducir el ruido impulsivo (por ejemplo, "sal y pimienta").
- Filtro Blur (desenfoque promedio) (cv2.blur):
Aplica un filtro de desenfoque promedio (también conocido como filtro box blur) con un tamaño de máscara definido por el segundo valor de FILTER_SIZES[1]. En este caso, se utiliza una máscara de 5x5.
El filtro realiza un promedio de los píxeles vecinos, suavizando la imagen y reduciendo el ruido.
- Filtro Gaussiano (cv2.GaussianBlur):
Aplica un filtro gaussiano con un tamaño de máscara definido por el tercer valor de FILTER_SIZES[2]. En este caso, se utiliza una máscara de 7x7.
El filtro gaussiano aplica un suavizado ponderado, donde los píxeles más cercanos al centro de la máscara tienen mayor peso. Este filtro es efectivo para reducir el ruido y difuminar los detalles de la imagen.

![image](https://github.com/user-attachments/assets/caa2170a-18ea-4b90-ad8c-25dc662d681b)

**Deteccion de Bordes**

Se realia una funcion par el procesamiento de detección de bordes en una imagen de video, 
comparando los resultados de diferentes algoritmos de detección de bordes,
con y sin aplicar filtros de suavizado. A continuación, se explica cada paso de forma descriptiva:

- Conversión a escala de grises: Convierte la imagen original (frame) a escala de grises (gray) para simplificar el procesamiento en las etapas de detección de bordes y suavizado.
- Aplicación de suavizado Gaussiano: Genera una versión suavizada de la imagen en escala de grises aplicando un filtro Gaussiano. Esto ayuda a reducir el ruido, lo cual es importante antes de aplicar algoritmos de detección de bordes para mejorar la precisión.
- Detección de bordes usando Canny:
  - Sin suavizado: Aplica el algoritmo de detección de bordes de Canny directamente sobre la imagen en escala de grises (gray), resaltando bordes sin ningún pre-procesamiento de suavizado.
  - Con suavizado: Aplica el mismo algoritmo sobre la versión suavizada de la imagen, obteniendo una versión más refinada de los bordes.
- Detección de bordes usando Sobel:
  - Sin suavizado: Calcula los gradientes en las direcciones X e Y (horizontal y vertical) sobre la imagen original en escala de grises usando el filtro de Sobel, y combina ambos para generar una imagen de bordes sin suavizado.
  - Con suavizado: Realiza el mismo cálculo de gradientes pero sobre la versión suavizada de la imagen, produciendo una versión suavizada de los bordes.
![image](https://github.com/user-attachments/assets/bae935b5-e6c4-4f23-8081-453465e2dbb9)

# Segunda Parte
## Operaciones Morfologicas

La función `operaciones()` aplica varias operaciones morfológicas a una imagen, las cuales son transformaciones basadas en la forma y estructura de los objetos presentes en la imagen. Las operaciones que realiza y su propósito son:

- Erosión: Reduce los objetos en la imagen, eliminando pequeñas irregularidades.
- Dilatación: Expande los objetos en la imagen, rellenando huecos o ampliando detalles.
- Top Hat: Resalta las áreas brillantes que son más pequeñas que el kernel, útil para detectar detalles en un fondo oscuro.
- Black Hat: Resalta las áreas oscuras más pequeñas que el kernel, útil para detectar detalles oscuros sobre un fondo claro.
- Imagen Mejorada: Combina Top Hat y Black Hat para resaltar características prominentes de la imagen.
  
Estas operaciones son aplicadas utilizando un kernel de 5x5, y los resultados de cada operación se colocan en una imagen final combinada para su visualización. La función también añade etiquetas a cada resultado para identificarlos.

![image](https://github.com/user-attachments/assets/4ae53cc5-7674-4cdd-b93e-80f779cc110a)



