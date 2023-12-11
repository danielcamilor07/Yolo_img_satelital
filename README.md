# Doctorado en Automática
**Universidad de Pamplona**
**Doctorado en automatica**
**Procesamiento Digital de Imágenes**

**Docente: Juan D. Pulgarín-Giraldo**

**Estudiante: Daniel Camilo Ruiz Ayala**

**Estudiante: Karla Yohana Sánchez-Mojica**

Enlace de google colab:https://colab.research.google.com/drive/1f41yXjDMULqcaTgASFhktEyhAHbaemPZ?usp=sharing

<div style="font-size: 5px;">Se recopilo un conjunto de datos de imágenes de zonas urbanas con vegetación de algunas ciudades de Colombia, a través del software SASPlanet, que permite elegir varios satélites para descargar imágenes de alta resolución; en esta oportunidad se utiliza el satélite de Google. Se descargan aproximadamente 52 imágenes de ciudades del territorio Colombiano como Bogotá, Bucaramanga, Cúcuta entra otras, con unas dimensiones de 5333 pixeles de ancho y 2573 pixeles de alto, con una resolución de 96 ppp y una profundidad de 24 bits, en formato JPG, mediante código se generan recortes de dimensión (640 x 640) pixeles, generando una base de datos de 1462 imágenes con zonas verdes.</div>

<div style="font-size: 5px;">Aqui encontraras el codigo utilizado para entrenar y validar la arquitectura de YOLOnV8, y luego podras utilizar el modelo entrenado para detectar zonas verdes en nuevas imagenes satelitales.</div>

<div style="font-size: 5px;">En la carpeta Dataset encontraras las imagenes con las que se entreno y valido el modelo, tambien para evitar entrenar nuevamente el modelo, se descargo el archivo "best.pt" generalmente se refiere a un punto de control (checkpoint) del modelo que ha sido guardado durante el entrenamiento y se considera el mejor en términos de su rendimiento en un conjunto de datos de validación.</div>

<div style="font-size: 5px;">En la carpeta codigo_segmentacion encontraras el codigo desarrollado en conda python para generar la base de datos de imagenes, el cual toma una o varias imagenes satelitales, las recorta, procesa, segmenta y genera las etiquetas en formato YOLO.</div>


!git clone https://github.com/danielcamilor07/Yolo_img_satelital.git

información YOLOv8 consultada de la pagina de Ultralytics
https://docs.ultralytics.com/modes/predict/#plotting-results

#ENTRENAMIENTO Y VALIDACIÓN

%pip install ultralytics
import ultralytics
ultralytics.checks()

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='/content/Yolo_img_satelital/Dataset/dataset.yaml', epochs=200, imgsz=640, batch=20)

PREDICCIÓN

from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2 as cv2

# Ruta al modelo entrenado (asegúrate de que sea la ruta correcta)
modelo_entrenado = '/content/runs/detect/train2/weights/best.pt'

# Crear una instancia del modelo YOLO cargando el modelo entrenado
modelo = YOLO(modelo_entrenado)

# Ruta a la imagen que deseas predecir
ruta_imagen_nueva = '/content/parquepamplona.jpg'

# Realizar la predicción en la imagen
resultados = modelo(ruta_imagen_nueva)

# Load a pretrained YOLOv8n model
model = YOLO(modelo_entrenado)

# Run inference on an image
results = model(ruta_imagen_nueva)  # results list

# View results
for r in results:
    print(r.boxes)  # print the Boxes object containing the detection bounding boxes

# Visualizar los resultados
for r in results:
    im_array = r.plot()  # Generar un array numpy con las predicciones
    im = Image.fromarray(im_array[..., ::-1])  # Crear una imagen PIL RGB
    im.show()  # Mostrar la imagen
    im.save('resultadosparque.jpg')  # Guardar la imagen
    imagen=cv2.imread('resultadosparque.jpg')
    plt.imshow(im)
    plt.show()
