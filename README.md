# Doctorado en Automática
**Universidad de Pamplona**
**Doctorado en automatica**
**Procesamiento Digital de Imágenes**

**Docente: Juan D. Pulgarín-Giraldo**

**Estudiante: Daniel Camilo Ruiz Ayala**

**Estudiante: Karla Yohana Sánchez-Mojica**

Enlace de google colab:https://colab.research.google.com/drive/1f41yXjDMULqcaTgASFhktEyhAHbaemPZ?usp=sharing

<div style="font-size: 5px;">Mediante el software libre llamado SAS Planet, exportamos imagenes del satelite de google earth, ampliamos la zona de interes con un zoom de 19, y delimitamos la zona en base a la resolución de la pantalla para tener siempre la misma dimensión al exportarse.</div>

<div style="font-size: 5px;">Configuramos la exportación comenzando con la selección del zoom a 21 y georeferenciación de la imagen en formato W, seleccionamos la salida de la imagen en formato .JPG</div><br>

<div style="font-size: 5px;">Al final generamos una imagen .jpg con dimensiones de ancho=5333px y alto=2573px, YOLO recibe imagenes de 416px para eso vamos a dividir la imagenes obteniendo una matrix de 12x6 para un total de 72 imagenes.</div>


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
