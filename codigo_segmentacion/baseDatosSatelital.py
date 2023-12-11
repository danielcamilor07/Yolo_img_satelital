"""
Created on Thu Dec  7 17:51:14 2023

@author: daniel camilo ruiz ayala
"""

#---------------------------------------------------
#-----rutas a configurar----------------------------
#ruta='C:/Users/daniel/Downloads/base_datos_satelital/'
ruta='C:/Users/daniel/Downloads/prueba/'
#---------------------------------------------------
#----dimenciones de las imagenes--------------------
ancho_recorte = 640 #YOLO
alto_recorte = 640 #YOLO
#ancho_recorte = 1000
#alto_recorte = 1000
#------Librerias a utilizar-------------------------
#lectura de archivos
import os
from PIL import Image
import shutil
#procesamiento de datos
import cv2 as cv2
import numpy as np
#visualización de datos 
import matplotlib.pyplot as plt
#----------------------------------------------------
#lectura de las imagenes de alta resolución que se
#encuentren en la carpeta o ruta configurada
files_img = os.listdir(ruta)
#filtro solo las imagenes
files_img = [archivo for archivo in files_img if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
print("Numero imagenes satelitales: " + str(len(files_img)))

data=[]

for k, nameImg in enumerate(files_img):
            filename=str(ruta) + str(nameImg)
            data.append([k,filename,nameImg])
            
#----------------------------------------------------
#creación de las carpetas donde se guardaran los recortes
#de las imagenes y los labels en formato .txt

def crear_carpeta_si_no_existe(ruta_carpeta):
    # Verificar si la carpeta ya existe
    if not os.path.exists(ruta_carpeta):
        # Si no existe, crearla
        os.makedirs(ruta_carpeta)
        print(f'Carpeta creada en: {ruta_carpeta}')
    else:
        print(f'La carpeta ya existe en: {ruta_carpeta}')


ruta_carpeta_images = ruta + 'images'
ruta_carpeta_label =  ruta + 'labels'
ruta_carpeta_sinZona =  ruta + 'sinzona'
# Llamar a la función para crear la carpeta si no existe
crear_carpeta_si_no_existe(ruta_carpeta_images)
crear_carpeta_si_no_existe(ruta_carpeta_label)
crear_carpeta_si_no_existe(ruta_carpeta_sinZona)

#----------------------------------------------------
#recorte de las imagenes satelitales para poder procesarlar
def recorte(ruta_salida, ruta_imagen, nombre_img, ancho_fragmento, alto_fragmento):
    nombre_img=os.path.splitext(nombre_img)[0]
    imagen_original = Image.open(ruta_imagen)
    # Número de filas y columnas
    num_filas = imagen_original.height // alto_fragmento
    num_columnas = imagen_original.width // ancho_fragmento

    # Directorio para almacenar los fragmentos
    directorio_salida = ruta_salida
    os.makedirs(directorio_salida, exist_ok=True)

    # Divide la imagen y guarda los fragmentos
    for fila in range(num_filas):
        for columna in range(num_columnas):
            # Calcula las coordenadas de la región de interés
            left = columna * ancho_fragmento
            upper = fila * alto_fragmento
            right = left + ancho_fragmento
            lower = upper + alto_fragmento

            # Extrae el fragmento de la imagen
            fragmento = imagen_original.crop((left, upper, right, lower))

            # Guarda el fragmento en un archivo
            nombre_archivo = f"{nombre_img}_{fila}_{columna}.jpg"
            ruta_fragmento = os.path.join(directorio_salida, nombre_archivo)
            fragmento.save(ruta_fragmento)


ruta_salida_images = ruta_carpeta_images + '/'
ruta_salida_labels = ruta_carpeta_label + '/'
ruta_carpeta_sinZona=ruta_carpeta_sinZona + '/'
for i in range(len(data)):
    recorte(ruta_salida_images, data[i][1], data[i][2], ancho_recorte, alto_recorte)

print("División de imagen completada.")
#----------------------------------------------------
#lectura de las imagenes que se encuentran en la carpeta images
files_recortes = os.listdir(ruta_salida_images)
#filtro solo las imagenes
files_img_rec = [archivo for archivo in files_recortes if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
print("Numero de recortes: " + str(len(files_img_rec)))

dataRecortes=[]

for k, nameRecorte in enumerate(files_img_rec):
            filenameR=str(ruta_salida_images) + str(nameRecorte)
            dataRecortes.append([k,filenameR,nameRecorte])

print("Numero recortes satelitales: " + str(len(dataRecortes)))
#------------------------------------------------------
#funciones para segmentar las zonas verdes de los recortes
def graficar(img_orig,titulo='grafica'):
    plt.imshow(img_orig)
    plt.title(titulo)
    plt.colorbar()
    plt.show()
    
def erosionar(Capar,tamanio_kernel):
    elemento_estructurante = cv2.getStructuringElement(cv2.MORPH_RECT, (tamanio_kernel, tamanio_kernel))
    imagen_erodida = cv2.erode(Capar, elemento_estructurante)
    return (imagen_erodida)
    
def dilatar(Capar,tamanio_kernel):
    elemento_estructurante = cv2.getStructuringElement(cv2.MORPH_RECT, (tamanio_kernel, tamanio_kernel))
    imagen_dilatada = cv2.dilate(Capar, elemento_estructurante)
    return (imagen_dilatada)

def generarCapa(nmin,nmax,imagen_RGB):
    if imagen_RGB is None or (isinstance(imagen_RGB, np.ndarray) and len(imagen_RGB.shape) != 3):
        # Agrega un manejo apropiado para la entrada no válida
        return 0
    imagen_RGB = np.array(imagen_RGB)    
    CanalRojo  = imagen_RGB[:,:,0]
    CanalVerde = imagen_RGB[:,:,1]
    CanalAzul  = imagen_RGB[:,:,2]

    CanalRojo[CanalRojo < nmin] = 0
    CanalRojo[CanalRojo > nmax] = 0
    CanalVerde[CanalVerde < nmin] = 0
    CanalVerde[CanalVerde > nmax] = 0
    CanalAzul[CanalAzul < nmin] = 0
    CanalAzul[CanalAzul > nmax] = 0

    imagen_RGB[:,:,0] = CanalRojo 
    imagen_RGB[:,:,1] = CanalVerde 
    imagen_RGB[:,:,2] = CanalAzul
    
    imagen_gris = cv2.cvtColor(imagen_RGB, cv2.COLOR_RGB2GRAY)
    umbral, capa = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY)
    if np.max(capa)>0:
        capa=capa/np.max(capa)
    capa=1-capa
    return capa


def filtro3Canales(nmin,nmax,ruta):
    src_img = cv2.imread(ruta)
    if src_img is not None:
        # Llamar a cv::cvtColor aquí
        imagen_BGR = cv2.imread(ruta)
        imagen_RGB = cv2.cvtColor(imagen_BGR, cv2.COLOR_BGR2RGB)
        CanalRojo  = imagen_RGB[:,:,0]
        CanalVerde = imagen_RGB[:,:,1]
        CanalAzul  = imagen_RGB[:,:,2]

        CanalRojo[CanalRojo < nmin] = 0
        CanalRojo[CanalRojo > nmax] = 0
        CanalVerde[CanalVerde < nmin] = 0
        CanalVerde[CanalVerde > nmax] = 0
        CanalAzul[CanalAzul < nmin] = 0
        CanalAzul[CanalAzul > nmax] = 0
        
        CanalVerde=CanalVerde/CanalVerde.max()
        CanalVerde=CanalVerde*255
        
        CanalRojo=CanalRojo/CanalRojo.max()
        CanalRojo=CanalRojo*255
        
        CanalAzul=CanalAzul/CanalAzul.max()
        CanalAzul=CanalAzul*255

        imagen_RGB[:,:,0] = CanalRojo 
        imagen_RGB[:,:,1] = CanalVerde 
        imagen_RGB[:,:,2] = CanalAzul
    else:
        print("Error al cargar la imagen función filtro3canales")
        print('Ruta:', ruta)
        imagen_RGB = 0
    
    return imagen_RGB
    

def generarImfil(ruta_rec, minv, maxv):
    imagen = cv2.imread(ruta_rec)
    imagen_fl = filtro3Canales(minv,maxv,ruta_rec)
    return imagen_fl


def multiplicar_RGB_Capa(imagen_RGB, capa):
    CanalRojo  = imagen_RGB[:,:,0]
    CanalVerde = imagen_RGB[:,:,1]
    CanalAzul  = imagen_RGB[:,:,2]

    CanalRojo=CanalRojo*capa
    CanalVerde=CanalVerde*capa
    CanalAzul= CanalAzul*capa
    
    imagen_RGB[:,:,0] = CanalRojo 
    imagen_RGB[:,:,1] = CanalVerde 
    imagen_RGB[:,:,2] = CanalAzul
    
    return imagen_RGB


def generarCapaSalida(imagen_fl,kr,kg,kb,minx,maxx):
    # Convertir a escala de grises
    B = imagen_fl[:,:,0]
    G = imagen_fl[:,:,1]
    R = imagen_fl[:,:,2]
    #ESCALA DE GRISES
    Rd = R.astype(float)
    Gd = G.astype(float)
    Bd = B.astype(float)
    #conversión a gris
    cp=(Rd*kr)+(Gd*kg)+(Bd*kb)
    capa_s=cp.astype(int)
    capa_s[capa_s < minx] = 0
    capa_s[capa_s > maxx] = 0
    capa_s[capa_s > 0] = 255
    umbral, capaOut = cv2.threshold(capa_s.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
    capaOut=capaOut/np.max(capaOut)
    capaOut=1-capaOut
    return capaOut

def segmentarVerde(ruta_rec):
    imagen_fl=generarImfil(ruta_rec,40,160)
    capa1=generarCapa(60,200,imagen_fl)
    capa1=1-capa1

    imagen_fl=generarImfil(ruta_rec,40,160)
    capa2=generarCapa(200,255,imagen_fl)

    #elimino zonas de semento, rojas, azules
    imagen_fl=generarImfil(ruta_rec,140,255)
    capa3=generarCapa(150,255,imagen_fl)

    capa_4=filtro3Canales(150,255,ruta_rec)
    capa4_gris = cv2.cvtColor(capa_4, cv2.COLOR_RGB2GRAY)
    umbral, capa4 = cv2.threshold(capa4_gris, 0, 255, cv2.THRESH_BINARY)
    capa4=capa4/np.max(capa4)
    capa4=1-capa4

    capa1_4=capa1*capa2*capa3*capa4

    imagen_f1=generarImfil(ruta_rec,0,255)
    imagen_f2 = multiplicar_RGB_Capa(imagen_f1,capa1_4)

    capa5=generarCapaSalida(imagen_f2,0.9,0.05,0.05,115,225)
    capa6=generarCapaSalida(imagen_f2,1,0,0,110,255)
    capa7=generarCapaSalida(imagen_f2,0,0,1,70,255)
    capa8=generarCapaSalida(imagen_f2,0,1,0,0,25)
    capa9=generarCapaSalida(imagen_f2,0.05,0.9,0.05,0,35)

    imagen_f3 = multiplicar_RGB_Capa(imagen_f2,capa5*capa6*capa7*capa8*capa9)

    imagen_f3 = imagen_f3.astype(np.float32)
    f3_normalizada = imagen_f3 / np.max(imagen_f3)
    f3_normalizada = f3_normalizada * 255
    f3_normalizada = f3_normalizada.astype(np.uint8)

    resta=f3_normalizada[:,:,2]-f3_normalizada[:,:,0]
    resta[resta<110]=0
    resta[resta>110]=255
    resta=resta.astype(np.uint8)

    umbral, capa10 = cv2.threshold(resta, 0, 255, cv2.THRESH_BINARY)
    if np.max(capa10)>0:
        capa10=capa10/np.max(capa10)
    capa10=1-capa10

    imagen_f4 = multiplicar_RGB_Capa(imagen_f3,capa10)

    resta2=(f3_normalizada[:,:,0]*0.9)-(f3_normalizada[:,:,1]*0.7)
    resta2[resta2<1]=0
    resta2[resta2>1]=255
    resta2=resta2.astype(np.uint8)

    umbral, capa11 = cv2.threshold(resta2, 0, 255, cv2.THRESH_BINARY)
    if np.max(capa11)>0:
        capa11=capa11/np.max(capa11)
    capa11=1-capa11

    imagen_f5 = multiplicar_RGB_Capa(imagen_f4,capa11)

    capa1_11=np.uint8(dilatar(erosionar(imagen_f5,7),31))
    return capa1_11

#------------------------------------------------------
#----------Función para detectar contornos-------------
def contornoRecorteYOLO(original_image,titulo,ruta_archivo,ruta_el,ruta_mov):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

    # Aplicar un umbral para obtener una imagen binaria
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dibujar contornos en la imagen original
    cv2.drawContours(original_image, contours, -1, (255, 0, 0), 2)

    # Obtener las dimensiones de la imagen original
    height, width = original_image.shape[:2]

    # Crear una lista para almacenar las etiquetas YOLO
    yolo_labels = []

    # Iterar sobre los contornos
    for contour in contours:
        # Obtener las coordenadas del rectángulo delimitador
        x, y, w, h = cv2.boundingRect(contour)
        print(x,y,w,h)
        # Normalizar las coordenadas
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        normalized_width = w / width
        normalized_height = h / height
        # Mostrar la imagen con contornos
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Agregar la etiqueta YOLO a la lista
        yolo_labels.append(f"0 {x_center} {y_center} {normalized_width} {normalized_height}")
    
    # Mostrar la imagen con contornos
    graficar(original_image,titulo)
    # Imprimir las etiquetas YOLO
    # Abre el archivo en modo de escritura ('w')
    # Si el archivo no existe, se creará; si existe, se sobrescribirá
    nameext=str(titulo)
    namelabel=os.path.splitext(nameext)[0]
    ruta_archivo=ruta_archivo + namelabel +'.txt'
    if len(yolo_labels)>0:
        # Escribe contenido en el archivo
        with open(ruta_archivo, 'w') as archivo:
            for label in yolo_labels:
                archivo.write(str(label) + '\n')
                #print(label)
    else:
        try:
            # Rutas de origen y destino
            ruta_origen = ruta_el
            ruta_destino = ruta_mov + namelabel +'.jpg'
            # Copiar el archivo de origen al destino
            print(ruta_origen)
            print(ruta_destino)
            shutil.copy(ruta_origen, ruta_destino)
            # Intenta eliminar el archivo
            os.remove(ruta_el)
            print(f'Archivo {ruta_archivo} eliminado exitosamente.')
        except OSError as e:
            # Maneja posibles errores, por ejemplo, si el archivo no existe
            print(f'Error al eliminar el archivo {ruta_el}: {e}')

#------------------------------------------------------
#llamo a la función segmentar y envio los recortes
for i in range(len(dataRecortes)):
    recorteSegmentado=segmentarVerde(dataRecortes[i][1])
    contornoRecorteYOLO(recorteSegmentado,dataRecortes[i][2],ruta_salida_labels,dataRecortes[i][1],ruta_carpeta_sinZona)
    
print('Finalizado con exito')
    
    

