from astropy.io import fits
import numpy as np
import os

# Directorio que contiene los archivos FITS
directorio = r'D:\imagenes de javier\fits'

# Contador para llevar la cuenta del número total de imágenes
total_imagenes = 0
i=0
# Itera sobre todos los archivos FITS en el directorio
for archivo in os.listdir(directorio):
    i+=1
    print(i)
    if archivo.endswith('.fits'):
        ruta_archivo = os.path.join(directorio, archivo)
        with fits.open(ruta_archivo) as hdulist:
            datos_imagen = hdulist[0].data.astype(np.float64)  # Convertir a float64
            total_imagenes += 1
            if total_imagenes == 1:
                promedio_datos = datos_imagen.copy()  # Inicializar con los datos de la primera imagen
            else:
                promedio_datos += (datos_imagen - promedio_datos) / total_imagenes

# Convertir los datos promediados de vuelta a uint16
# promedio_datos = promedio_datos.astype(np.uint16)
print(promedio_datos)
print(promedio_datos.shape)
