#30/3/2024: este codigo no me funciona porque no tengo suficiente memoria para correrlo
from astropy.io import fits
import numpy as np
import os

# Directorio que contiene los archivos FITS
directorio = r'D:\imagenes de javier\fits'

# Contador para llevar la cuenta del número total de imágenes
total_imagenes = 0

datos_imagenes = []

i=0
# Itera sobre todos los archivos FITS en el directorio
for archivo in os.listdir(directorio):
    i+=1
    print(i)
    if archivo.endswith('.fits'):
        ruta_archivo = os.path.join(directorio, archivo)
        with fits.open(ruta_archivo) as hdulist:
            datos_imagen = hdulist[0].data.astype(np.float64)  # Convertir a float64
            datos_imagenes.append(hdulist[0].data)
            total_imagenes += 1
            if total_imagenes == 1:
                promedio_datos = datos_imagen.copy()  # Inicializar con los datos de la primera imagen
            else:
                promedio_datos += (datos_imagen - promedio_datos) / total_imagenes

# Calcular la varianza del conjunto de datos de todas las imágenes
varianza_acumulada = np.sum((datos_imagen - promedio_datos) ** 2 for datos_imagen in datos_imagenes) / (total_imagenes-1)

# Convertir los datos de la varianza a uint16
#varianza_final = varianza_acumulada.astype(np.uint16)

# Guardar la imagen FITS con los datos de la varianza
#nombre_archivo_varianza = 'varianza_imagen.fits'
#fits.writeto(nombre_archivo_varianza, varianza_final, overwrite=True)

#print(f"Se ha creado la imagen de varianza '{nombre_archivo_varianza}' con los datos de varianza del conjunto de {total_imagenes} imágenes.")

print(promedio_datos.shape, "dimension del promedio")
print(varianza_acumulada.shape, "dimension de la varianza")
print(promedio_datos, "matriz promedio")
print(varianza_acumulada, "matriz varianza")
