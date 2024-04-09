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
#varianza=[]
##for i in datos_imagen:
##    x=(i-promedio_datos)**2
##    varianza.append(x)
##print(varianza, "lista varianza")
##varianza_datos=np.sum(varianza)/(total_imagenes-1)

varianza_datos = np.sum((datos_imagen - promedio_datos) ** 2 for datos_imagen in datos_imagenes) / (total_imagenes-1)

# Convertir los datos de la varianza a uint16
#varianza_final = varianza_acumulada.astype(np.uint16)

# Guardar la imagen FITS con los datos de la varianza
#nombre_archivo_varianza = 'varianza_imagen.fits'
#fits.writeto(nombre_archivo_varianza, varianza_final, overwrite=True)

#print(f"Se ha creado la imagen de varianza '{nombre_archivo_varianza}' con los datos de varianza del conjunto de {total_imagenes} imágenes.")

print(promedio_datos.shape, "dimension del promedio")
print(varianza_datos.shape, "dimension de la varianza")
print(promedio_datos, "matriz promedio")
print(varianza_datos, "matriz varianza")

#-------------------------------
# Guardar la imagen FITS con los datos de la varianza
#nombre_archivo_varianza = 'varianza_imagen.fits'
#fits.writeto(nombre_archivo_varianza, varianza_acumulada, overwrite=True)

# Guardar la imagen FITS con los datos del promedio
#nombre_archivo_promedio = 'promedio_imagen.fits'
#fits.writeto(nombre_archivo_promedio, promedio_datos, overwrite=True)

#print(f"Se han creado las imágenes de promedio '{nombre_archivo_promedio}' y varianza '{nombre_archivo_varianza}'.")

import matplotlib.pyplot as plt

# Leer los datos del archivo FITS de promedio y varianza
#promedio_datos = fits.getdata('promedio_imagen.fits')
#varianza_datos = fits.getdata('varianza_imagen.fits')

# Convertir los datos de la imagen de 2D a 1D para facilitar el trazado
promedio_flat = promedio_datos.flatten()
varianza_flat = varianza_datos.flatten()
promedio_2=varianza_flat/promedio_flat
#------------
umbral = 1000
elementos_mayor_umbral = varianza_flat[varianza_flat > umbral]
print(elementos_mayor_umbral)
print(len(elementos_mayor_umbral))
print(varianza_flat)
print(len(varianza_flat))
#----------------------------------
# Graficar la varianza vs el promedio

plt.figure(1)
#plt.figure(figsize=(8, 6))
plt.scatter(promedio_flat, varianza_flat, s=1, alpha=0.5)
#plt.scatter(promedio_flat, promedio_2, s=1, alpha=0.5)
plt.xlabel('Promedio de los datos')
plt.ylabel('Varianza de los datos')
plt.title('Varianza vs Promedio')
plt.grid(True)


# Crear el mapa de colores

plt.figure(2)
plt.hist2d(promedio_flat, varianza_flat, bins=50, cmap='inferno')

# Agregar barra de colores
plt.colorbar(label='intencidad')


plt.figure(3)
plt.hist(promedio_2,bins=20)
plt.xlabel('Ganancia')
plt.show()
