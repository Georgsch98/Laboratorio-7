#30/3/2024: este codigo no me funciona porque no tengo suficiente memoria para correrlo
from astropy.io import fits
import numpy as np
import os

# Directorio que contiene los archivos FITS
#directorio = r'C:\Users\georg\OneDrive\Escritorio\Labo 6\codigos labo 6\im_sintetizada'
directorio = r'F:\con_papel_de_calcar'
#directorio = r'D:\imagenes de javier\fits'
directorio_salida = r'F:\filtros'
if not os.path.exists(directorio_salida):
    os.makedirs(directorio_salida)
contador=0
for archivo in os.listdir(directorio):
   
    if archivo.endswith('.fits'):
        ruta_archivo = os.path.join(directorio, archivo)
        with fits.open(ruta_archivo) as hdulist:
            datos_imagen = hdulist[0].data.astype(np.float64)
            filas, columnas = datos_imagen.shape
            print(datos_imagen.shape)
            contador+=1
            print(contador)
            # Crear las matrices de máscaras para este archivo
            mascara_11 = np.zeros((int(filas/2),int(columnas/2)))
            mascara_12 = np.zeros((int(filas/2),int(columnas/2)))
            mascara_21 = np.zeros((int(filas/2),int(columnas/2)))
            mascara_22 = np.zeros((int(filas/2),int(columnas/2)))

            # Iterar sobre las filas y columnas para generar las máscaras
            for i in range(int(filas/2)):
                for j in range(int(columnas/2)):
                        mascara_11[i, j]=datos_imagen[2*i,2*j]
                        mascara_12[i, j]=datos_imagen[2*i+1,2*j]
                        mascara_21[i, j]=datos_imagen[2*i,2*j+1]
                        mascara_22[i, j]=datos_imagen[2*i+1,2*j+1]
            # Guardar las máscaras en archivos FITS
            nombre_base = os.path.splitext(archivo)[0]
            fits.writeto(os.path.join(directorio_salida, f'{nombre_base}_m11.fits'), mascara_11, overwrite=True)
            fits.writeto(os.path.join(directorio_salida, f'{nombre_base}_m12.fits'), mascara_12, overwrite=True)
            fits.writeto(os.path.join(directorio_salida, f'{nombre_base}_m21.fits'), mascara_21, overwrite=True)
            fits.writeto(os.path.join(directorio_salida, f'{nombre_base}_m22.fits'), mascara_22, overwrite=True)
