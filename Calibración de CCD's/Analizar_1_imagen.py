#30/3/2024: este codigo no me funciona porque no tengo suficiente memoria para correrlo

import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


directorio = r'F:\filtros'
archivo = "D1_0s__m21.fits"
ruta_archivo = os.path.join(directorio, archivo)

# Verificar si el archivo existe antes de intentar abrirlo
if os.path.exists(ruta_archivo):
    # Abre el archivo FITS
    hdulist = fits.open(ruta_archivo)
    datos_imagen = hdulist[0].data-256

    # Calcula la varianza en cada fila
    varianza_por_fila = np.var(datos_imagen,axis=1, ddof=1)
    promedio_por_fila = np.mean(datos_imagen,axis=1)
#    varianza_por_fila = np.var(datos_imagen[0:200,:],axis=1, ddof=1)
#    promedio_por_fila = np.mean(datos_imagen[0:200,:],axis=1)



# Realiza el ajuste lineal

def modelo_lineal(x, g, b):
    return g * x + b

parametros, covarianza = curve_fit(modelo_lineal, promedio_por_fila, varianza_por_fila)
g,b = parametros
varianza_por_fila_ajustado = modelo_lineal(promedio_por_fila, g, b)


# Imprime los parámetros del ajuste
print("Parámetros del ajuste:")
print(f"Pendiente (g): {g}")
print(f"Intercepto (b): {b}")

# Visualiza los datos y el ajuste
plt.figure(0)
plt.scatter(promedio_por_fila, varianza_por_fila, label='Datos')
plt.plot(promedio_por_fila, varianza_por_fila_ajustado, color='red', label='Ajuste lineal')
plt.xlabel('Esperanza')
plt.ylabel('Varianza')
plt.legend()
plt.grid(True)

# Extraer la diagonal de la matriz de covarianza
diagonal_covarianza = np.diag(covarianza)

# Calcular el error estándar de los parámetros
error_estandar_g = diagonal_covarianza[0]
error_estandar_b = diagonal_covarianza[1]

# Imprimir los errores estándar
print("Errores estándar del ajuste:")
print(f"Error estándar de la pendiente (m): {error_estandar_g}")
print(f"Error estándar del intercepto (b): {error_estandar_b}")


print(varianza_por_fila)
plt.figure(1)
x_cordinate=np.linspace(1,len(varianza_por_fila),len(varianza_por_fila))
plt.plot(promedio_por_fila,varianza_por_fila, '*')
#plt.plot(x_cordinate,datos_imagen[:, 20])
plt.xlabel('Esperanza')
plt.ylabel('Varianza')
plt.title('Varianza vs Esperanza')
plt.grid()


M=[[1,2,3],[1,1,4],[1,1,1]]
print()
plt.figure(2)
plt.hist2d(promedio_por_fila,varianza_por_fila, bins=50, cmap='inferno')
plt.xlabel('Esperanza')
plt.ylabel('Varianza')
plt.title('Varianza vs Esperanza')
# Agregar barra de colores
plt.colorbar(label='Intensidad')

plt.show()