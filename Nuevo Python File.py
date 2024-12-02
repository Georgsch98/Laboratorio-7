import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython

'''
# Configuración de los gráficos
custom_params = {
    'font.size': 15,
    'lines.linewidth': 2,
    'axes.labelsize': 35,
    'axes.titlesize': 19,
    'axes.titleweight': 'bold',
    'axes.labelpad': 10,
    'xtick.labelsize': 27,
    'ytick.labelsize': 27,
    'font.family': 'serif',
    'legend.fontsize': 27
}

plt.rcParams.update(custom_params)

ipython = get_ipython()

'''


# Cargar el archivo
filename_azul = "varianza_vs_esperanza_B.txt"  # Cambia esto por el nombre de tu archivo
filename_verde1 = "varianza_vs_esperanza_G1.txt"
filename_verde2 = "varianza_vs_esperanza_G2.txt"
filename_rojo = "varianza_vs_esperanza_R.txt"
# Leer el archivo .txt en un DataFrame
data_azul = pd.read_csv(filename_azul, sep="\s+", header=0, names=["varianza", "esperanza", "color"])
data_verde1 = pd.read_csv(filename_verde1, sep="\s+", header=0, names=["varianza", "esperanza", "color"])
data_verde2 = pd.read_csv(filename_verde2, sep="\s+", header=0, names=["varianza", "esperanza", "color"])
data_rojo = pd.read_csv(filename_rojo, sep="\s+", header=0, names=["varianza", "esperanza", "color"])


# Filtrar las filas por color == "B"(azul);"G1"(verde 1);"G2"(verde 2);"R"(rojo)

filtro_azul = data_azul[data_azul["color"] == "B"].copy()    #Azul
filtro_verde1 = data_verde1[data_verde1["color"] == "G1"].copy() #Verde 1
filtro_verde2 = data_verde2[data_verde2["color"] == "G2"].copy() #Verde 2
filtro_rojo = data_rojo[data_rojo["color"] == "R"].copy()    #Rojo


# Extraer las primeras dos columnas como arrays

#AZUL
varianza_azul = filtro_azul["varianza"].to_numpy()
esperanza_azul = filtro_azul["esperanza"].to_numpy()
ganancia_azul = varianza_azul / esperanza_azul
#VERDE 1
varianza_verde1 = filtro_verde1["varianza"].to_numpy()
esperanza_verde1 = filtro_verde1["esperanza"].to_numpy()
ganancia_verde1 = varianza_verde1 / esperanza_verde1
#VERDE 2
varianza_verde2 = filtro_verde2["varianza"].to_numpy()
esperanza_verde2 = filtro_verde2["esperanza"].to_numpy()
ganancia_verde2 = varianza_verde2 / esperanza_verde2
#ROJO
varianza_rojo = filtro_rojo["varianza"].to_numpy()
esperanza_rojo = filtro_rojo["esperanza"].to_numpy()
ganancia_rojo = varianza_rojo / esperanza_rojo
# Sacamos bordes que no aportan mucha estadistica 

#AZUL
index_azul =np.argwhere(ganancia_azul<1)
ganancia_reducida_azul=ganancia_azul[index_azul]

# VERDE 1
index_verde1 = np.argwhere(ganancia_verde1 < 1)
ganancia_reducida_verde1 = ganancia_verde1[index_verde1]

# VERDE 2
index_verde2 = np.argwhere(ganancia_verde2 < 1)
ganancia_reducida_verde2 = ganancia_verde2[index_verde2]

# ROJO
index_rojo = np.argwhere(ganancia_rojo < 1)
ganancia_reducida_rojo = ganancia_rojo[index_rojo]


# Primer gráfico: Dispersión de Varianza vs Esperanza con opacidad

# Crear bins automáticamente usando pandas
num_bins = 30  # Número de bins
filtro_azul["esperanza_bin"], bin_edges = pd.cut(
    filtro_azul["esperanza"], bins=num_bins, retbins=True, labels=False, right=False
)

# Calcular estadísticas por bin
agrupados_azul = filtro_azul.groupby("esperanza_bin").agg(
    varianza_media=("varianza", "mean"),
    varianza_std=("varianza", "std"),
    esperanza_media=("esperanza", "mean"),
    cantidad_datos=("esperanza", "size"),
)
agrupados_azul["varianza_std_normalizada"] = agrupados_azul["varianza_std"] / np.sqrt(agrupados_azul["cantidad_datos"])

# Crear bins automáticamente usando pandas
num_bins = 30  # Número de bins
filtro_verde1["esperanza_bin"], bin_edges = pd.cut(
    filtro_verde1["esperanza"], bins=num_bins, retbins=True, labels=False, right=False
)

# Calcular estadísticas por bin
agrupados_verde1 = filtro_verde1.groupby("esperanza_bin").agg(
    varianza_media=("varianza", "mean"),
    varianza_std=("varianza", "std"),
    esperanza_media=("esperanza", "mean"),
    cantidad_datos=("esperanza", "size"),
)

agrupados_verde1["varianza_std_normalizada"] = agrupados_verde1["varianza_std"] / np.sqrt(agrupados_verde1["cantidad_datos"])

# Crear bins automáticamente usando pandas
num_bins = 30  # Número de bins
filtro_verde2["esperanza_bin"], bin_edges = pd.cut(
    filtro_verde2["esperanza"], bins=num_bins, retbins=True, labels=False, right=False
)

# Calcular estadísticas por bin
agrupados_verde2 = filtro_verde2.groupby("esperanza_bin").agg(
    varianza_media=("varianza", "mean"),
    varianza_std=("varianza", "std"),
    esperanza_media=("esperanza", "mean"),
    cantidad_datos=("esperanza", "size"),
)

agrupados_verde2["varianza_std_normalizada"] = agrupados_verde2["varianza_std"] / np.sqrt(agrupados_verde2["cantidad_datos"])


# Crear bins automáticamente usando pandas
num_bins = 30  # Número de bins
filtro_rojo["esperanza_bin"], bin_edges = pd.cut(
    filtro_rojo["esperanza"], bins=num_bins, retbins=True, labels=False, right=False
)

# Calcular estadísticas por bin
agrupados_rojo = filtro_rojo.groupby("esperanza_bin").agg(
    varianza_media=("varianza", "mean"),
    varianza_std=("varianza", "std"),
    esperanza_media=("esperanza", "mean"),
    cantidad_datos=("esperanza", "size"),
)
agrupados_rojo["varianza_std_normalizada"] = agrupados_rojo["varianza_std"] / np.sqrt(agrupados_rojo["cantidad_datos"])




# Crear la variable ganancia

filtro_azul["ganancia"] = filtro_azul["varianza"] / filtro_azul["esperanza"]

# Crear bins automáticamente para la ganancia
num_bins = 30  # Número de bins
filtro_azul["ganancia_bin"], bin_edges = pd.cut(
    filtro_azul["esperanza"], bins=num_bins, retbins=True, labels=False, right=False
)

# Calcular estadísticas por bin para la ganancia
agrupados_ganancia_azul = filtro_azul.groupby("ganancia_bin").agg(
    ganancia_media=("ganancia", "mean"),
    ganancia_std=("ganancia", "std"),
    esperanza_media=("esperanza", "mean"),
    cantidad_datos=("ganancia", "count")
)

# Normalización de la desviación estándar
agrupados_ganancia_azul["ganancia_std_normalizada"] = agrupados_ganancia_azul["ganancia_std"] / np.sqrt(agrupados_ganancia_azul["cantidad_datos"])

#VERDE1

filtro_verde1["ganancia"] = filtro_verde1["varianza"] / filtro_verde1["esperanza"]

# Crear bins automáticamente para la ganancia
num_bins = 30  # Número de bins
filtro_verde1["ganancia_bin"], bin_edges = pd.cut(
    filtro_verde1["esperanza"], bins=num_bins, retbins=True, labels=False, right=False
)

# Calcular estadísticas por bin para la ganancia
agrupados_ganancia_verde1 = filtro_verde1.groupby("ganancia_bin").agg(
    ganancia_media=("ganancia", "mean"),
    ganancia_std=("ganancia", "std"),
    esperanza_media=("esperanza", "mean"),
    cantidad_datos=("ganancia", "count")
)

# Normalización de la desviación estándar
agrupados_ganancia_verde1["ganancia_std_normalizada"] = agrupados_ganancia_verde1["ganancia_std"] / np.sqrt(agrupados_ganancia_verde1["cantidad_datos"])

#Verde2

filtro_verde2["ganancia"] = filtro_verde2["varianza"] / filtro_verde2["esperanza"]

# Crear bins automáticamente para la ganancia
num_bins = 30  # Número de bins
filtro_verde2["ganancia_bin"], bin_edges = pd.cut(
    filtro_verde2["esperanza"], bins=num_bins, retbins=True, labels=False, right=False
)

# Calcular estadísticas por bin para la ganancia
agrupados_ganancia_verde2 = filtro_verde2.groupby("ganancia_bin").agg(
    ganancia_media=("ganancia", "mean"),
    ganancia_std=("ganancia", "std"),
    esperanza_media=("esperanza", "mean"),
    cantidad_datos=("ganancia", "count")
)

# Normalización de la desviación estándar
agrupados_ganancia_verde2["ganancia_std_normalizada"] = agrupados_ganancia_verde2["ganancia_std"] / np.sqrt(agrupados_ganancia_verde2["cantidad_datos"])

#ROJO
filtro_rojo["ganancia"] = filtro_rojo["varianza"] / filtro_rojo["esperanza"]

num_bins = 30  # Número de bins
filtro_rojo["ganancia_bin"], bin_edges = pd.cut(
    filtro_rojo["esperanza"], bins=num_bins, retbins=True, labels=False, right=False
)
agrupados_ganancia_rojo = filtro_rojo.groupby("ganancia_bin").agg(
    ganancia_media=("ganancia", "mean"),
    ganancia_std=("ganancia", "std"),
    esperanza_media=("esperanza", "mean"),
    cantidad_datos=("ganancia", "count")
)

# Normalización de la desviación estándar
agrupados_ganancia_rojo["ganancia_std_normalizada"] = agrupados_ganancia_rojo["ganancia_std"] / np.sqrt(agrupados_ganancia_rojo["cantidad_datos"])

valor_ganancia_azul=np.mean(ganancia_reducida_azul)
error_ganancia_azul=np.std(ganancia_reducida_azul,ddof=1)/np.sqrt(len(ganancia_reducida_azul))
print(f"ganancia azul= ({valor_ganancia_azul} ± {error_ganancia_azul} )")

valor_ganancia_verde1 = np.mean(ganancia_reducida_verde1)
error_ganancia_verde1 = np.std(ganancia_reducida_verde1, ddof=1)/np.sqrt(len(ganancia_reducida_azul))
print(f"ganancia verde1 = ({valor_ganancia_verde1} ± {error_ganancia_verde1})")

valor_ganancia_verde2 = np.mean(ganancia_reducida_verde2)
error_ganancia_verde2 = np.std(ganancia_reducida_verde2, ddof=1)/np.sqrt(len(ganancia_reducida_azul))
print(f"ganancia verde2 = ({valor_ganancia_verde2} ± {error_ganancia_verde2})")

valor_ganancia_rojo = np.mean(ganancia_reducida_rojo)
error_ganancia_rojo = np.std(ganancia_reducida_rojo, ddof=1)/np.sqrt(len(ganancia_reducida_azul))
print(f"ganancia rojo = ({valor_ganancia_rojo} ± {error_ganancia_rojo})")

################################################################################
"""
# Calcular la ganancia
filtro_azul["ganancia"] = filtro_azul["varianza"] / filtro_azul["esperanza"]

# Filtrar los datos donde la ganancia es menor o igual a 1
filtro_azul_reducido = filtro_azul[filtro_azul["ganancia"] <= 1]

# Calcular la ganancia reducida si es necesario
filtro_azul_reducido["ganancia_reducida"] = filtro_azul_reducido["ganancia"]
num_bins = 30  # Número de bins
filtro_azul_reducido["ganancia_bin"], bin_edges = pd.cut(
    filtro_azul_reducido["esperanza"], bins=num_bins, retbins=True, labels=False, right=False
)

# Calcular estadísticas por bin
agrupados_ganancia_azul_reducido = filtro_azul_reducido.groupby("ganancia_bin").agg(
    ganancia_media=("ganancia", "mean"),
    ganancia_std=("ganancia", "std"),
    esperanza_media=("esperanza", "mean"),
    cantidad_datos=("ganancia", "count")
)

# Normalizar la desviación estándar (dividido por la raíz de la cantidad de datos)
agrupados_ganancia_azul_reducido["ganancia_std_normalizada"] = agrupados_ganancia_azul_reducido["ganancia_std"] / np.sqrt(agrupados_ganancia_azul_reducido["cantidad_datos"])

# Graficar
plt.figure(figsize=(10, 6))

# Graficar los puntos de dispersión
plt.scatter(filtro_azul_reducido["esperanza"], filtro_azul_reducido["ganancia_reducida"], color='blue', alpha=0.6, label='Puntos de ganancia reducida')

# Graficar las barras de error
plt.errorbar(
    agrupados_ganancia_azul_reducido["esperanza_media"], 
    agrupados_ganancia_azul_reducido["ganancia_media"], 
    yerr=agrupados_ganancia_azul_reducido["ganancia_std_normalizada"], 
    fmt='o', 
    color='black', 
    ecolor='gray', 
    elinewidth=2, 
    capsize=4, 
    label='Error en ganancia'
)

# Etiquetas y título
plt.xlabel('Esperanza')
plt.ylabel('Ganancia Reducida')
plt.title('Ganancia Reducida vs Esperanza')

# Mostrar la leyenda
plt.legend()

# Mostrar el gráfico
plt.show()
"""
########################################################################
"""
#GRAFICOS


# Crear el gráfico de dispersión

plt.figure(figsize=(10, 6))
#plt.hist2d(esperanza_rojo, varianza_rojo, bins=500, cmap='Reds',vmin=0, vmax=2000)
plt.scatter(esperanza_rojo, varianza_rojo, color='red', s=1, label='Datos de rojo', alpha=0.005,marker='s')  # Gráfico de dispersión
#'''
plt.errorbar(
    agrupados_rojo["esperanza_media"],  # Media de esperanza por bin
    agrupados_rojo["varianza_media"],  # Media de varianza por bin
    yerr=agrupados_rojo["varianza_std"],  # Desviación estándar de varianza por bin
    fmt='s',
    color='black',
    label='Promedios y barras de error'
)#'''
#plt.colorbar(label='Intensidad')
#plt.title('Gráfico de Varianza vs Esperanza', fontsize=14)
plt.xlabel('Esperanza [ADU]',fontsize=14)
plt.ylabel('Varianza [ADU]$^2$',fontsize=14)
plt.ylim(0,500)
#plt.xlim(0,650)
plt.legend()
plt.grid(True)
#plt.show()
plt.close()
"""



# Segundo gráfico: Histograma de Ganancia
plt.figure(figsize=(8, 6))
plt.hist(ganancia_reducida_azul, bins=500,histtype='step', color='blue', alpha=0.3)
plt.hist(ganancia_reducida_verde1, bins=500,histtype='step', color='green', alpha=0.3)
plt.hist(ganancia_reducida_verde2, bins=500,histtype='step', color='orange', alpha=0.3)
plt.hist(ganancia_reducida_rojo, bins=500,histtype='step', color='red', alpha=0.3)
plt.title('Histograma de Ganancia', fontsize=14)
plt.xlabel('Ganancia')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()
#plt.close()

"""

fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True, sharey=True)  # 4 filas, 1 columna, compartiendo el eje x
plt.xlim(0, 600)
plt.ylim(0,400)
# Título general de la figura
fig.suptitle('Variance vs Mean for each color', fontsize=16)

# Primer gráfico (Azul)
axs[0].scatter(esperanza_azul, varianza_azul, color='blue', s=1 , label='Blue', alpha=0.005, marker='o')
axs[0].errorbar(
    agrupados_azul["esperanza_media"],  # Media de esperanza por bin
    agrupados_azul["varianza_media"],  # Media de varianza por bin
    yerr=agrupados_azul["varianza_std_normalizada"],  # Desviación estándar de varianza por bin
    fmt='o',
    color='black',
    label='error'
)
#axs[0].set_title('Azul')
axs[0].set_ylabel('Variance [ADU]$^2$')
axs[0].grid(True)
axs[0].legend()

# Segundo gráfico (Verde 1)
axs[1].scatter(esperanza_verde1, varianza_verde1, color='green', s=1, label='Green 1', alpha=0.005, marker='x')
axs[1].errorbar(
    agrupados_verde1["esperanza_media"],  # Media de esperanza por bin
    agrupados_verde1["varianza_media"],  # Media de varianza por bin
    yerr=agrupados_verde1["varianza_std_normalizada"],  # Desviación estándar de varianza por bin
    fmt='x',
    color='black',
    label='error'
)
#axs[1].set_title('Verde 1')
axs[1].set_ylabel('Variance [ADU]$^2$')
axs[1].grid(True)
axs[1].legend()

# Tercer gráfico (Verde 2)
axs[2].scatter(esperanza_verde2, varianza_verde2, color='darkgreen', s=1, label='Green 2', alpha=0.005, marker='x')
axs[2].errorbar(
    agrupados_verde2["esperanza_media"],  # Media de esperanza por bin
    agrupados_verde2["varianza_media"],  # Media de varianza por bin
    yerr=agrupados_verde2["varianza_std_normalizada"],  # Desviación estándar de varianza por bin
    fmt='x',
    color='black',
    label='error'
)
#axs[2].set_title('Verde 2')
axs[2].set_ylabel('Variance [ADU]$^2$')
axs[2].grid(True)
axs[2].legend()

# Cuarto gráfico (Rojo)
axs[3].scatter(esperanza_rojo, varianza_rojo, color='red', s=1, label='Red', alpha=0.005, marker='s')
axs[3].errorbar(
    agrupados_rojo["esperanza_media"],  # Media de esperanza por bin
    agrupados_rojo["varianza_media"],  # Media de varianza por bin
    yerr=agrupados_rojo["varianza_std_normalizada"],  # Desviación estándar de varianza por bin
    fmt='s',
    color='black',
    label='error'
)
#axs[3].set_title('Rojo')
axs[3].set_xlabel('Mean [ADU]')
axs[3].set_ylabel('Variance [ADU]$^2$')
axs[3].grid(True)
axs[3].legend()

# Ajustar el diseño para evitar solapamiento
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Deja espacio para el título general

# Mostrar el gráfico
plt.show()
"""

#ganancia en funcion de la esperanza

plt.figure(figsize=(8, 6))
#plt.ylim(0,5)
#plt.scatter(esperanza_azul,ganancia_azul, s=1, color='blue', alpha=0.3,label='ganancia_azul')
plt.errorbar(
    agrupados_ganancia_azul["esperanza_media"],  # Media de esperanza por bin
    agrupados_ganancia_azul["ganancia_media"],  # Media de varianza por bin
    yerr=agrupados_ganancia_azul["ganancia_std_normalizada"],  # Desviación estándar de varianza por bin
    fmt='o',
    color='blue',
    label='Blue',
    alpha=0.3,
)

#plt.scatter(esperanza_verde1,ganancia_verde1,s=1, color='green', alpha=0.3,label='ganancia_verde1')
plt.errorbar(
    agrupados_ganancia_verde1["esperanza_media"],  # Media de esperanza por bin
    agrupados_ganancia_verde1["ganancia_media"],  # Media de varianza por bin
    yerr=agrupados_ganancia_verde1["ganancia_std_normalizada"],  # Desviación estándar de varianza por bin
    fmt='x',
    color='green',
    label='Green1',
    alpha=0.3,
)
#plt.scatter(esperanza_verde2,ganancia_verde2,s=1, color='darkgreen', alpha=0.3,label='ganancia_verde2')
plt.errorbar(
    agrupados_ganancia_verde2["esperanza_media"],  # Media de esperanza por bin
    agrupados_ganancia_verde2["ganancia_media"],  # Media de varianza por bin
    yerr=agrupados_ganancia_verde2["ganancia_std_normalizada"],  # Desviación estándar de varianza por bin
    fmt='x',
    color='darkgreen',
    label='Green2',
    alpha=0.3,
)
#plt.scatter(esperanza_rojo,ganancia_rojo,s=1, color='red', alpha=0.3,label='ganancia_rojo')
plt.errorbar(
    agrupados_ganancia_rojo["esperanza_media"],  # Media de esperanza por bin
    agrupados_ganancia_rojo["ganancia_media"],  # Media de varianza por bin
    yerr=agrupados_ganancia_rojo["ganancia_std_normalizada"],  # Desviación estándar de varianza por bin
    fmt='s',
    color='red',
    label='Red',
    alpha=0.3,
)  
#plt.title('Histograma de Ganancia', fontsize=14)
plt.xlabel('Esperanza')
plt.ylabel('Ganancia')
plt.legend()
plt.grid(True)
plt.show()
#plt.close()
"""
#fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=True)
fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True, sharey=True)  # Aumenta la altura
#plt.ylim(0,5)
# Azul
#axs[0].scatter(esperanza_azul, ganancia_azul, s=1, color='blue',alpha=0.5, label='Azul')
#axs[0].hist2d(esperanza_azul, 1/ganancia_azul, color='blue', bins=500, cmap='Blues',vmin=0, vmax=5000, label='Azul')
axs[0].errorbar(
    agrupados_ganancia_azul["esperanza_media"],  # Media de esperanza por bin
    agrupados_ganancia_azul["ganancia_media"],  # Media de varianza por bin
    yerr=agrupados_ganancia_azul["ganancia_std_normalizada"],  # Desviación estándar de varianza por bin
    fmt='o',
    color='blue',
    label='Error'
)
#axs[0].set_title('Ganancia Azul')
axs[0].grid(True)
axs[0].legend()
axs[0].set_ylabel('gain')
# Verde 1
#axs[1].scatter(esperanza_verde1, ganancia_verde1, s=1, color='green',alpha=0.5, label='Verde 1')
#axs[1].hist2d(esperanza_verde1, 1/ganancia_verde1, color='green', bins=500, cmap='Greens',vmin=0, vmax=5000, label='Verde 1')
#axs[1].set_title('Ganancia Verde 1')
axs[1].errorbar(
    agrupados_ganancia_verde1["esperanza_media"],  # Media de esperanza por bin
    agrupados_ganancia_verde1["ganancia_media"],  # Media de varianza por bin
    yerr=agrupados_ganancia_verde1["ganancia_std_normalizada"],  # Desviación estándar de varianza por bin
    fmt='x',
    color='green',
    label='Error'
)
axs[1].grid(True)
axs[1].legend()
axs[1].set_ylabel('gain')

# Verde 2
#axs[2].scatter(esperanza_verde2, ganancia_verde2, s=1, color='darkgreen', alpha=0.5, label='Verde 2')
#axs[2].hist2d(esperanza_verde2, 1/ganancia_verde2, color='darkgreen', bins=500, cmap='Greens',vmin=0, vmax=5000, label='Verde 2')
axs[2].errorbar(
    agrupados_ganancia_verde2["esperanza_media"],  # Media de esperanza por bin
    agrupados_ganancia_verde2["ganancia_media"],  # Media de varianza por bin
    yerr=agrupados_ganancia_verde2["ganancia_std_normalizada"],  # Desviación estándar de varianza por bin
    fmt='x',
    color='darkblue',
    label='Error'
)
#axs[2].set_title('Ganancia Verde 2')
axs[2].grid(True)
axs[2].legend()
axs[2].set_ylabel('gain')

# Rojo
#axs[3].scatter(esperanza_rojo, ganancia_rojo, s=1, color='red',alpha=0.5, label='Rojo')
#axs[3].hist2d(esperanza_rojo, 1/ganancia_rojo, color='red',bins=500, cmap='Reds',vmin=0, vmax=5000, label='Rojo')
#axs[3].set_title('Ganancia Rojo')
axs[3].errorbar(
    agrupados_ganancia_rojo["esperanza_media"],  # Media de esperanza por bin
    agrupados_ganancia_rojo["ganancia_media"],  # Media de varianza por bin
    yerr=agrupados_ganancia_rojo["ganancia_std_normalizada"],  # Desviación estándar de varianza por bin
    fmt='s',
    color='red',
    label='Error'
)
axs[3].grid(True)
axs[3].legend()
axs[3].set_xlabel('Mean [ADU]')
axs[3].set_ylabel('gain')
# Ajustar el diseño
plt.tight_layout()
plt.show()
#plt.close()



plt.figure(figsize=(8, 6))
#plt.ylim(0,5)
plt.hist2d(esperanza_azul,1/ganancia_azul,bins=500 ,cmap='Blues',label='ganancia_azul')

plt.xlabel('Esperanza')
plt.ylabel('Ganancia')
plt.legend()
plt.grid(True)
plt.show()
plt.close()

"""
