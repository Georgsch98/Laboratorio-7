#!/bin/bash

# Dirección del servidor remoto y usuario
REMOTE_USER="lambda"
REMOTE_HOST="192.120.6.20"
REMOTE_PASS="skipperccd"  # Contraseña para SSH

# Define los métodos de decodificación que quieres aplicar
decode_methods="decode_EyT decode_A"

# Función para capturar imagen en el servidor remoto
capture_image() {
  local g=$1
  local t=$2
  local i=$3
  local remote_filename="Nombre_${g}g_${t}s_${i}.jpg"
  
  # Captura la imagen en el servidor remoto y devuelve el nombre del archivo
  sshpass -p ${REMOTE_PASS} ssh ${REMOTE_USER}@${REMOTE_HOST} << EOF
libcamera-still -n -r -o ${remote_filename} --shutter ${t}000000 --gain $g --awbgains 1,1 --immediate
echo ${remote_filename}
EOF
}

# Función para descargar la imagen del servidor remoto
download_image() {
  local remote_filename=$1
  # Descarga la imagen del servidor remoto al directorio actual
  sshpass -p ${REMOTE_PASS} scp ${REMOTE_USER}@${REMOTE_HOST}:${remote_filename} .
}

# Función principal
main() {
  for g in 1; do
    for t in 50 30; do
      for (( i = 0; i < 1; i++ )); do  # Cambié a 1 para solo capturar una imagen como ejemplo
        echo ################## $i ###################     
        
        # Capturar imagen en el servidor remoto y obtener el nombre del archivo
        remote_filename=$(capture_image $g $t $i)
        
        # Descargar la imagen del servidor remoto
        download_image ${remote_filename}
        
        # Llamar al script de Python y pasarle el nombre del archivo y los métodos de decodificación
        python3 decodificador_GyJ.py ${remote_filename} $decode_methods

      done
    done
  done
}

# Ejecutar la función principal
main
