#!/bin/bash

for g in 1; do
  for t in 50 30; do

    for (( i = 0; i < 500; i++ )); do
      echo ################## $i ###################     
      filename=Nombre_${g}g_${t}s_${i}.jpg
      libcamera-still -n -r -o $filename --shutter ${t}000000 --gain $g --awbgains 1,1 --immediate
      echo
      echo

      # Llama al script de Python y pásale el nombre del archivo
      python3 decodificador_GyJ.py $filename

    done

  done
done
