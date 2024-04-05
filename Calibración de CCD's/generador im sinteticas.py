from astropy.io import fits
import numpy as np
import os
import random

carpeta_salida = 'im_sintetizada'
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)
N=3040
M=4056
g=1
t=2

for p in range (t):
    Matriz=np.zeros((N,M))

    for i in range(N):
        for j in range(M):
            mu=i+j*N
            k=np.random.poisson(mu)
            k_g=g*k
            Matriz[i][j]=k_g
           
    matriz_im = 'matriz_im_'+str(p)+'.fits'
    print(matriz_im)
fits.writeto(matriz_im, Matriz, overwrite=True)
