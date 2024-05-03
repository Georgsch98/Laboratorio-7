from astropy.io import fits
import numpy as np
import os
import random

carpeta_salida = 'im_sintetizada'
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)
N=100
M=100
g=1
t=100
contador=0
for p in range (t):
    contador+=1
    print(contador)
    Matriz=np.zeros((N,M))
    sigma=100
    for i in range(N):
        for j in range(M):
            mu=i+j*N
            #mu=10
            k=np.random.poisson(mu)
            k_g=g*k
            k_g=np.random.normal(k_g,sigma)
            Matriz[i][j]=k_g
           
    matriz_im = os.path.join(carpeta_salida, 'matriz_im_' + str(p) + '.fits')
    print(matriz_im)
    fits.writeto(matriz_im, Matriz, overwrite=True)

    #print(Matriz)
