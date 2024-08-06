#%%
"""Exportamos lo que vamos a utilizar"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import SensorScreen, SlitScreen, Decoder, Options
from decoding_algorythms import decode_image
from image_preprocessing import process_image
import codedapertures as ca
from PIL import Image, ImageOps
#%%
if len(sys.argv) != 2:
    print("Usage: python process_image.py <filename>")
    sys.exit(1)

filename = sys.argv[1]
#%% 
"""Definimos estas variables que necesita el codigo de Eitan y Trinidad"""
sensor = SensorScreen([300, 300], [300, 300],readout_noise= 0.01, dark_current= 0.00001, exposure_time= 5)
slit = SlitScreen([100, 100], [100, 100], 'pinhole', 5, mura_config= {"rank": 6,"tile": 1,"center": True})
decoder = Decoder(True, 'fourier', fourier_config= {"threshold": 1e-5}) 
#%%
img = Image.open(filename)
img = img.resize((500, 500))
img_gray = img.convert('L') #Esto lo convierte en una escala de grises, donde solo importa la intensidad
img_array = np.array(img_gray)
sensor.screen = img_array

pattern, reconstruccion = decode_image(sensor, slit, decoder, 'fourier')  

#%%
f, ax = plt.subplots(ncols=2, figsize=(18,4))
f.subplots_adjust(wspace=0.4)
a = ax[0].imshow(sensor.screen)
plt.colorbar(a,ax=ax[0])
ax[0].set_title('Imagen codificada')
b = ax[1].imshow(reconstruccion)
plt.colorbar(b,ax=ax[1])
ax[1].set_title('Imagen decodificada por TyE')
#c=ax[2].imshow(slit)
#ax[2].set_title('Ranura')

plt.show()