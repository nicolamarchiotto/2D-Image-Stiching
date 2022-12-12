import matplotlib.pyplot as plt
import cv2
import os
import math
import numpy as np

from functionsFile import *

relativePathFolder, outputSavePath, defaultStartIdx = getPathToImages(18)

images = []
for filename in os.listdir(relativePathFolder):
    img = cv2.imread(os.path.join(relativePathFolder, filename))
    if img is not None:
        images.append(img)

if len(images) < 5:
    fig, ax = plt.subplots(1, len(images), figsize=(15, 12))
    for j in range(len(images)):
        ax[j].imshow(images[j][:,:,::-1])           
else:
    fig, ax = plt.subplots(math.floor(len(images)/5), 5, figsize=(15, 5))
    k = 0
    if math.floor(len(images)/5) == 1:
        for j in range(5):
            ax[j].imshow(images[j][:,:,::-1])           
    else:
        for i in range(math.floor(len(images)/5)):
            for j in range(5):
                ax[i,j].imshow(images[k][:,:,::-1])
                k+=1
                
plt.savefig(f"{outputSavePath}images.jpg")