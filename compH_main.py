import matplotlib.pyplot as plt
import cv2
import os
import math
import numpy as np

from functionsFile import *
from getMosaice import getMosaice
from compH_getMosaice import compH_getMosaice

showImageSubplot = False
useGrayImages = False
showIntermediateSteps = False
saveOutput = True
showMatches = False

relativePathFolder, outputSavePath, defaultStartIdx = getPathToImages(3)

images = []
images_g = []
for filename in os.listdir(relativePathFolder):
    img = cv2.imread(os.path.join(relativePathFolder, filename))
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img is not None:
        images.append(img)
        images_g.append(img_g)

print("Number of images found ", len(images))

if showImageSubplot:
    if len(images) < 5:
        fig, ax = plt.subplots(1, len(images), figsize=(15, 12))
        for j in range(len(images)):
            ax[j].imshow(images[j][:,:,::-1])           
    else:
        fig, ax = plt.subplots(math.floor(len(images)/5), 5, figsize=(15, 5))
        fig.suptitle('Some of the images')

        k = 0
        for i in range(math.floor(len(images)/5)):
            for j in range(5):
                ax[i,j].imshow(images[k][:,:,::-1])
                k+=1
    plt.show()

mosaiceCollection = []

imagesToUseIdx = list(range(len(images)))
defaultStartIdxUsed = False

while len(imagesToUseIdx) != 0:
    print("*\n*\nNEW MOSAICE COMPUTATION ...\n*\n*")
    if not defaultStartIdxUsed:
        defaultStartIdxUsed = True
        startImgIdx = defaultStartIdx
        imagesToUseIdx.remove(defaultStartIdx)
    else:
        startImgIdx = imagesToUseIdx[0]
        imagesToUseIdx.pop(0)

    mosaice, imagesToUseIdx = compH_getMosaice(images, images_g, startImgIdx, showIntermediateSteps, imagesToUseIdx, outputSavePath, showMatches)

    mosaiceCollection.append(mosaice)

for i in range(len(mosaiceCollection)):
    plt.figure(i)
    plt.imshow(mosaiceCollection[i][:,:,::-1])
    plt.title('Mosaice number ' + str(i))
    if saveOutput:
        cv2.imwrite(f"{outputSavePath}final_result_{i}.jpg", mosaiceCollection[i])
plt.show()