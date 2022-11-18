import matplotlib.pyplot as plt
# from matplotlib import image as mpimg
import cv2
import os
import math
import numpy as np
# from scipy.ndimage.morphology import distance_transform_edt

from functionsFile import *
from getMosaice import getMosaice

showImageSubplot = False
useGrayImages = False
showIntermediateSteps = False
saveOutput = True
showMatches = False

# DATA SET IDX
# 0 arena
# 1 big_house
# 2 bridge
# 3 building_site
# 4 carmel
# 5 diamondhead
# 6 fishbowl
# 7 golden_gate
# 8 halfdome
# 9 hotel
# 10 office
# 11 ponte_nuovo
# 12 rio
# 13 river
# 14 roof
# 15 san_pietro
# 16 shangai
# 17 yard
# 18 lab
# 19 cavignal
relativePathFolder, outputSavePath, defaultStartIdx,  checksOnDeterminant, determinantCheckLowerBound, determinantCheckUpperBound = getPathToImages(20)

images = []
for filename in os.listdir(relativePathFolder):
    img = cv2.imread(os.path.join(relativePathFolder, filename))
    if img is not None:
        images.append(img)

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

    mosaice, imagesToUseIdx = getMosaice(images, startImgIdx, useGrayImages, checksOnDeterminant, determinantCheckLowerBound, determinantCheckUpperBound, showIntermediateSteps, imagesToUseIdx)

    mosaiceCollection.append(mosaice)

for i in range(len(mosaiceCollection)):
    plt.figure(i)
    plt.imshow(mosaiceCollection[i][:,:,::-1])
    plt.title('Mosaice number ' + str(i))
    if saveOutput:
        cv2.imwrite(f"{outputSavePath}final_result_{i}.jpg", mosaiceCollection[i])
plt.show()