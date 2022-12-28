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

relativePathFolder, outputSavePath, defaultStartIdx = getPathToImages(2)

images = []
images_g = []
for filename in os.listdir(relativePathFolder):
    img = cv2.imread(os.path.join(relativePathFolder, filename))
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img is not None:
        images.append(img)
        images_g.append(img_g)

img = images[1]
rows = img.shape[0]   
cols = img.shape[1]

M = np.float32([[1,0,100],[0,1,50]])
img = cv2.warpAffine(img,M,(cols+120,rows+250))

plt.figure(1)
plt.imshow(img[:,:,::-1])
plt.title('test ' + str(1))
plt.show()