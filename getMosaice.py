import matplotlib.pyplot as plt
import cv2
import numpy as np 

from functionsFile import *

def getMosaice( images, 
                startImgIndex,
                useGrayImages,
                showIntermediateSteps,
                imagesToUseIdx,
                outputSavePath,
                showMatches
                ):
    
    sift = cv2.SIFT_create()

    print("Extracting features of imgages...")
    keypoints, descriptors = extractFeatures(images, sift, useGrayImages)

    base_img = images[startImgIndex]

     # Augment base image with black countour, assuming images shape all equal to the one of startImage
    top = int(base_img.shape[0])
    bottom = top
    right = int(base_img.shape[1])
    left = right

    print("Number of iteration to do at max: ", len(imagesToUseIdx))

    for i in range(len(imagesToUseIdx)):
        print("\nIteration ", i)
        base_img = cv2.copyMakeBorder(base_img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
        base_img_g = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        
        if useGrayImages:
            base_kp, base_descr = sift.detectAndCompute(base_img_g, None)
        else:
            base_kp, base_descr = sift.detectAndCompute(base_img, None)
            
        bestMatchesImageIdx = 0
        bestMatches = []
        
        # CHOOSING THE BEST IMAGE TO STICH
        
        # The image to stich is the one with lower mean distance value for the first
        # number of matches found
        for j in imagesToUseIdx:
            matches = getMatches(base_descr,  descriptors[j])
            orderedMatches = getSortedMatches(matches, 0.7)
            
            if len(orderedMatches) > len(bestMatches):
                bestMatches = orderedMatches
                bestMatchesImageIdx = j

        bestMatchesKeyPoints = keypoints[bestMatchesImageIdx]    
    
        # HOMOGRAPHY COMPUTATION
        
        if len(bestMatches) < 80:
            print("Too few matches found: ", len(bestMatches)," Discarding image ", bestMatchesImageIdx)
            break
    
        print("Choosem image index ", bestMatchesImageIdx)
        print("Number of matches for H computation ", len(bestMatches))

        h = getHMatrix(bestMatches, base_kp, bestMatchesKeyPoints,  len(bestMatches), 130)
    
        # DETERMINANT CHECK
        # The determinant of a transformation matrix can be seen as a scaling factor
        # If det(h) is too big, for ex. greater than 10, the computation was probably wrong
        
        determinantH = np.abs(np.linalg.det(h))
        print("Determinant of H matrix ", determinantH)
        if determinantH < 0.2:
            print("H not greater than minimum bound")
            break

        print("Applying homography transformation ...")
        
        imagesToUseIdx.remove(bestMatchesImageIdx)

        # Warped image
        img_warped = cv2.warpPerspective(images[bestMatchesImageIdx], h, (base_img.shape[1], base_img.shape[0]))

        # VISUALIZATION

        before_mosaice = base_img

        after_mosaice = blending(base_img, img_warped, True)

        drawMatches(before_mosaice, base_kp, images[bestMatchesImageIdx], keypoints[bestMatchesImageIdx], bestMatches[0:10], showMatches, outputSavePath, i)
        
        if showIntermediateSteps:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img_warped[:,:,::-1])
            ax[0].axis('off') 
            ax[0].set_title("image warped - idx "+str(bestMatchesImageIdx))

            ax[1].imshow(after_mosaice[:,:,::-1])
            ax[1].axis('off')  
            ax[1].set_title("after stiching - iter " + str(i))

        base_img = trim_black_countour(after_mosaice)

    print("Image idx not used ", imagesToUseIdx)
    out_img = trim_black_countour(base_img)

    return out_img, imagesToUseIdx
    
