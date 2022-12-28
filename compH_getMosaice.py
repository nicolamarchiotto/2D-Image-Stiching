import matplotlib.pyplot as plt
import cv2
import numpy as np 

from functionsFile import *

def compH_getMosaice( images,
                images_g,
                startImgIndex,
                showIntermediateSteps,
                imagesToUseIdx,
                outputSavePath,
                showMatches
                ):
    
    sift = cv2.SIFT_create()

    print("Extracting features of imgages...")
    keypoints, descriptors = extractFeaturesSIFT(images, sift, True)

    H_to_start_image = np.eye(3)
    # base_img = images[startImgIndex]
    # base_img_g = images_g[startImgIndex]

     # Augment base image with black countour, assuming images shape all equal to the one of startImage
    # top = int(base_img.shape[0])
    # bottom = top
    # right = int(base_img.shape[1])
    # left = right

    print("Number of iteration to do at max: ", len(imagesToUseIdx))

    prevIterChosenImgIndex = 0
    
    for i in range(len(imagesToUseIdx)):
        if i == 0:
            destinationImgIdx = startImgIndex
        else:
            destinationImgIdx = prevIterChosenImgIndex
            
        # search for image to stich, the one with the greater number of correspondences


        print("\nIteration ", i)
        # base_img = cv2.copyMakeBorder(base_img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
        
        dest_img_g = images_g[destinationImgIdx]
        dest_kp, dest_descr = sift.detectAndCompute(dest_img_g, None)
            
        sourceImgIdx = 0
        bestMatches = []
        
        # CHOOSING THE BEST IMAGE TO STICH
        
        # The image to stich is the one with lower mean distance value for the first
        # number of matches found
        # 
        # The source image is the one to be stiched in the reference frame of the destination image
        # homography matrix moves source reference frame image to destination image reference frame
        
        for j in imagesToUseIdx:
            matches = getMatches(dest_descr,  descriptors[j])
            orderedMatches = getSortedMatches(matches, 0.7)
            
            if len(orderedMatches) > len(bestMatches):
                bestMatches = orderedMatches
                sourceImgIdx = j

        source_kp = keypoints[sourceImgIdx]    
    
        # HOMOGRAPHY COMPUTATION
        
        if len(bestMatches) < 80:
            print("Too few matches found: ", len(bestMatches)," Discarding image ", source_kp)
            break
    
        print("Choosem image index ", source_kp)
        print("Number of matches for H computation ", len(bestMatches))

        h = compH_getHMatrixSIFT(bestMatches, dest_kp, source_kp,  len(bestMatches), 130)
    
        # DETERMINANT CHECK
        # The determinant of a transformation matrix can be seen as a scaling factor
        # If det(h) is too big, for ex. greater than 10, the computation was probably wrong
        
        determinantH = np.abs(np.linalg.det(h))
        print("Determinant of H matrix ", determinantH)
        if determinantH < 0.2:
            print("H not greater than minimum bound")
            break

        print("Applying homography transformation ...")
        
        imagesToUseIdx.remove(sourceImgIdx)

        # Warped image
        H_to_start_image = H_to_start_image * h
        
        source_warped = cv2.warpPerspective(images[sourceImgIdx], H_to_start_image, (images[destinationImgIdx].shape[1], images[destinationImgIdx].shape[0]))

        # VISUALIZATION

        before_mosaice = dest_img_g

        after_mosaice = blending(base_img, source_warped, True)

        #drawMatches(before_mosaice, dest_kp, images[sourceImgIdx], keypoints[sourceImgIdx], bestMatches[0:10], showMatches, outputSavePath, i)
        
        if showIntermediateSteps:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img_warped[:,:,::-1])
            ax[0].axis('off') 
            ax[0].set_title("image warped - idx "+str(sourceImgIdx))

            ax[1].imshow(after_mosaice[:,:,::-1])
            ax[1].axis('off')  
            ax[1].set_title("after stiching - iter " + str(i))

        base_img = trim_black_countour(after_mosaice)
        base_img_g = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

    print("Image idx not used ", imagesToUseIdx)
    out_img = trim_black_countour(base_img)

    return out_img, imagesToUseIdx
    