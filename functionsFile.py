import matplotlib.pyplot as plt
import cv2
import numpy as np 
from scipy.ndimage.morphology import distance_transform_edt

# FUNCTIONS

def getPathToImages(datasetIndex):
    if datasetIndex ==0:
        relativePathFolder = 'images/arena/source_images/'
        outputSavePath = 'images/arena/'
        startImgIndex = 4
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3
    
    elif datasetIndex == 1:
        relativePathFolder = 'images/big_house/source_images/'
        outputSavePath = 'images/big_house/'
        startImgIndex = 1
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3
        
    elif datasetIndex == 2:
        relativePathFolder = 'images/bridge/source_images/'
        outputSavePath = 'images/bridge/'
        startImgIndex = 0
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3
    
    elif datasetIndex == 3:
        relativePathFolder = 'images/building_site/source_images/'
        outputSavePath = 'images/building_site/'
        startImgIndex = 0
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3
        
    elif datasetIndex == 4:
        relativePathFolder = 'images/carmel/source_images/'
        outputSavePath = 'images/carmel/'
        startImgIndex = 9
        determinantCheck = True
        determinantLowerBound = 0.5
        determinantUpperBound = 20
        
    elif datasetIndex == 5:
        relativePathFolder = 'images/diamondhead/source_images/'
        outputSavePath = 'images/diamondhead/'
        startImgIndex = 10
        determinantCheck = True
        determinantLowerBound = 0.5
        determinantUpperBound = 50

    elif datasetIndex == 6:
        relativePathFolder = 'images/fishbowl/source_images/'
        outputSavePath = 'images/fishbowl/'
        startImgIndex = 0
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3

    elif datasetIndex == 7:
        relativePathFolder = 'images/golden_gate/source_images/'
        outputSavePath = 'images/golden_gate/'
        startImgIndex = 3
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3

    elif datasetIndex == 8:
        relativePathFolder = 'images/halfdome/source_images/'
        outputSavePath = 'images/halfdome/'
        startImgIndex = 0
        determinantCheck = True
        determinantLowerBound = 0.5
        determinantUpperBound = 3

    elif datasetIndex == 9:
        relativePathFolder = 'images/hotel/source_images/'
        outputSavePath = 'images/hotel/'
        startImgIndex = 4
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3

    elif datasetIndex == 10:
        relativePathFolder = 'images/office/source_images/'
        outputSavePath = 'images/office/'
        startImgIndex = 0
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3

    elif datasetIndex == 11:
        relativePathFolder = 'images/ponte_nuovo/source_images/'
        outputSavePath = 'images/ponte_nuovo/'
        startImgIndex = 0
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3

    elif datasetIndex == 12:
        relativePathFolder = 'images/rio/source_images/'
        outputSavePath = 'images/rio/'
        startImgIndex = 25
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3

    elif datasetIndex == 13:
        relativePathFolder = 'images/river/source_images/'
        outputSavePath = 'images/river/'
        startImgIndex = 0
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3

    elif datasetIndex == 14:
        relativePathFolder = 'images/roof/source_images/'
        outputSavePath = 'images/roof/'
        startImgIndex = 0
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3

    elif datasetIndex == 15:
        relativePathFolder = 'images/san_pietro/source_images/'
        outputSavePath = 'images/san_pietro/'
        startImgIndex = 0
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3

    elif datasetIndex == 16:
        relativePathFolder = 'images/shangai/source_images/'
        outputSavePath = 'images/shangai/'
        startImgIndex = 7
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3

    elif datasetIndex == 17:
        relativePathFolder = 'images/yard/source_images/'
        outputSavePath = 'images/yard/'
        startImgIndex = 0
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3

    elif datasetIndex == 18:
        relativePathFolder = 'images/lab/source_images/'
        outputSavePath = 'images/lab/'
        startImgIndex = 7
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3

    elif datasetIndex == 19:
        relativePathFolder = 'images/cavignal/source_images/'
        outputSavePath = 'images/cavignal/'
        startImgIndex = 7
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3
        
    elif datasetIndex == 20:
        relativePathFolder = 'images/hot_air_baloon/source_images/'
        outputSavePath = 'images/hot_air_baloon/'
        startImgIndex = 12
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3
    
    elif datasetIndex == 21:
        relativePathFolder = 'images/city/source_images/'
        outputSavePath = 'images/city/'
        startImgIndex = 10
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3
    
    else:
        relativePathFolder = 'wrongCaseIdx'
        outputSavePath = 'wrongCaseIdx'
        startImgIndex = -1
        determinantCheck = False
        determinantLowerBound = 0.5
        determinantUpperBound = 3
    
    
    return (relativePathFolder, outputSavePath, startImgIndex,  determinantCheck, determinantLowerBound, determinantUpperBound)

# Removes black borders from image 
def trim_black_countour(image): 
    mask = np.argwhere(image != 0)

    min_x = np.min(mask[:,1])
    max_x = np.max(mask[:,1])

    min_y = np.min(mask[:,0])
    max_y = np.max(mask[:,0])

    return image[min_y:max_y, min_x:max_x,:]

# Compose the mosaice from 2 images using color adjust 
def blending(I1, I2, blendingOn):

    '''
    WEIGHTENED BLENDING:
    I_blended = (I1 * w1 + I2 * w2)/(w1 + w2)
    I1 = ref_img, I2 = warped_img
    distance_transform_edt gives more weight to the px in the centre of the image and less on the borders.
    Link: https://www.youtube.com/watch?v=D9rAOAL12SY
    '''

    # STITCHING THE TWO IMAGES
    if blendingOn:
        w1 = distance_transform_edt(I1) # This command correspond to the bwdist() in MATLAB
        w1 = np.divide(w1, np.max(w1))
        w2 = distance_transform_edt(I2)
        w2 = np.divide(w2, np.max(w2))
        I_blended = cv2.add(np.multiply(I1, w1), np.multiply(I2, w2))
        w_tot = w1 + w2
        I_blended = np.divide(I_blended, w_tot, out=np.zeros_like(I_blended), where=w_tot != 0).astype("uint8")
    else:
        I_blended = cv2.add(I1,I2)        
    return I_blended

def getMatches(descr_1, descr_2):
    # FlannBasedMatcher
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    return matcher.knnMatch(descr_1, descr_2, 2)

def get_distance(e):
    return e.distance
 
def getSortedMatches(matches, percentageDistanceThesh):
        # percentageDistanceThesh must be between 0 and 1
        sortedMatches = []
        for m, n in matches:
            if m.distance < percentageDistanceThesh * n.distance:
                sortedMatches.append(m)
        
        list.sort(sortedMatches, key = get_distance)

        return sortedMatches
                
def getHMatrix(matches, kp_base, kp_toWarp, numberOfPoints, iterations):
    
    baseImage_idx = matches[0].queryIdx
    toWarpImage_idx = matches[0].trainIdx
    pts_base = np.array([[kp_base[baseImage_idx].pt[0], kp_base[baseImage_idx].pt[1]]])
    pts_toWarp = np.array([[kp_toWarp[toWarpImage_idx].pt[0], kp_toWarp[toWarpImage_idx].pt[1]]])

    for x in range(numberOfPoints-1):
        baseImage_idx = matches[x].queryIdx
        toWarpImage_idx = matches[x].trainIdx

        pts_base = np.concatenate((pts_base, [[kp_base[baseImage_idx].pt[0], kp_base[baseImage_idx].pt[1]]]))
        pts_toWarp = np.concatenate((pts_toWarp, [[kp_toWarp[toWarpImage_idx].pt[0], kp_toWarp[toWarpImage_idx].pt[1]]]))

    h, status = cv2.findHomography(pts_toWarp, pts_base, cv2.RANSAC, maxIters = iterations)

    return h

def drawMatches(img1, keypoints1, img2, keypoints2, good_matches, saveImage, outputSavePath, iteration):
    #-- Draw matches
    
    #img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    #cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches[0:10], img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #-- Show detected matches
    # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
               singlePointColor = None,
               # matchesMask = matchesMask, # draw only inliers
               flags = 2)

    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(img_matches[:,:,::-1])
    plt.title('Good Matches')
    plt.show()
    if saveImage:
        cv2.imwrite(f"{outputSavePath}matches_{iteration}.jpg", img_matches)

def extractFeatures(images, sift, useGrayImages):
    # EXTRACTING FEATURES
    #Get gray images
    images_g = []
    for i in range(len(images)):
        images_g.append(cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY))

    # compute keypoints and descriptors of all images except baseImage
    keypoints = []
    descriptors = []
    for i in range(len(images)):
        if useGrayImages:
            kp, descr = sift.detectAndCompute(images_g[i], None)
        else:
            kp, descr = sift.detectAndCompute(images[i], None)
            
        keypoints.append(kp)
        descriptors.append(descr)

    return keypoints, descriptors