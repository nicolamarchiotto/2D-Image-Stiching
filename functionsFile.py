import matplotlib.pyplot as plt
import cv2
import numpy as np 
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib

def getPathToImages(datasetIndex):

    if datasetIndex == 0:
        relativePathFolder = 'images/andalo/source_images/'
        outputSavePath = 'images/andalo/'
        startImgIndex = 4

    elif datasetIndex == 1:
        relativePathFolder = 'images/arena/source_images/'
        outputSavePath = 'images/arena/'
        startImgIndex = 4
    
    elif datasetIndex == 2:
        relativePathFolder = 'images/big_house/source_images/'
        outputSavePath = 'images/big_house/'
        startImgIndex = 1
        
    elif datasetIndex == 3:
        relativePathFolder = 'images/bridge/source_images/'
        outputSavePath = 'images/bridge/'
        startImgIndex = 1
    
    elif datasetIndex == 4:
        relativePathFolder = 'images/building_site/source_images/'
        outputSavePath = 'images/building_site/'
        startImgIndex = 0
        
    elif datasetIndex == 5:
        relativePathFolder = 'images/diamondhead/source_images/'
        outputSavePath = 'images/diamondhead/'
        startImgIndex = 10

    elif datasetIndex == 6:
        relativePathFolder = 'images/fishbowl/source_images/'
        outputSavePath = 'images/fishbowl/'
        startImgIndex = 0

    elif datasetIndex == 7:
        relativePathFolder = 'images/golden_gate/source_images/'
        outputSavePath = 'images/golden_gate/'
        startImgIndex = 3

    elif datasetIndex == 8:
        relativePathFolder = 'images/halfdome/source_images/'
        outputSavePath = 'images/halfdome/'
        startImgIndex = 4

    elif datasetIndex == 9:
        relativePathFolder = 'images/hotel/source_images/'
        outputSavePath = 'images/hotel/'
        startImgIndex = 4
    
    elif datasetIndex == 10:
        relativePathFolder = 'images/laroda/source_images/'
        outputSavePath = 'images/laroda/'
        startImgIndex = 6

    elif datasetIndex == 11:
        relativePathFolder = 'images/office/source_images/'
        outputSavePath = 'images/office/'
        startImgIndex = 0

    elif datasetIndex == 12:
        relativePathFolder = 'images/ponte_nuovo/source_images/'
        outputSavePath = 'images/ponte_nuovo/'
        startImgIndex = 0

    elif datasetIndex == 13:
        relativePathFolder = 'images/rio/source_images/'
        outputSavePath = 'images/rio/'
        startImgIndex = 25

    elif datasetIndex == 14:
        relativePathFolder = 'images/river/source_images/'
        outputSavePath = 'images/river/'
        startImgIndex = 0

    elif datasetIndex == 15:
        relativePathFolder = 'images/roof/source_images/'
        outputSavePath = 'images/roof/'
        startImgIndex = 0

    elif datasetIndex == 16:
        relativePathFolder = 'images/san_pietro/source_images/'
        outputSavePath = 'images/san_pietro/'
        startImgIndex = 0

    elif datasetIndex == 17:
        relativePathFolder = 'images/san_pietro_martire/source_images/'
        outputSavePath = 'images/san_pietro_martire/'
        startImgIndex = 0

    elif datasetIndex == 18:
        relativePathFolder = 'images/shangai/source_images/'
        outputSavePath = 'images/shangai/'
        startImgIndex = 7

    elif datasetIndex == 19:    
        relativePathFolder = 'images/torricelle/source_images/'
        outputSavePath = 'images/torricelle/'
        startImgIndex = 0
    
    elif datasetIndex == 20:
        relativePathFolder = 'images/yard/source_images/'
        outputSavePath = 'images/yard/'
        startImgIndex = 3
    
    else:
        relativePathFolder = 'wrongCaseIdx'
        outputSavePath = 'wrongCaseIdx'
        startImgIndex = -1
    
    return (relativePathFolder, outputSavePath, startImgIndex)

# Removes black borders from image 
def trim_black_countour(image): 
    mask = np.argwhere(image != 0)

    min_x = np.min(mask[:,1])
    max_x = np.max(mask[:,1])

    min_y = np.min(mask[:,0])
    max_y = np.max(mask[:,0])

    return image[min_y:max_y, min_x:max_x,:]

# Compose the mosaice from 2 images using color adjust 
def blending(I1, I2, blendingOn, outputSavePath):
    '''
    WEIGHTED BLENDING:
    I_blended = (I1 * w1 + I2 * w2)/(w1 + w2)
    I1 = ref_img, I2 = warped_img
    Link: https://www.youtube.com/watch?v=D9rAOAL12SY
    '''
   
    # STITCHING THE TWO IMAGES
    if blendingOn:
        gray1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(str(outputSavePath + "/loftr/gray_" + str(1) + ".jpg"), gray1)
        _,thresh1 = cv2.threshold(gray1, 0, 255, 0)
        # cv2.imwrite(str(outputSavePath + "/loftr/thresh_" + str(1) + ".jpg"), thresh1)
        cnts1, _ = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        gray2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(str(outputSavePath + "/loftr/gray_" + str(2) + ".jpg"), gray2)
        _,thresh2 = cv2.threshold(gray2, 0, 255, 0)
        # cv2.imwrite(str(outputSavePath + "/loftr/thresh_" + str(2) + ".jpg"), thresh2)
        cnts2,_ = cv2.findContours(thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        for c in cnts1:
            x1,y1,w1,h1 = cv2.boundingRect(c)

        for c in cnts2:
            x2,y2,w2,h2 = cv2.boundingRect(c)

        leftMargin = x1 if x1 < x2 else x2
        toptMargin = y1 if y1 < y2 else y2
        rightMargin = x1+w1 if x1+w1 > x2+w2 else x2+w2
        bottomMargin = y1+h1 if y1+h1 > y2+h2 else y2+h2


        addToLeft = leftMargin
        addToTop = toptMargin
        addToBottom = I1.shape[0]-bottomMargin
        addToRight = I1.shape[1]-rightMargin
        
        subImage1 = I1[toptMargin:bottomMargin,leftMargin:rightMargin]
        subImage2 = I2[toptMargin:bottomMargin,leftMargin:rightMargin]
        
        # cv2.imwrite(str(outputSavePath + "/loftr/subimage_" + str(1) + ".jpg"), subImage1)
        # cv2.imwrite(str(outputSavePath + "/loftr/subimage_" + str(2) + ".jpg"), subImage2)

        I1 = subImage1
        I2 = subImage2
        w1 = distance_transform_edt(I1) # This command correspond to the bwdist() in MATLAB
        w1 = np.divide(w1, np.max(w1))
        w2 = distance_transform_edt(I2)
        w2 = np.divide(w2, np.max(w2))
        I_blended = cv2.add(np.multiply(I1, w1), np.multiply(I2, w2))
        w_tot = w1 + w2
        I_blended = np.divide(I_blended, w_tot, out=np.zeros_like(I_blended), where=w_tot != 0).astype("uint8")
        I_blended = cv2.copyMakeBorder(I_blended, addToTop, addToBottom, addToLeft, addToRight, cv2.BORDER_CONSTANT, None, 0)
    else:
        I_blended = cv2.add(I1,I2)        
        
    return I_blended

def getSiftMatches(descr_1, descr_2):
    # FlannBasedMatcher
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    return matcher.knnMatch(descr_1, descr_2, 2)

def get_distance(e):
    return e.distance
 
def getSiftSortedMatches(matches, percentageDistanceThesh):
        # percentageDistanceThesh must be between 0 and 1
        sortedMatches = []
        for m, n in matches:
            if m.distance < percentageDistanceThesh * n.distance:
                sortedMatches.append(m)
        
        list.sort(sortedMatches, key = get_distance)

        return sortedMatches
                
def getHMatrixSIFT(matches, kp_base, kp_toWarp, numberOfPoints, iterations):
    
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
    # print("SIFT: h ", h )
    return h

def drawSiftMatches(img1, keypoints1, img2, keypoints2, matches, showMatches, outputSavePath, iteration):
    #-- Draw matches
    
    #img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    #cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches[0:10], img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #-- Show detected matches
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
               singlePointColor = None,
               # matchesMask = matchesMask, # draw only inliers
               flags = 2)

    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None,  **draw_params)

    if showMatches:
        plt.imshow(img_matches[:,:,::-1])
        plt.title('Good Matches')
        plt.show()
    cv2.imwrite(f"{outputSavePath}matches_{iteration}.jpg", img_matches)

def extractFeaturesSIFT(images, sift, useGrayImages):
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

def make_matching_figure_loftr(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig