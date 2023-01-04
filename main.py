from src.loftr import LoFTR, default_cfg
import torch
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from functionsFile import *
import os
from anytree import Node, RenderTree, PreOrderIter, Resolver

# Detection method of features, 0 SIFT, 1 LOFTR NEURAL NETWORK

features_detection_method = 1

# Initialize LoFTR
matcher = LoFTR(config=default_cfg)
#matcher.load_state_dict(torch.load("weights/indoor_ds_new.ckpt")['state_dict'])
matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval()

# Initialize SIFT
sift = cv2.SIFT_create()

# get dataset path
relativePathFolder, outputSavePath, startImageIdx = getPathToImages(8)


images_col = []
images_g = []

# sift data structures
sift_image_kp = []
sift_image_descr = []

# loftr data structure
loftr_images_tensor = []

#
#
# READING IMAGES
#
#

for filename in os.listdir(relativePathFolder):
    img_col = cv2.imread(os.path.join(relativePathFolder, filename))
    
    if img_col is not None:
        img_col = cv2.resize(img_col, (640, 480))
        img_g = cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY)

        images_col.append(img_col)
        images_g.append(img_g)

        # computing tensors if loftr
        if features_detection_method == 1:
            img_tensor =  torch.from_numpy(img_g)[None][None]/ 255.
            loftr_images_tensor.append(img_tensor)


# computing keypoints and descriptors if sift
if features_detection_method == 0:  
    sift_image_kp, sift_image_descr = extractFeaturesSIFT(images_col, sift, False)

imagesToBeStichedIdxArray = list(range(len(images_col)))
imagesToBeStichedIdxArray.remove(startImageIdx)


# Storing H using a tree structure:
# The name of the node is the index of the image, the index represents the order in which the images were read
# Each node has 3 params, name, imageIdx which is the same as the name, and H
# The head as no parameter H, 
# The sons of a node have as H the homography transformation which move the son's ref frame into the parent ref frame
# The strucure allows easy homography contatenation computation

r = Resolver('name')
treeHead = Node(name=str(startImageIdx), imageIdx = startImageIdx)

#
#
# BUILDING CORRESPONDANCES TREE
#
#

with torch.no_grad():
    # while there are still image to be stiched    
    while(len(imagesToBeStichedIdxArray) != 0):
        imagesToBeStichedIdxArray_len = len(imagesToBeStichedIdxArray)
        print("\n**********\n")
        print("searching new correspondace\n")
        
        # choosing the best image to be stiched, the one with more corresondances
        # with a tree node

        loftr_chosenTargetImg_idx = 0
        loftr_chosenTargetImg_matches = []
        loftr_chosenSourceImg_idx = 0
        loftr_chosenSourceImg_matches = []


        sift_bestMatches = []
        sift_chosenTargetImg_idx = 0 
        sift_chosenSourceImg_idx = 0

        for node in PreOrderIter(treeHead):
            targetIdx = node.imageIdx

            for i in range(len(images_col)):
                if i in imagesToBeStichedIdxArray:
                    if features_detection_method == 0:
                        matches = getSiftMatches(sift_image_descr[targetIdx],  sift_image_descr[i])
                        orderedMatches = getSiftSortedMatches(matches, 0.7)
                        if len(orderedMatches) > len(sift_bestMatches):
                            sift_bestMatches = orderedMatches
                            sift_chosenSourceImg_idx = i
                            sift_chosenTargetImg_idx = targetIdx
                    else:
                        # loftr based matcher
                        batch = {'image0': loftr_images_tensor[targetIdx], 'image1': loftr_images_tensor[i]}

                        matcher(batch)    
                        mkpts_target = batch['mkpts0_f'].cpu().numpy()
                        mkpts_source = batch['mkpts1_f'].cpu().numpy()
                        mconf = batch['mconf'].cpu().numpy()
                        
                        # valid correspondance if more than n matches are found between the two images
                        # print("correspondances ", len(mkpts_target))
                        if len(mkpts_target) > len(loftr_chosenSourceImg_matches):
                            loftr_chosenSourceImg_idx = i
                            loftr_chosenTargetImg_idx = targetIdx

                            loftr_chosenSourceImg_matches = mkpts_source
                            loftr_chosenTargetImg_matches = mkpts_target

        if features_detection_method == 0:
            matchesLen = len(sift_bestMatches)
            t_idx = sift_chosenTargetImg_idx
            s_idx = sift_chosenSourceImg_idx
        else:
            matchesLen = len(loftr_chosenSourceImg_matches)
            t_idx = loftr_chosenTargetImg_idx
            s_idx = loftr_chosenSourceImg_idx
            
        # discard correspondance if less than n matches are found, breaking search
        if matchesLen < 50:
            print("no images with enough correspondances, breaking iter")
            break
        else:
            print("Found correspondance from chosenSourceImg_idx", s_idx , " to chosenTargetImg_idx", t_idx)
            print("Number of matches ", matchesLen)

        # transformation matrix which move image to be stiched in target image
        if features_detection_method == 0:
            h = getHMatrixSIFT(sift_bestMatches, sift_image_kp[t_idx], sift_image_kp[s_idx],  len(sift_bestMatches), 50)
        else:
            h, status = cv2.findHomography(loftr_chosenSourceImg_matches, loftr_chosenTargetImg_matches, cv2.RANSAC, maxIters = 50)
    
        imagesToBeStichedIdxArray.remove(s_idx)
            
        # attach node to the right one  
        for node in PreOrderIter(treeHead):
            if node.imageIdx == t_idx:
                nodeToAttachTo = node
                break
    
        nodeToLink = Node(name=str(s_idx), imageIdx=s_idx, H=h, parent=nodeToAttachTo)

        print("\nprinting tree")
        for pre, fill, node in RenderTree(treeHead):
            print("%s%s" % (pre, node.name))    

        print("number of image to still to find target ", len(imagesToBeStichedIdxArray),"\n")
           
#
#
# BUILDING MOSAICE 
#
#

print("Building the mosaice")

top_bottom_shift = int(len(images_col)*480)
left_right_shift = int(len(images_col)*640)

# to prevent base image too big and prevent the program to crash
if len(images_col) > 20:
    top_bottom_shift = int(top_bottom_shift/10)
    left_right_shift = int(left_right_shift/2)

mosaice = images_col[startImageIdx]
base_img_augm = cv2.copyMakeBorder(mosaice, top_bottom_shift, top_bottom_shift, left_right_shift, left_right_shift, cv2.BORDER_CONSTANT, None, 0)


resultIdx = 0
for node in PreOrderIter(treeHead):
    if node.parent is not None:
        print("stiching image idx", node.name)
        sourceImg = images_col[node.imageIdx]
        h = node.H

        c = node
        # composing the H matrix to warp image to startImageIdx ref frame
        while c.parent.parent is not None:
            c = c.parent
            h = np.matmul(c.H, h)

        
        # to prevent the warped image to be placed outside the image limits,
        # the additional translation to center the warped image in the center as the starting one,
        # is applied at priori
        H_t = np.float32([[1,0,left_right_shift],[0,1,top_bottom_shift],[0,0,1]])
        h = np.matmul(H_t, h)

        img_warped = cv2.warpPerspective(sourceImg, h, (base_img_augm.shape[1], base_img_augm.shape[0]))

        I_blended = blending(base_img_augm, img_warped, True, outputSavePath)

        if features_detection_method == 0:
            cv2.imwrite(str(outputSavePath + "/sift/img_warped_" + str(resultIdx) + ".jpg"), img_warped)
            cv2.imwrite(str(outputSavePath + "/sift/img_blended_" + str(resultIdx) + ".jpg"), I_blended)
        else:
            cv2.imwrite(str(outputSavePath + "/loftr/img_warped_" + str(resultIdx) + ".jpg"), img_warped)
            cv2.imwrite(str(outputSavePath + "/loftr/img_blended_" + str(resultIdx) + ".jpg"), I_blended)

        base_img_augm = I_blended
        resultIdx+=1

trimmed = trim_black_countour(base_img_augm)
if features_detection_method == 0:
    cv2.imwrite( str("".join([outputSavePath, "/final_result_sift.jpg"])), trimmed)
else:
    cv2.imwrite( str("".join([outputSavePath, "/final_result_loftr.jpg"])), trimmed)
