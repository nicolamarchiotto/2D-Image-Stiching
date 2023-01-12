from src.loftr import LoFTR, default_cfg
import torch
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from functionsFile import *
import os
from anytree import Node, RenderTree, PreOrderIter, Resolver
from src.utils.plotting import make_matching_figure

# Detection method of features, 0 SIFT, 1 LOFTR NEURAL NETWORK
features_detection_method = 0

# Save intermediate steps, requires folders named sift and loftr in the image dataset folder
saveIntermediateSteps = True
saveDrawMatches = True

# Initialize LoFTR
matcher = LoFTR(config=default_cfg)
#matcher.load_state_dict(torch.load("weights/indoor_ds_new.ckpt")['state_dict'])
matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval()

# Initialize SIFT
sift = cv2.SIFT_create()

# get dataset path
relativePathFolder, outputSavePath, startImageIdx = getPathToImages(3)


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
        loftr_mconf = []

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
            
                        mkpts_target, mkpts_source, mconf = getLoftrSortedMatches(mkpts_target, mkpts_source, mconf, 0.7)
                        # delete matches with confidance less than 0.7


                        # valid correspondance if more than n matches are found between the two images
                        # print("correspondances ", len(mkpts_target))
                        if len(mkpts_target) > len(loftr_chosenSourceImg_matches):
                            loftr_chosenSourceImg_idx = i
                            loftr_chosenTargetImg_idx = targetIdx

                            loftr_chosenSourceImg_matches = mkpts_source
                            loftr_chosenTargetImg_matches = mkpts_target
                            loftr_mconf = mconf

        if features_detection_method == 0:
            matchesLen = len(sift_bestMatches)
            t_idx = sift_chosenTargetImg_idx
            s_idx = sift_chosenSourceImg_idx
        else:
            matchesLen = len(loftr_chosenSourceImg_matches)
            t_idx = loftr_chosenTargetImg_idx
            s_idx = loftr_chosenSourceImg_idx
            
        sift_draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        # matchesMask = matchesMask, # draw only inliers
                        flags = 2)
            
        if saveDrawMatches:
            if features_detection_method == 0:
              
                img_matches = cv2.drawMatches(images_col[t_idx],  sift_image_kp[t_idx], images_col[s_idx],  sift_image_kp[s_idx], sift_bestMatches, None,  **sift_draw_params)
                text ="SIFT\nAll matches: " + str(len(sift_bestMatches))+"\nTarget idx: "+str(t_idx)+" Source idx: "+str(s_idx) 
                txt_color = (0, 0, 0) #if img_matches[:100, :200].mean() > 200 else (255, 255, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.8
                org = (50, 50)
                thickness = 1
                y0, dy = 25, 30
                for i, line in enumerate(text.split('\n')):
                    y = y0 + i*dy
                    cv2.putText(img_matches, line, (10, y ), cv2.FONT_HERSHEY_SIMPLEX, fontScale, txt_color, thickness, cv2.LINE_AA)
                cv2.imwrite(f"{outputSavePath}/sift/matchesAll_{t_idx}_{s_idx}.jpg", img_matches)
            else:
                color = cm.jet(loftr_mconf)
                text = [
                    'LoFTR',
                    'All matches: {}'.format(len(loftr_chosenSourceImg_matches)),
                    'Target idx: '+str(t_idx)+' Source idx: '+str(s_idx),
                ]
                img_matches = make_matching_figure(images_col[t_idx][:,:,::-1], images_col[s_idx][:,:,::-1], loftr_chosenTargetImg_matches, loftr_chosenSourceImg_matches, color, text=text, path=f"{outputSavePath}/loftr/matchesAll_{t_idx}_{s_idx}.jpg")

        # discard correspondance if less than n matches are found, breaking search
        if matchesLen < 50:
            print("no images with enough correspondances, breaking iter")
            break
        else:
            print("Found correspondance from chosenSourceImg_idx", s_idx , " to chosenTargetImg_idx", t_idx)
            print("Number of matches ", matchesLen)

        # transformation matrix which move image to be stiched in target image
        # threshold by default equal to 3

        if features_detection_method == 0:
            h, status, target_kp_inliers, source_kp_inliers = getHMatrixSIFT(sift_bestMatches, sift_image_kp[t_idx], sift_image_kp[s_idx],  len(sift_bestMatches), 50)
            
            if saveDrawMatches:
                color = cm.jet(np.zeros(len(source_kp_inliers)))
                text = [
                    'SIFT',
                    'Inliers matches: {}'.format(len(target_kp_inliers)),
                    'Target idx: '+str(t_idx)+' Source idx: '+str(s_idx),

                ]
                img_matches_inliers = make_matching_figure(images_col[t_idx][:,:,::-1], images_col[s_idx][:,:,::-1], target_kp_inliers, source_kp_inliers, color, text=text, path=f"{outputSavePath}/sift/matchesInliers_{t_idx}_{s_idx}.jpg")
        else:
            h, status = cv2.findHomography(loftr_chosenSourceImg_matches, loftr_chosenTargetImg_matches, cv2.RANSAC, maxIters = 50)
            
            if saveDrawMatches:
                target_kp_inliers=[]
                source_kp_inliers=[]
                matchesMask = status.ravel().tolist()
                for i in range(len(matchesMask)):
                    if matchesMask[i]==1:
                        target_kp_inliers.append(loftr_chosenTargetImg_matches[i])
                        source_kp_inliers.append(loftr_chosenSourceImg_matches[i])
                        
                t_kp_inliers=np.array(target_kp_inliers)
                s_kp_inliers=np.array(source_kp_inliers)

                color = cm.jet(np.zeros(len(t_kp_inliers)))
                text = [
                        'LoFTR',
                        'Inliers matches: {}'.format(len(t_kp_inliers)),
                        'Target idx: '+str(t_idx)+' Source idx: '+str(s_idx),
                    ]
                
                img_matches_inliers = make_matching_figure(images_col[t_idx][:,:,::-1], images_col[s_idx][:,:,::-1], t_kp_inliers, s_kp_inliers, color, text=text, path=f"{outputSavePath}/loftr/matchesInliers_{t_idx}_{s_idx}.jpg")

        matchesMask = status.ravel().tolist()
        numberOfInliers = matchesMask.count(1)
        numberOfOutliers = matchesMask.count(0)

        print("RANSAC found: inliers ", numberOfInliers, " outliers ", numberOfOutliers)

        # image with only inliers matches


        imagesToBeStichedIdxArray.remove(s_idx)
            
        # attach node to the right one  
        for node in PreOrderIter(treeHead):
            if node.imageIdx == t_idx:
                nodeToAttachTo = node
                break
    
        nodeToLink = Node(name=str(s_idx), imageIdx=s_idx, H=h, parent=nodeToAttachTo, numOfMatches=matchesLen, ransacInliers=numberOfInliers, ransacOutliers=numberOfOutliers)

        print("\nprinting tree")
        for pre, fill, node in RenderTree(treeHead):
            print("%s%s" % (pre, node.name))    

        print("number of image to still to find target ", len(imagesToBeStichedIdxArray),"\n")
           
#
#
# BUILDING MOSAICE 
#
#

if features_detection_method == 0:
    text_file = open(outputSavePath+"/sift/correspondances_tree_sift.txt", "w", encoding='utf-8')
else:
    text_file = open(outputSavePath+"/loftr/correspondances_tree_loftr.txt", "w", encoding='utf-8')
for pre, fill, node in RenderTree(treeHead):
    text_file.write(str(pre) + str(node.name)+"\n")
text_file.close()

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

        # requires folders named sift and loftr in the outputSavePath
        if saveIntermediateSteps:
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
