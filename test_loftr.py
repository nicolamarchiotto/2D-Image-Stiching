from src.loftr import LoFTR, default_cfg
import torch
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from functionsFile import *
import os
from anytree import Node, RenderTree, PreOrderIter

# Initialize LoFTR
matcher = LoFTR(config=default_cfg)
#matcher.load_state_dict(torch.load("weights/indoor_ds_new.ckpt")['state_dict'])
matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval()

relativePathFolder, outputSavePath, startImageIdx = getPathToImages(2)


# Storing H in the following way:
# The notation follows the one of the homogenous transformation {j}_T_i
# j is the ref frame of arrival, i is the one in which j will be moved
# the tuples to store the informations are composed by (idx_of_image_of_arrival j, idx_of_img_to_warp i, {j}_H_i)  
# the tuples are store in the array homographyMatrixStorage

homographyMatrixStorage = []

images_col = []
images_g = []
images_tensor = []

for filename in os.listdir(relativePathFolder):
    img_col = cv2.imread(os.path.join(relativePathFolder, filename))
    
    if img_col is not None:
        img_col = cv2.resize(img_col, (640, 480))
        img_g = cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY)
        img_tensor =  torch.from_numpy(img_g)[None][None]/ 255.

        images_col.append(img_col)
        images_g.append(img_g)
        images_tensor.append(img_tensor)


# fig, ax = plt.subplots(2, 2, figsize=(15, 12))
# ax[0, 0].imshow(img0_col[:,:,::-1])
# ax[0, 1].imshow(img1_col[:,:,::-1])
# ax[1, 0].imshow(img0_g, cmap='gray')
# ax[1, 1].imshow(img1_g, cmap='gray')
# plt.savefig(str("".join([sourcePath, "/plot.jpg"])), bbox_inches='tight', pad_inches=0)
# print("after subplot")

imagesToBeStichedIdxArray = list(range(len(images_col)))
imagesToBeStichedIdxArray.remove(startImageIdx)

from anytree import Node, RenderTree, Resolver

r = Resolver('name')
treeHead = Node(name=str(startImageIdx), imageIdx = startImageIdx)

with torch.no_grad():
    # while there are still image to be stiched    
    while(len(imagesToBeStichedIdxArray) != 0):
        imagesToBeStichedIdxArray_len = len(imagesToBeStichedIdxArray)
        print("\n**********\n")
        print("new iteration")
        
        # choosing the best image to be stiched, the one with more corresondances
        # with a tree node

        chosenTargetImg_idx = 0
        chosenTargetImg_matches = []
        
        chosenSourceImg_idx = 0
        chosenSourceImg_matches = []

        for node in PreOrderIter(treeHead):
            targetIdx = node.imageIdx

            for i in range(len(images_tensor)):
                if i in imagesToBeStichedIdxArray:
                    batch = {'image0': images_tensor[targetIdx], 'image1': images_tensor[i]}

                    matcher(batch)    
                    mkpts_target = batch['mkpts0_f'].cpu().numpy()
                    mkpts_source = batch['mkpts1_f'].cpu().numpy()
                    mconf = batch['mconf'].cpu().numpy()
                    
                    # valid correspondance if more than n matches are found between the two images
                    # print("correspondances ", len(mkpts_target))
                    if len(mkpts_target) > len(chosenSourceImg_matches):
                        chosenSourceImg_idx = i
                        chosenTargetImg_idx = targetIdx

                        chosenSourceImg_matches = mkpts_source
                        chosenTargetImg_matches = mkpts_target

        if len(chosenSourceImg_matches) < 100:
            print("no images with enough correspondances, breaking iter")
            break
        else:
            print("Found correspondance from ", chosenSourceImg_idx , " to ", chosenTargetImg_idx)
            print("NUmber of matches ", len(chosenSourceImg_matches))

            # transformation matrix which move image to be stiched in target image
            h, status = cv2.findHomography(chosenSourceImg_matches, chosenTargetImg_matches, cv2.RANSAC, maxIters = 50)
            tupleToSave = (chosenTargetImg_idx, chosenSourceImg_idx, h)
            homographyMatrixStorage.append(tupleToSave)
            imagesToBeStichedIdxArray.remove(chosenSourceImg_idx)

            print("chosenSourceImg_idx ", chosenSourceImg_idx)
            print("chosenTargetImg_idx ", chosenTargetImg_idx)
            
            # attach node to the right one  
            for node in PreOrderIter(treeHead):
                if node.imageIdx == chosenTargetImg_idx:
                    nodeToAttachTo = node
                    break
        
            nodeToLink = Node(name=str(chosenSourceImg_idx), imageIdx=chosenSourceImg_idx, H=h, parent=nodeToAttachTo)

            print("printing tree")
            for pre, fill, node in RenderTree(treeHead):
                print("%s%s" % (pre, node.name))    

            print("number of iterations still to perform ", len(imagesToBeStichedIdxArray))
           

# printing tree
for pre, fill, node in RenderTree(treeHead):
    print("%s%s" % (pre, node.name))

# Building the mosaice as a compositio of homography matrices 

top_bottom_shift = int(len(images_col)*480)
left_right_shift = int(len(images_col)*640)
print("shifts ", top_bottom_shift, left_right_shift)
mosaice = images_col[startImageIdx]
base_img_augm = cv2.copyMakeBorder(mosaice, top_bottom_shift, top_bottom_shift, left_right_shift, left_right_shift, cv2.BORDER_CONSTANT, None, 0)


resultIdx = 0
for node in PreOrderIter(treeHead):

    print(node.name)
    if node.parent is not None:
        print("stiching ", node.name)
        sourceImg = images_col[node.imageIdx]
        h = node.H

        c = node
        # composing the H matrix to warp image to startImageIdx ref frame
        while c.parent.parent is not None:
            c = c.parent
            h = np.matmul(c.H, h)

        print("h ", h)
        changedSign = False
        # if h[0,2] < 0  or h[1,2]:
        #     h[0,1] = -h[0,1]
        #     h[0,2] = -h[0,2]
        #     changedSign = True


        img_warped = cv2.warpPerspective(sourceImg, h, (base_img_augm.shape[1], base_img_augm.shape[0]))
        cv2.imwrite(str(outputSavePath + "/loftr/img_warped_" + str(resultIdx) + ".jpg"), img_warped)
        
        if changedSign:
            M = np.float32([[1,0,left_right_shift - h[0,1]],[0,1,top_bottom_shift - h[0,2]]])
        else:
            M = np.float32([[1,0,left_right_shift],[0,1,top_bottom_shift]])
        img_shifted = cv2.warpAffine(img_warped, M,(base_img_augm.shape[1], base_img_augm.shape[0]))
        cv2.imwrite(str(outputSavePath + "/loftr/img_shifted_" + str(resultIdx) + ".jpg"), img_shifted)

        I_blended = blending(base_img_augm, img_shifted, True)

        cv2.imwrite(str(outputSavePath + "/loftr/img_blended_" + str(resultIdx) + ".jpg"), I_blended)
        base_img_augm = I_blended
        resultIdx+=1

trimmed = trim_black_countour(I_blended)
cv2.imwrite( str("".join([outputSavePath, "/loftr/final_result.jpg"])), trimmed)


# arena tree
# 4
# ├── 8
# │   ├── 0
# │   │   └── 2
# │   │       └── 5
# │   └── 6
# │       └── 7
# │           └── 1
# │               └── 11
# │                   └── 3
# └── 13
#     ├── 12
#     └── 9
#         ├── 10
#         └── 14