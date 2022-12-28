with torch.no_grad():
    # while there are still image to be stiched    
    while(len(imagesToBeStichedIdxArray) != 0):
        print("new while iter")
        # choosing target image 
        
        # first iteration
        if len(homographyMatrixStorage) == 0:
            targetImageIdx = startImageIdx
            imagesToBeStichedIdxArray.remove(targetImageIdx)
        else:
            for t in homographyMatrixStorage:
                possible_target = t[1]
                if possible_target not in alreadyUsedAsTargetIdxArray:
                    targetImageIdx = possible_target
                    break
          
        targetImageTensor = images_tensor[targetImageIdx]
        targetImageCol = images_col[targetImageIdx]
        alreadyUsedAsTargetIdxArray.append(targetImageIdx)

        # analyzing the remainig images with the selected target image
        imagesToBeStichedIdxArray_len = len(imagesToBeStichedIdxArray)
        for i in range(len(images_tensor)):
            if i in imagesToBeStichedIdxArray:
                batch = {'image0': targetImageTensor, 'image1': images_tensor[i]}

                matcher(batch)    
                mkpts_target = batch['mkpts0_f'].cpu().numpy()
                mkpts_source = batch['mkpts1_f'].cpu().numpy()
                mconf = batch['mconf'].cpu().numpy()
                
                # valid correspondance if more than n matches are found between the two images
                print("correspondances ", len(mkpts_target))
                if len(mkpts_target) > 300:
                    chosenImg_idx = i
                    print("Found correspondance from ", chosenImg_idx , " to ", targetImageIdx)
                    
                    # transformation matrix which move image to be stiched in target image
                    h, status = cv2.findHomography(mkpts_source, mkpts_target, cv2.RANSAC, maxIters = 50)
                    tupleToSave = (targetImageIdx, chosenImg_idx, h)
                    homographyMatrixStorage.append(tupleToSave)
                    imagesToBeStichedIdxArray.remove(chosenImg_idx)

                    if targetImageIdx != int(treeHead.name):
                        nodeToAttachTo = r.get(treeHead, str(targetImageIdx))
                    else:
                        nodeToAttachTo = treeHead
                    nodeToLink = Node(name=str(chosenImg_idx), imageIdx=chosenImg_idx, H=h, parent=nodeToAttachTo)
        
        if imagesToBeStichedIdxArray_len == len(imagesToBeStichedIdxArray):
            print("no images with enough correspondances in while iter, breaking iter")
            break