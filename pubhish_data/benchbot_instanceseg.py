#publishes images, poses and camera info for voxblox++

#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import rospy
import cv2
import glob
import json
np.set_printoptions(threshold=sys.maxsize)

import segvisualize as visualize
import matplotlib.pyplot as plt

CLASS_NAMES = ['background', 'bottle', 'cup', 'knife', 'bowl', 'wine glass', 'fork', 
               'spoon', 'banana', 'apple', 'orange', 'cake', 'potted plant', 'mouse', 'keyboard', 
               'laptop', 'cell phone', 'book', 'clock', 'chair', 'table', 'couch', 'bed', 'toilet', 
               'tv', 'microwave', 'toaster', 'refrigerator', 'oven', 'sink', 'person', 'moniter', 'sofa']



    

class BenchBotRos:
    
    def __init__(self):
        self.done = 0
        self._class_colors = self.color_map(31)


    def color_map(self, N=256):
        """
        Return Color Map in PASCAL VOC format (rgb)
        \param N (int) number of classes
        \param normalized (bool) whether colors are normalized (float 0-1)
        \return (Nx3 numpy array) a color map
        """
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        cmap = np.zeros((N, 3), dtype=int)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3
            cmap[i] = np.array([r, g, b])
        return cmap    


    
    def getMasks(self, mask_im,bgr_image):
        instids = np.unique(mask_im.flatten())
        print(instids)
        ndets = len(instids)-1
        h = 540
        w = 960
        idlist = []
        masks = np.zeros((h,w,ndets), dtype=np.uint8)
        bb2D = np.zeros((ndets,4))
        classids = np.zeros(ndets,dtype=np.uint8)
        scores =  np.zeros(ndets)

        result = {}
        result['rois'] =[]
        result['masks'] = []
        result['class_ids'] =[]
        result['scores'] = []

        overlaymask = np.zeros(bgr_image.shape,dtype=np.uint8)

        

        for i in range(ndets):
            instid = instids[i+1]
            idx = int(str(instid)[:-3])
           
            classname = CLASS_NAMES[idx]
            mask = mask_im == instid
            thresh = mask.astype(np.uint8)
            points = cv2.findNonZero(thresh)

            x1,y1,w,h = cv2.boundingRect(points)
            y2 = y1 + h 
            x2 = x1 + w 
           
            bb2D[i,0] = y1
            bb2D[i,1] = x1
            bb2D[i,2] = y2
            bb2D[i,3] = x2

            print(classname, idx,  y1, x1, w,h )

            colour = tuple(self._class_colors[idx])
            print(colour)

            bgr_image = cv2.rectangle(bgr_image, (x1,y1), (x2,y2), colour ,2)
            cv2.putText(bgr_image, classname, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

            classids[i] = idx
            scores[i] = 1.0

            for y in range(y1,y2):
                for x in range(x1, x2):
                    cid = mask_im[y,x]
                    if(cid == instid):
                        masks[y,x,i] = 1
                        overlaymask[y,x] = colour
            
            
       
        result['class_ids'] = classids
        result['scores'] = scores
        result['masks'] = masks
        result['rois'] = bb2D

        bgr_image = cv2.addWeighted(bgr_image,0.7,overlaymask,0.3,0.0)

        return bgr_image

         

if __name__ == '__main__':


    fold = '/media/suman/data/benchbot_ws/dataset/house_1'


    rgb_ims = [f for f in sorted(glob.glob(fold + "/image/*.png"))]
    mask_ims = [f for f in sorted(glob.glob(fold + "/instseg/*.png"))]
    class_ids = [f for f in sorted(glob.glob(fold + "/classid/*.json"))]
    fac = 1
    
    nimgs = len(rgb_ims)
    bb=BenchBotRos()



   

    for i in range(0, nimgs):

        print(rgb_ims[i])
        print(mask_ims[i])

        bgr_image = cv2.imread(rgb_ims[i], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_ims[i], cv2.IMREAD_UNCHANGED)
        with open(class_ids[i]) as classidfile:
            curr_idlist = json.load(classidfile)
            print(curr_idlist)
        for item in curr_idlist:
            print(item,curr_idlist[item])
        


        
        result = bb.getMasks(mask,bgr_image)
        
        cv2.imshow("objs",result )
        cv2.waitKey(500)
       
