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
               'tv', 'microwave', 'toaster', 'refrigerator', 'oven', 'sink', 'person']



    

class BenchBotRos:
    
    def __init__(self):
        self.done = 0
        self._class_colors = visualize.random_colors(len(CLASS_NAMES))

    
    def _visualize(self, result, image):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        axes = fig.gca()
        visualize.display_instances(image, result['rois'], result['masks'],
                                    result['class_ids'], CLASS_NAMES,
                                    result['scores'], ax=axes,
                                    class_colors=self._class_colors)
        fig.tight_layout()
        canvas.draw()
        result = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

        _, _, w, h = fig.bbox.bounds
        result = result.reshape((int(h), int(w), 3))
        return result
    

    def _get_fig_ax(self):
        """Return a Matplotlib Axes array to be used in
        all visualizations. Provide a
        central point to control graph sizes.

        Change the default size attribute to control the size
        of rendered images
        """
        fig, ax = plt.subplots(1)
        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        return fig, ax

    def _visualize_plt(self, result, image):
        fig, ax = self._get_fig_ax()
        image = visualize.display_instances_plt(
            image,
            result['rois'],
            result['masks'],
            result['class_ids'],
            CLASS_NAMES,
            result['scores'],
            fig=fig,
            ax=ax)

        return image
        


       

    


    
    def getMasks(self, mask_im):
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

        

        for i in range(ndets):
            instid = instids[i+1]
            idx = int(str(instid)[:-3])
           
            classname = CLASS_NAMES[idx]
            mask = mask_im == instid
            points = cv2.findNonZero(mask.astype(np.uint8))

            x1,y1,w,h = cv2.boundingRect(points)
            y2 = y1 + h 
            x2 = x1 + w 
           
            bb2D[i,0] = y1
            bb2D[i,1] = x1
            bb2D[i,2] = y2
            bb2D[i,3] = x2

            print(classname, idx,  y1, x1, w,h )

            classids[i] = idx
            scores[i] = 1.0

            for y in range(y1,y2):
                for x in range(x1, x2):
                    cid = mask_im[y,x]
                    if(cid == instid):
                        masks[y,x,i] = 1
            
            
       
        result['class_ids'] = classids
        result['scores'] = scores
        result['masks'] = masks
        result['rois'] = bb2D

        return result

         

if __name__ == '__main__':


    fold = '/media/suman/data/benchbot_ws/dataset/house_1'


    rgb_ims = [f for f in sorted(glob.glob(fold + "/image/*.png"))]
    mask_ims = [f for f in sorted(glob.glob(fold + "/instseg/*.png"))]
    fac = 1
    
    nimgs = len(rgb_ims)
    bb=BenchBotRos()



   

    for i in range(0, nimgs):

        print(rgb_ims[i])
        print(mask_ims[i])

        bgr_image = cv2.imread(rgb_ims[i], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_ims[i], cv2.IMREAD_UNCHANGED)
        


        
        result = bb.getMasks(mask)
        visim = bb._visualize(result,bgr_image)
        cv2.imshow("objs",visim )
        cv2.waitKey(500)
       
