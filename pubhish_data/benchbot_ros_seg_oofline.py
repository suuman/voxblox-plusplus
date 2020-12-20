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
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from sensor_msgs.msg import RegionOfInterest
from std_msgs.msg import Header
from std_msgs.msg import Float64
from std_msgs.msg import Int8
import tf2_ros as tf2
from mask_rcnn_ros.msg import Result
import segvisualize as visualize
import matplotlib.pyplot as plt

CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle',
                           'airplane',
                           'bus',
                           'train',
                           'truck',
                           'boat',
                           'traffic light',
                           'fire hydrant',
                           'stop sign',
                           'parking meter',
                           'bench',
                           'bird',
                           'cat',
                           'dog',
                           'horse',
                           'sheep',
                           'cow',
                           'elephant',
                           'bear',
                           'zebra',
                           'giraffe',
                           'backpack',
                           'umbrella',
                           'handbag',
                           'tie',
                           'suitcase',
                           'frisbee',
                           'skis',
                           'snowboard',
                           'sports ball',
                           'kite',
                           'baseball bat',
                           'baseball glove',
                           'skateboard',
                           'surfboard',
                           'tennis racket',
                           'bottle',
                           'wine glass',
                           'cup',
                           'fork',
                           'knife',
                           'spoon',
                           'bowl',
                           'banana',
                           'apple',
                           'sandwich',
                           'orange',
                           'broccoli',
                           'carrot',
                           'hot dog',
                           'pizza',
                           'donut',
                           'cake',
                           'chair',
                           'couch',
                           'potted plant',
                           'bed',
                           'dining table',
                           'toilet',
                           'tv',
                           'laptop',
                           'mouse',
                           'remote',
                           'keyboard',
                           'cell phone',
                           'microwave',
                           'oven',
                           'toaster',
                           'sink',
                           'refrigerator',
                           'book',
                           'clock',
                           'vase',
                           'scissors',
                           'teddy bear',
                           'hair drier',
                           'toothbrush']

class BenchBotRos:

    def __init__(self):
        rospy.init_node('benchbot_node', anonymous=True)
        self.frame_id = "/benchbot_camera_frame"
        
        self.rgb_img_pub = rospy.Publisher("/camera/rgb/image_raw", Image, queue_size = 5)
        self.depth_img_pub = rospy.Publisher("/camera/depth/image_raw", Image, queue_size = 5)
        
        self.rgb_info = rospy.Publisher("/camera/rgb/camera_info", CameraInfo, queue_size=5)
        self.dep_info = rospy.Publisher("/camera/depth/camera_info", CameraInfo, queue_size=5)

        self.ts_info = rospy.Publisher("/ts", Float64, queue_size=5)
        self.end_info = rospy.Publisher("/genObjMap", Int8, queue_size=5)
        self.end_info.publish(0)

        self._class_names = rospy.get_param('~class_names', CLASS_NAMES)
        self._visualization = rospy.get_param('~visualization', True)
        self._last_msg = None

        self._class_colors = visualize.random_colors(len(CLASS_NAMES))
        
        self.br = tf2.TransformBroadcaster()

        self.camera_info = self.get_camera_info()

        self.header = Header(frame_id=self.frame_id)
        self.cvbridge = CvBridge()

        self._result_pub = rospy.Publisher('/mask_rcnn/result', Result, queue_size=1)
        self.vis_pub = rospy.Publisher('/mask_rcnn/visualization', Image, queue_size=1)

        self.frame = -1


    def get_camera_info(self):
        fx = 480
        fy = 480
        cx = 480
        cy = 270
        iw = 960
        ih = 540

        camera_info = CameraInfo()
        camera_info.height = ih
        camera_info.width = iw

        camera_info.distortion_model = "plumb_bob"
        camera_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        camera_info.R = [1.0, 0.0, 0.0 ,0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1]
        camera_info.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        return camera_info


    def getTransformMsg(self, curr_pose, timestamp):
        

        tx, ty, tz = curr_pose['translation']
        qx, qy, qz, qw = curr_pose['rotation']


        trans = TransformStamped()
        trans.header.stamp = timestamp
        trans.header.frame_id = 'world'
        trans.child_frame_id = self.frame_id
        trans.transform.translation.x = tx
        trans.transform.translation.y = ty
        trans.transform.translation.z = tz
        trans.transform.rotation.x = qx
        trans.transform.rotation.y = qy
        trans.transform.rotation.z = qz
        trans.transform.rotation.w = qw

        return trans

    
    def _build_result_msg(self, msg, result):
        result_msg = Result()
        result_msg.header = msg.header
        for i, (y1, x1, y2, x2) in enumerate(result['rois']):
            box = RegionOfInterest()
            box.x_offset = x1#np.asscalar(x1)
            box.y_offset = y1#np.asscalar(y1)
            box.height = y2-y1#np.asscalar(y2 - y1)
            box.width = x2-x1#np.asscalar(x2 - x1)
            result_msg.boxes.append(box)

            class_id = result['class_ids'][i]
            result_msg.class_ids.append(class_id)

            class_name = self._class_names[class_id]
            result_msg.class_names.append(class_name)

            score = result['scores'][i]
            result_msg.scores.append(score)

            mask = Image()
            mask.header = msg.header
            mask.height = result['masks'].shape[0]
            mask.width = result['masks'].shape[1]
            mask.encoding = "mono8"
            mask.is_bigendian = False
            mask.step = mask.width
            mask.data = (result['masks'][:, :, i] * 255).tobytes()
            result_msg.masks.append(mask)

        return result_msg

    
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
        


    def publish(self, bgr_image, depth_image, curr_pose,  result, n):

        ts = n / np.power(10.0, 6.0)

        timestamp = rospy.Time.from_sec(ts)
    
        self.frame = self.frame + 1
        
        self.header.stamp = timestamp
 
        self.header.seq = self.frame
  
        
        # traj
        transformstamp =  self.getTransformMsg(curr_pose, timestamp)

        # Write the RGBD data.
        bgr_msg = self.cvbridge.cv2_to_imgmsg(bgr_image, "8UC3")
        bgr_msg.encoding = "bgr8"
        bgr_msg.header = self.header
        self.rgb_img_pub.publish(bgr_msg)


        depth_msg = self.cvbridge.cv2_to_imgmsg(depth_image, "32FC1")
        depth_msg.header = self.header
        self.depth_img_pub.publish(depth_msg)


        self.camera_info.header = self.header
        self.rgb_info.publish(self.camera_info)
        self.dep_info.publish(self.camera_info)

        self.br.sendTransform(transformstamp)

        self.ts_info.publish(ts)
        
        result_msg = self._build_result_msg(bgr_msg, result)
        self._result_pub.publish(result_msg)


        if self._visualization:
            cv_result = self._visualize_plt(result, bgr_image)
            image_msg = self.cvbridge.cv2_to_imgmsg(cv_result, 'bgr8')
            self.vis_pub.publish(image_msg)

       
    
    def publishEndflag(self):
        self.end_info.publish(1)
    

    def spin(self):
        rospy.spin()
    
    def getMasks(self, rois, mask_im):
        ndets = len(rois)
        if(ndets==6):
            if(rois.ndim==1):
                ndets = 1
                rois = np.array([rois])
                #print(rois)
        
        print("ndets = ", ndets)
            
 
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

          


            x1 = int(rois[i,0])
            y1 = int(rois[i,1])
            x2 = int(rois[i,0]+rois[i,2])
            y2 = int(rois[i,1]+rois[i,3])
            if(y2>=540):
                y2 = 539
            if(x2>=960):
                x2 = 959
            if(x1<1):
                x1=0
            if(y1<1):
                y1 = 0

            
            
            bb2D[i,0] = y1
            bb2D[i,1] = x1
            bb2D[i,2] = y2
            bb2D[i,3] = x2
            #print(bb2D)



            #print(classname, CLASS_NAMES.index(classname),  y1, x1, y2, x2 )

            idx= rois[i, 5]
            classids[i] = idx
            scores[i] = rois[i,4]

            for y in range(y1,y2+1):
                for x in range(x1, x2+1):
                    cid = mask_im[y,x]
                    if(cid == i+1):
                        masks[y,x,i] = 1
            
            
       
        result['class_ids'] = classids
        result['scores'] = scores
        result['masks'] = masks
        result['rois'] = bb2D

        #print(result)
        return result

         

if __name__ == '__main__':


    fold = '/media/suman/data/benchbot_ws/dataset_pGT/office_3'

    rgb_ims = [f for f in sorted(glob.glob(fold + "/image/*.png"))]
    depth_ims = [f for f in sorted(glob.glob(fold + "/depth/*.tiff"))]
    gt_poses = [f for f in sorted(glob.glob(fold + "/poses_est/*.json"))]
    mask_ims = [f for f in sorted(glob.glob(fold + "/mask_est/*.png"))]
    rois_2ds = [f for f in sorted(glob.glob(fold + "/mask_est/*.txt"))]
    fac = 1
    
    nimgs = len(rgb_ims)
    nimgss = nimgs/fac


    BB = BenchBotRos()
    print(BB._class_names)

    rate = rospy.Rate(0.5)
    imcount = 0

    for i in range(0, nimgs,fac):

        print(rgb_ims[i])
        print(depth_ims[i])
        print(gt_poses[i])
        print(rois_2ds[i])
        print(mask_ims[i])

        bgr_image = cv2.imread(rgb_ims[i], cv2.IMREAD_UNCHANGED)
        depth_image = cv2.imread(depth_ims[i], cv2.IMREAD_UNCHANGED)
        depth_image = depth_image.astype(np.float32)
        imcount = imcount + 1

        with open(gt_poses[i]) as posefile:
            curr_pose = json.load(posefile)[0] 
        
        with open(rois_2ds[i]) as roisfile:
            rois = np.loadtxt(roisfile)
            #print(rois.ndim)
        
        maskim = cv2.imread(mask_ims[i], cv2.IMREAD_UNCHANGED)
        
        result = BB.getMasks(rois, maskim)
        
        BB.publish(bgr_image, depth_image, curr_pose, result, imcount)

        print(i+1)

        if(i==0):
            rate.sleep()
            rate.sleep()
            
        elif(imcount==nimgss-1):
            BB.publishEndflag()
        else:
            rate.sleep()
            
    BB.publishEndflag()
    print("Generating Map....")
    rate.sleep()
    BB.publishEndflag()
 
    BB.spin()
