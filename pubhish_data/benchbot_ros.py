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
from std_msgs.msg import Header
from std_msgs.msg import Float64
from std_msgs.msg import Int8
import tf2_ros as tf2

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
        
        self.br = tf2.TransformBroadcaster()

        self.camera_info = self.get_camera_info()

        self.header = Header(frame_id=self.frame_id)
        self.cvbridge = CvBridge()

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



    def publish(self, bgr_image, depth_image, curr_pose,  n):

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
       
    
    def publishEndflag(self):
        self.end_info.publish(1)
    

    def spin(self):
        rospy.spin()

         

if __name__ == '__main__':

    fold = '/media/suman/data/benchbot_ws/benchbot/dataset/miniroomt'

    rgb_ims = [f for f in sorted(glob.glob(fold + "/image/*.png"))]
    depth_ims = [f for f in sorted(glob.glob(fold + "/depth/*.tiff"))]
    gt_poses = [f for f in sorted(glob.glob(fold + "/poses/*.json"))]

    nimgs = len(rgb_ims)

    BB = BenchBotRos()

    rate = rospy.Rate(0.5)

    for i in range(nimgs):

        bgr_image = cv2.imread(rgb_ims[i], cv2.IMREAD_UNCHANGED)
        depth_image = cv2.imread(depth_ims[i], cv2.IMREAD_UNCHANGED)

        with open(gt_poses[i]) as posefile:
            curr_pose = json.load(posefile)[0] 
        
        BB.publish(bgr_image, depth_image, curr_pose, i+1)

        print(i+1)

        if(i==0):
            rate.sleep()
	    rate.sleep()
        elif(i==nimgs-1):
            BB.publishEndflag()
        else:
            rate.sleep()
            
    
    BB.spin()
