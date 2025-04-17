#!/usr/bin/env python3

# Standard Python libraries
import os
import numpy as np
import cv2
import time

# ROS2 libraries
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge  # For converting ROS images to OpenCV format

# ROS messages
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from my_msgs.msg import Float32MultiArrayStamped # Customized message type

# For exact-time matching
from message_filters import Subscriber, TimeSynchronizer # Subscribe to multiple topics and synchronize their messages

# DeepFusion logic imports
from deepfusion.data_fusion import data_fusion  # Main fusion function
from deepfusion.coordinate_transformation import TransformationKitti  # Calibration handling
from deepfusion.config import Config  # Configuration loader


class RealTimeFusionNode(Node):
    """
    A ROS2 node for real-time sensor fusion of 2D and 3D detections.
    It subscribes to image, LiDAR, 2D & 3D detections, synchronizes them,
    applies fusion, and publishes the results.
    """

    def __init__(self, cfg):
        super().__init__('real_time_fusion_node')
        self.cfg = cfg
        self.cat_list = self.cfg.get('cat_list', ['Car', 'Pedestrian'])  # Load object categories
        self.get_logger().info(f"Configured categories: {self.cat_list}")

        self.bridge = CvBridge()  # For converting ROS image messages to OpenCV

        # Subscribers 
        self.img_sub = Subscriber(self, Image, '/camera/image_raw')
        self.pc_sub = Subscriber(self, PointCloud2, '/lidar/points')
        self.det2_car_sub = Subscriber(self, Float32MultiArrayStamped, '/detection_2d/car')
        self.det3_car_sub = Subscriber(self, Float32MultiArrayStamped, '/detection_3d/car')
        self.det2_ped_sub = Subscriber(self, Float32MultiArrayStamped, '/detection_2d/pedestrian')
        self.det3_ped_sub = Subscriber(self, Float32MultiArrayStamped, '/detection_3d/pedestrian')

        # Synchronize the 6 topics
        self.ts = TimeSynchronizer([
            self.img_sub,
            self.pc_sub,
            self.det2_car_sub,
            self.det3_car_sub,
            self.det2_ped_sub,
            self.det3_ped_sub
        ], queue_size=10)
        self.ts.registerCallback(self.sync_cb)

        # Calibration subscriber
        self.calib_str = None
        self.calib_kitti = None
        self.calib_sub = self.create_subscription(String, '/camera/calibration', self.calib_callback, 10)

        # Publishers 
        self.raw_detections_image_pub = self.create_publisher(Image, '/raw_detections_image', 10)
        self.pub_fused_3d = self.create_publisher(Float32MultiArrayStamped, '/fusion/fused_3d', 10)
        self.pub_only_3d = self.create_publisher(Float32MultiArrayStamped, '/fusion/only_3d', 10)
        self.pub_only_2d = self.create_publisher(Float32MultiArrayStamped, '/fusion/only_2d', 10)

        self.get_logger().info("RealTimeFusionNode: 6-topic TimeSync initialized")

    def calib_callback(self, msg: String):
        """Receives KITTI-style calibration string and parses it."""
        self.calib_str = msg.data
        self.get_logger().info("Received calibration data")
        try:
            self.calib_kitti = TransformationKitti(self.calib_str, is_string=True)
            self.get_logger().info("Calibration parsed successfully.")
        except Exception as e:
            self.get_logger().error(f"Calibration parsing failed: {e}")
            self.calib_kitti = None

    def sync_cb(self, img_msg, pc_msg, det2_car_msg, det3_car_msg, det2_ped_msg, det3_ped_msg):
        """Callback when all 6 topics are synchronized."""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')  # Convert image
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        # --- Parse Detections ---
        arr_2d_car = self.parse_2d_detections(det2_car_msg, 'Car')
        arr_3d_car = self.parse_3d_detections(det3_car_msg, 'Car')
        arr_2d_ped = self.parse_2d_detections(det2_ped_msg, 'Pedestrian')
        arr_3d_ped = self.parse_3d_detections(det3_ped_msg, 'Pedestrian')

        detection_dict = {
            'Car': {'2d': arr_2d_car, '3d': arr_3d_car},
            'Pedestrian': {'2d': arr_2d_ped, '3d': arr_3d_ped}
        }

        # Publish raw 2D bounding box overlay on image 
        self.publish_raw_detections_image(cv_img, detection_dict)

        # Process each category
        for cat in self.cat_list:
            arr_2d = detection_dict[cat]['2d']
            arr_3d = detection_dict[cat]['3d']

            if arr_2d.shape[0] == 0 and arr_3d.shape[0] == 0:
                continue 

            # Extract fusion inputs
            dets_3d_camera = arr_3d[:, 7:14]
            ori_array = arr_3d[:, 14].reshape(-1, 1)
            other_array = arr_3d[:, 1:7]
            dets_3dto2d_image = arr_3d[:, 2:6]
            additional_info = np.concatenate((ori_array, other_array), axis=1)
            dets_2d_frame = arr_2d[:, 1:5] if arr_2d.shape[0] > 0 else np.empty((0, 4))

            # Run data fusion 
            fused_3d, only3d, only2d = data_fusion(
                dets_3d_camera,
                dets_2d_frame,
                dets_3dto2d_image,
                additional_info
            )

            # Publish fusion results
            now = self.get_clock().now().to_msg()

            msg_fused = Float32MultiArrayStamped()
            msg_fused.header.stamp = now
            msg_fused.data = [float(x) for row in fused_3d['dets_3d_fusion'] for x in row]
            self.pub_fused_3d.publish(msg_fused)

            msg_only3d = Float32MultiArrayStamped()
            msg_only3d.header.stamp = now
            msg_only3d.data = [float(x) for row in only3d['dets_3d_only'] for x in row]
            self.pub_only_3d.publish(msg_only3d)

            msg_only2d = Float32MultiArrayStamped()
            msg_only2d.header.stamp = now
            msg_only2d.data = [float(x) for row in only2d for x in row]
            self.pub_only_2d.publish(msg_only2d)

            self.get_logger().info(
                f"[{cat}] Fusion: fused={len(fused_3d['dets_3d_fusion'])}, "
                f"only3d={len(only3d['dets_3d_only'])}, only2d={len(only2d)}"
            )

    def parse_2d_detections(self, msg: Float32MultiArrayStamped, cat='Car'):
        """Parses 2D detection message into numpy array of shape (N, 6)."""
        if len(msg.data) > 0:
            arr = np.array(msg.data, dtype=np.float32).reshape(-1, 6)
        else:
            arr = np.empty((0, 6))
        return arr

    def parse_3d_detections(self, msg: Float32MultiArrayStamped, cat='Car'):
        """Parses 3D detection message into numpy array of shape (N, 15)."""
        if len(msg.data) > 0:
            arr = np.array(msg.data, dtype=np.float32).reshape(-1, 15)
        else:
            arr = np.empty((0, 15))
        return arr

    def publish_raw_detections_image(self, cv_img, detection_dict):
        """Draws raw 2D boxes on the image and publishes it."""
        if cv_img is None:
            return
        img_draw = cv_img.copy()

        for cat, color in zip(['Car', 'Pedestrian'], [(0, 255, 0), (255, 0, 0)]):
            if cat in self.cat_list:
                for det in detection_dict[cat]['2d']:
                    x1, y1, x2, y2 = map(int, det[1:5])
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_draw, cat, (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        try:
            msg_out = self.bridge.cv2_to_imgmsg(img_draw, encoding='bgr8')
            msg_out.header.stamp = self.get_clock().now().to_msg()
            msg_out.header.frame_id = "camera_link"
            self.raw_detections_image_pub.publish(msg_out)
        except Exception as e:
            self.get_logger().error(f"Failed to publish detection image: {e}")

def main(args=None):
    config_file = '/home/prabuddhi/ros2_ws1/src/deepfusion/config/kitti_real_time.yaml'
    rclpy.init(args=args)
    cfg, _ = Config(config_file) # Load config YAML
    node = RealTimeFusionNode(cfg)
    try:
        rclpy.spin(node)  # Keep node alive
    except KeyboardInterrupt: 
        node.get_logger().info("Shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

