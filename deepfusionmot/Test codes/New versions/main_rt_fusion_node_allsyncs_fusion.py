#!/usr/bin/env python3
import os
import math
import numpy as np
import cv2
import time

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

# ROS messages
from sensor_msgs.msg import Image, PointCloud2
from my_msgs.msg import Float32MultiArrayStamped  # aggregated bounding boxes

# For exact-time matching across multiple topics
from message_filters import Subscriber, TimeSynchronizer

# DeepFusionMOT project imports
from deepfusionmot.data_fusion import data_fusion
from deepfusionmot.DeepFusionMOT import DeepFusionMOT
from deepfusionmot.coordinate_transformation import convert_x1y1x2y2_to_tlwh
from deepfusionmot.config import Config


class RealTimeFusionNode(Node):
    """
    A node that uses a single TimeSynchronizer for 6 topics:
      1) /camera/image_raw        (sensor_msgs/Image)
      2) /lidar/points           (sensor_msgs/PointCloud2)
      3) /detection_2d/car       (my_msgs/Float32MultiArrayStamped)
      4) /detection_3d/car       (my_msgs/Float32MultiArrayStamped)
      5) /detection_2d/pedestrian (my_msgs/Float32MultiArrayStamped)
      6) /detection_3d/pedestrian (my_msgs/Float32MultiArrayStamped)

    After modifying "Code 1" to publish aggregated detections,
    each Float32MultiArrayStamped now has either Nx6 (2D) or Nx15 (3D).
    We fuse them here with data_fusion() and track with DeepFusionMOT.
    """

    def __init__(self, cfg):
        super().__init__('real_time_fusion_node')
        self.cfg = cfg

        # Read categories from config
        self.cat_list = self.cfg.get('cat_list', ['Car', 'Pedestrian'])
        self.get_logger().info(f"Configured categories: {self.cat_list}")

        self.bridge = CvBridge()

        # Create a DeepFusionMOT tracker for each category
        self.trackers = {}
        for cat in self.cat_list:
            self.trackers[cat] = DeepFusionMOT(cfg, cat)

        # 1) Create Subscribers for each needed topic
        self.img_sub  = Subscriber(self, Image, '/camera/image_raw')
        self.pc_sub   = Subscriber(self, PointCloud2, '/lidar/points')

        self.det2_car_sub = Subscriber(self, Float32MultiArrayStamped, '/detection_2d/car')
        self.det3_car_sub = Subscriber(self, Float32MultiArrayStamped, '/detection_3d/car')
        self.det2_ped_sub = Subscriber(self, Float32MultiArrayStamped, '/detection_2d/pedestrian')
        self.det3_ped_sub = Subscriber(self, Float32MultiArrayStamped, '/detection_3d/pedestrian')

        # 2) TimeSynchronizer with queue_size=10
        self.ts = TimeSynchronizer([
            self.img_sub,
            self.pc_sub,
            self.det2_car_sub,
            self.det3_car_sub,
            self.det2_ped_sub,
            self.det3_ped_sub,
        ], queue_size=10)
        self.ts.registerCallback(self.sync_cb)

        # 3) Publishers for debug images
        self.raw_detections_image_pub = self.create_publisher(Image, '/raw_detections_image', 10)
        self.annotated_image_pub      = self.create_publisher(Image, '/annotated_image', 10)

        self.get_logger().info("RealTimeFusionNode with 6-topic TimeSynchronizer in place.")

    # ------------------------------------------------------------
    # Callback from TimeSynchronizer
    # ------------------------------------------------------------
    def sync_cb(self,
                img_msg: Image,
                pc_msg: PointCloud2,
                det2_car_msg: Float32MultiArrayStamped,
                det3_car_msg: Float32MultiArrayStamped,
                det2_ped_msg: Float32MultiArrayStamped,
                det3_ped_msg: Float32MultiArrayStamped):
        """
        Called whenever all 6 topics have a message with the same (sec, nsec).
        """
        stamp_sec  = img_msg.header.stamp.sec
        stamp_nsec = img_msg.header.stamp.nanosec
        self.get_logger().info(
            f"TimeSync => stamp=({stamp_sec},{stamp_nsec})"
        )

        # Convert image to OpenCV
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Parse bounding boxes from messages
        arr_2d_car = self.parse_2d_detections(det2_car_msg, category='Car')
        arr_3d_car = self.parse_3d_detections(det3_car_msg, category='Car')

        arr_2d_ped = self.parse_2d_detections(det2_ped_msg, category='Pedestrian')
        arr_3d_ped = self.parse_3d_detections(det3_ped_msg, category='Pedestrian')

        # Combine them in a dictionary
        detection_dict = {
            'Car': {
                '2d': arr_2d_car,
                '3d': arr_3d_car,
            },
            'Pedestrian': {
                '2d': arr_2d_ped,
                '3d': arr_3d_ped,
            }
        }

        # Publish 2D bounding box overlay
        self.publish_raw_detections_image(cv_img, detection_dict)

        # Perform data fusion + tracking
        self.fuse_and_track(cv_img, detection_dict)

    # ------------------------------------------------------------
    # Parse Nx6 or Nx15 arrays
    # ------------------------------------------------------------
    def parse_2d_detections(self, msg: Float32MultiArrayStamped, category='Car'):
        """
        Each 2D detection => [frame_idx, x1, y1, x2, y2, score].
        """
        n = len(msg.data)
        if n > 0:
            arr = np.array(msg.data, dtype=np.float32).reshape(-1, 6)
            self.get_logger().info(
                f"[2D {category}] stamp=({msg.header.stamp.sec},{msg.header.stamp.nanosec}), shape={arr.shape}"
            )
        else:
            arr = np.empty((0, 6))
            self.get_logger().info(
                f"[2D {category}] (empty) => stamp=({msg.header.stamp.sec},{msg.header.stamp.nanosec})"
            )
        return arr

    def parse_3d_detections(self, msg: Float32MultiArrayStamped, category='Car'):
        """
        Each 3D detection => [frame_idx, x1_3D, y1_3D, x2_3D, y2_3D, score_3D, h, w, l, x, y, z, theta, orientation].
        Adjust if the columns differ in your actual data.
        """
        n = len(msg.data)
        if n > 0:
            arr = np.array(msg.data, dtype=np.float32).reshape(-1, 15)
            self.get_logger().info(
                f"[3D {category}] stamp=({msg.header.stamp.sec},{msg.header.stamp.nanosec}), shape={arr.shape}"
            )
        else:
            arr = np.empty((0, 15))
            self.get_logger().info(
                f"[3D {category}] (empty) => stamp=({msg.header.stamp.sec},{msg.header.stamp.nanosec})"
            )
        return arr

    # ------------------------------------------------------------
    # Publish 2D bounding boxes overlay
    # ------------------------------------------------------------
    def publish_raw_detections_image(self, cv_img, detection_dict):
        if cv_img is None:
            return

        img_draw = cv_img.copy()

        # Car bounding boxes => green
        if 'Car' in self.cat_list:
            arr_2d = detection_dict['Car']['2d']
            color  = (0, 255, 0)
            for det in arr_2d:
                # Format: [frame_idx, x1, y1, x2, y2, score]
                x1, y1, x2, y2 = map(int, det[1:5])
                cv2.rectangle(img_draw, (x1,y1), (x2,y2), color, 2)
                cv2.putText(img_draw, "Car", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Ped bounding boxes => blue
        if 'Pedestrian' in self.cat_list:
            arr_2d = detection_dict['Pedestrian']['2d']
            color  = (255, 0, 0)
            for det in arr_2d:
                x1, y1, x2, y2 = map(int, det[1:5])
                cv2.rectangle(img_draw, (x1,y1), (x2,y2), color, 2)
                cv2.putText(img_draw, "Ped", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Convert back to ROS Image and publish
        try:
            raw_msg = self.bridge.cv2_to_imgmsg(img_draw, encoding='bgr8')
            raw_msg.header.stamp = self.get_clock().now().to_msg()
            raw_msg.header.frame_id = "camera_link"
            self.raw_detections_image_pub.publish(raw_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish raw detection image: {e}")

    # ------------------------------------------------------------
    # Fusion + Tracking
    # ------------------------------------------------------------
    def fuse_and_track(self, cv_img, detection_dict):
        """
        1) For each category => parse 2D/3D arrays,
        2) Prepare them for data_fusion(...),
        3) Call self.trackers[cat].update(...),
        4) Print/log the tracker outputs.
        """
        for cat in self.cat_list:
            arr_2d = detection_dict[cat]['2d']  # Nx6 => [frame_idx, x1, y1, x2, y2, score]
            arr_3d = detection_dict[cat]['3d']  # Nx15

            n2d = arr_2d.shape[0]
            n3d = arr_3d.shape[0]
            self.get_logger().info(f"Fusion => {cat}, #2D={n2d}, #3D={n3d}")

            if n2d == 0 and n3d == 0:
                continue

            # Derive a frame_idx from the first row of whichever array is non-empty
            # If both are non-empty, we can pick arr_2d for convenience
            if n2d > 0:
                frame_idx = int(arr_2d[0, 0])
            else:
                frame_idx = int(arr_3d[0, 0])

            # For example, columns 7..14 => [h, w, l, x, y, z, theta]; adjust if needed
            dets_3d_camera = arr_3d[:, 7:14]  # shape [N, 7]
            ori_array = arr_3d[:, 14].reshape(-1, 1)  # orientation
            other_array = arr_3d[:, 1:7]             # x1_3D, y1_3D, x2_3D, y2_3D, score_3D
            dets_3dto2d_image = arr_3d[:, 2:6]       # x1_3D, y1_3D, x2_3D, y2_3D

            additional_info = np.concatenate((ori_array, other_array), axis=1)

            # 2D bounding boxes => columns [x1, y1, x2, y2]
            dets_2d_frame = arr_2d[:, 1:5] if n2d > 0 else np.empty((0,4))

            # Data fusion => dictionary results
            dets_3d_fusion, dets_3d_only, dets_2d_only = data_fusion(
                dets_3d_camera,      # shape [N,7]
                dets_2d_frame,       # shape [M,4]
                dets_3dto2d_image,   # shape [N,4]
                additional_info      # shape [N, (1+6)=7]
            )

            # Convert 2D-only boxes from (x1,y1,x2,y2) => (x, y, w, h)
            if len(dets_2d_only) > 0:
                dets_2d_only_tlwh = np.array([convert_x1y1x2y2_to_tlwh(b) for b in dets_2d_only])
            else:
                dets_2d_only_tlwh = np.empty((0,4))

            # Call the tracker's update method
            start_time = time.time()
            trackers_out = self.trackers[cat].update(
                dets_3d_fusion,
                dets_2d_only_tlwh,
                dets_3d_only,
                self.cfg,
                frame_idx  # pass the frame index
            )
            cycle_time = time.time() - start_time

            self.get_logger().info(
                f"Tracking {cat}: took {cycle_time:.3f}s, #tracks_out={len(trackers_out)}"
            )
            if len(trackers_out) > 0:
                self.get_logger().info(f"Tracker outputs:\n{trackers_out}")

        # Optionally publish a 3D-annotated image if you do 3D box projection
        self.publish_3d_annotated_image(cv_img, np.empty((0, 10)))

    # ------------------------------------------------------------
    # Publish 3D-annotated image (placeholder)
    # ------------------------------------------------------------
    def publish_3d_annotated_image(self, cv_img, trackers):
        annotated = cv_img.copy()
        # If you want to project 3D bounding boxes, do so here
        try:
            ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            ann_msg.header.stamp = self.get_clock().now().to_msg()
            ann_msg.header.frame_id = "camera_link"
            self.annotated_image_pub.publish(ann_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish 3D-annotated image: {e}")

def main(args=None):
    config_file = '/home/prabuddhi/ros2_ws1/src/deepfusionmot/config/kitti_real_time.yaml'
    rclpy.init(args=args)
    cfg, _ = Config(config_file)

    node = RealTimeFusionNode(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

