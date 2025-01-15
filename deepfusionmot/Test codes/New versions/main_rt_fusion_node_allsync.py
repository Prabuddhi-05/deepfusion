#!/usr/bin/env python3
import os
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

# ROS messages
from sensor_msgs.msg import Image, PointCloud2
from my_msgs.msg import Float32MultiArrayStamped  # aggregated bounding boxes

# For exact-time matching across multiple topics
from message_filters import Subscriber, TimeSynchronizer

# Example project imports (adapt to your structure)
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

    After modifying "Code 1" to aggregate all detections of a category & frame
    into a single message, each Float32MultiArrayStamped now contains:
      - 2D:   Nx6  (frame_idx, x1, y1, x2, y2, score)
      - 3D:   Nx15 (frame_idx, plus 14 more floats for 3D)

    Each topic has exactly one message (possibly empty) per timestamp, so that
    TimeSynchronizer can match them perfectly.
    """

    def __init__(self, cfg):
        super().__init__('real_time_fusion_node')
        self.cfg = cfg

        # Read categories from config
        self.cat_list = self.cfg.get('cat_list', ['Car', 'Pedestrian'])
        self.get_logger().info(f"Configured categories: {self.cat_list}")

        self.bridge = CvBridge()

        # Example: create one tracker per category
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

        # 2) TimeSynchronizer with queue_size
        #    The callback fires ONLY if all 6 topics share the same stamp
        self.ts = TimeSynchronizer([
            self.img_sub,
            self.pc_sub,
            self.det2_car_sub,
            self.det3_car_sub,
            self.det2_ped_sub,
            self.det3_ped_sub,
        ], queue_size=10)
        self.ts.registerCallback(self.sync_cb)

        # 3) Publishers: raw bounding boxes image, plus optional 3D annotated image
        self.raw_detections_image_pub = self.create_publisher(Image, '/raw_detections_image', 10)
        self.annotated_image_pub      = self.create_publisher(Image, '/annotated_image', 10)

        self.get_logger().info("RealTimeFusionNode with 6-topic TimeSynchronizer in place.")

    # ------------------------------------------------------------
    # The single callback from TimeSynchronizer
    # ------------------------------------------------------------
    def sync_cb(self,
                img_msg: Image,
                pc_msg: PointCloud2,
                det2_car_msg: Float32MultiArrayStamped,
                det3_car_msg: Float32MultiArrayStamped,
                det2_ped_msg: Float32MultiArrayStamped,
                det3_ped_msg: Float32MultiArrayStamped):
        """
        Called whenever all 6 topics have a message with the exact same (sec, nsec).
        If a detection is truly absent, we expect an 'empty' Float32MultiArrayStamped
        for that topic at this timestamp (length=0).
        """
        stamp_sec  = img_msg.header.stamp.sec
        stamp_nsec = img_msg.header.stamp.nanosec
        self.get_logger().info(
            f"TimeSync callback => stamp=({stamp_sec},{stamp_nsec})"
        )

        # Convert image to OpenCV
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Parse bounding boxes
        arr_2d_car = self.parse_2d_detections(det2_car_msg, category='Car')
        arr_3d_car = self.parse_3d_detections(det3_car_msg, category='Car')

        arr_2d_ped = self.parse_2d_detections(det2_ped_msg, category='Pedestrian')
        arr_3d_ped = self.parse_3d_detections(det3_ped_msg, category='Pedestrian')

        # Combine them in a dictionary for easy usage
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

        # Publish 2D bounding box overlay for quick visualization
        self.publish_raw_detections_image(cv_img, detection_dict)

        # Optionally do fusion + tracking
        self.fuse_and_track(cv_img, detection_dict)

        # pc_msg is available if you need 3D sensor data for fusion

    # ------------------------------------------------------------
    # Parse Nx6 or Nx15 arrays
    # ------------------------------------------------------------
    def parse_2d_detections(self, msg: Float32MultiArrayStamped, category='Car'):
        """
        Each 2D detection is [frame_idx, x1, y1, x2, y2, score],
        so we reshape to (-1, 6).
        """
        n = len(msg.data)
        if n > 0:
            arr = np.array(msg.data, dtype=np.float32).reshape(-1, 6)
            self.get_logger().info(
                f"[2D {category}] stamp=({msg.header.stamp.sec},{msg.header.stamp.nanosec}), shape={arr.shape}\n{arr}"
            )
        else:
            arr = np.empty((0, 6))
            self.get_logger().info(
                f"[2D {category}] (empty) => stamp=({msg.header.stamp.sec},{msg.header.stamp.nanosec})"
            )
        return arr

    def parse_3d_detections(self, msg: Float32MultiArrayStamped, category='Car'):
        """
        Each 3D detection is [frame_idx, ...14 more floats...],
        so we reshape to (-1, 15).
        """
        n = len(msg.data)
        if n > 0:
            arr = np.array(msg.data, dtype=np.float32).reshape(-1, 15)
            self.get_logger().info(
                f"[3D {category}] stamp=({msg.header.stamp.sec},{msg.header.stamp.nanosec}), shape={arr.shape}\n{arr}"
            )
        else:
            arr = np.empty((0, 15))
            self.get_logger().info(
                f"[3D {category}] (empty) => stamp=({msg.header.stamp.sec},{msg.header.stamp.nanosec})"
            )
        return arr

    # ------------------------------------------------------------
    # Publish 2D bounding boxes overlay (just for quick debug)
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
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_draw, "Car", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Pedestrian bounding boxes => blue
        if 'Pedestrian' in self.cat_list:
            arr_2d = detection_dict['Pedestrian']['2d']
            color  = (255, 0, 0)
            for det in arr_2d:
                x1, y1, x2, y2 = map(int, det[1:5])
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_draw, "Ped", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Convert to ROS Image and publish
        try:
            raw_msg = self.bridge.cv2_to_imgmsg(img_draw, encoding='bgr8')
            # Use the node's current time or the image's original stamp
            raw_msg.header.stamp = self.get_clock().now().to_msg()
            raw_msg.header.frame_id = "camera_link"
            self.raw_detections_image_pub.publish(raw_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish raw detection image: {e}")

    # ------------------------------------------------------------
    # Fusion / Tracking placeholder
    # ------------------------------------------------------------
    def fuse_and_track(self, cv_img, detection_dict):
        for cat in self.cat_list:
            arr_2d = detection_dict[cat]['2d']
            arr_3d = detection_dict[cat]['3d']
            n2d = arr_2d.shape[0]
            n3d = arr_3d.shape[0]
            self.get_logger().info(f"Fusion => {cat}, #2D={n2d}, #3D={n3d}")

            # run_fusion_and_update is a placeholder for your logic
            trackers_out = self.run_fusion_and_update(self.trackers[cat], arr_2d, arr_3d, cat)
            if trackers_out.size > 0:
                trackers_out[:, 0] = self.assign_unique_ids(cat, trackers_out[:, 0])

        # Optionally publish a 3D-annotated image if you do box projection
        self.publish_3d_annotated_image(cv_img, np.empty((0, 10)))

    def run_fusion_and_update(self, tracker, arr_2d, arr_3d, cat_name):
        """
        Example stub. You would integrate your actual data_fusion + tracker.update() logic here.
        Return an array where the first column is track_id, 
        and the remaining columns are bounding box or state info.
        """
        n2d = arr_2d.shape[0]
        n3d = arr_3d.shape[0]
        self.get_logger().info(f"run_fusion_and_update '{cat_name}': 2D={n2d}, 3D={n3d}")
        if n2d == 0 and n3d == 0:
            return np.empty((0, 10))

        # Suppose you call tracker.update(...) here.
        # trackers_out = tracker.update(arr_2d, arr_3d)
        # For now, return an empty array or a mock example
        return np.empty((0, 10))

    # ------------------------------------------------------------
    # Publish 3D-annotated image (placeholder)
    # ------------------------------------------------------------
    def publish_3d_annotated_image(self, cv_img, trackers):
        annotated = cv_img.copy()
        # If you had 3D boxes => project them into the image, draw them, etc.
        try:
            ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            ann_msg.header.stamp = self.get_clock().now().to_msg()
            ann_msg.header.frame_id = "camera_link"
            self.annotated_image_pub.publish(ann_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish 3D-annotated image: {e}")

    # ------------------------------------------------------------
    # Helper for unique track IDs per category
    # ------------------------------------------------------------
    def assign_unique_ids(self, cat, ids):
        prefix_map = {
            'Car': 1000,
            'Pedestrian': 2000
        }
        prefix = prefix_map.get(cat, 3000)
        return prefix + ids

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

