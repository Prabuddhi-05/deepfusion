#!/usr/bin/env python3
import os
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

# ROS 2 message imports
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String

# Import the custom stamped message
from my_msgs.msg import Float32MultiArrayStamped

# Use TimeSynchronizer for exact timestamp matching
from message_filters import Subscriber, TimeSynchronizer

# Local/Project imports
from deepfusionmot.data_fusion import data_fusion
from deepfusionmot.DeepFusionMOT import DeepFusionMOT
from deepfusionmot.coordinate_transformation import convert_x1y1x2y2_to_tlwh
from deepfusionmot.config import Config  # Configuration loader

class RealTimeFusionNode(Node):  # ROS2 Node
    def __init__(self, cfg):
        super().__init__('real_time_fusion_node')
        self.cfg = cfg  # Store configuration passed to the node

        # Read and store categories from config
        self.cat_list = self.cfg.get('cat_list', ['Car', 'Pedestrian'])
        self.get_logger().info(f"Configured categories: {self.cat_list}")

        # Initialize trackers for each category
        self.trackers = {}
        for cat in self.cat_list:
            self.trackers[cat] = DeepFusionMOT(cfg, cat)

        self.bridge = CvBridge()

        # Detection cache: (sec, nanosec) -> { '2d_car': np.array, ... }
        self.detection_cache = {}

        # ---------------------------------------------------------
        # Exact-time Synchronization for /camera/image_raw + /lidar/points
        # ---------------------------------------------------------
        image_sub = Subscriber(self, Image, '/camera/image_raw')
        lidar_sub = Subscriber(self, PointCloud2, '/lidar/points')

        # TimeSynchronizer - requires exact matching of timestamps
        self.ts = TimeSynchronizer([image_sub, lidar_sub], queue_size=10)
        self.ts.registerCallback(self.synced_sensor_callback)

        # ---------------------------------------------------------
        # Subscriptions for detections
        # ---------------------------------------------------------
        if 'Car' in self.cat_list:
            self.create_subscription(Float32MultiArrayStamped, '/detection_2d/car', self.detection_2d_car_cb, 10)
            self.create_subscription(Float32MultiArrayStamped, '/detection_3d/car', self.detection_3d_car_cb, 10)

        if 'Pedestrian' in self.cat_list:
            self.create_subscription(Float32MultiArrayStamped, '/detection_2d/pedestrian', self.detection_2d_ped_cb, 10)
            self.create_subscription(Float32MultiArrayStamped, '/detection_3d/pedestrian', self.detection_3d_ped_cb, 10)

        # ---------------------------------------------------------
        # Publishers
        # ---------------------------------------------------------
        self.annotated_image_pub = self.create_publisher(Image, '/annotated_image', 10)
        self.raw_detections_image_pub = self.create_publisher(Image, '/raw_detections_image', 10)

        self.get_logger().info("RealTimeFusionNode initialized, using **TimeSynchronizer** for exact timestamps.")

    # ============================================================
    #  Synchronized Sensor Callback (Exact Timestamps)
    # ============================================================
    def synced_sensor_callback(self, img_msg: Image, pc_msg: PointCloud2):
        self.get_logger().info(
            f"Received synchronized Image + PointCloud2\n"
            f"  Image timestamp: sec={img_msg.header.stamp.sec}, nanosec={img_msg.header.stamp.nanosec}\n"
            f"  PointCloud timestamp: sec={pc_msg.header.stamp.sec}, nanosec={pc_msg.header.stamp.nanosec}"
        )

        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8') # Converts ROS2 images to OpenCV images
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
            
        # Log the dimensions of the converted OpenCV image
        self.get_logger().info(f"Image shape: {cv_img.shape[0]} x {cv_img.shape[1]} (H x W)")
        
        # Store the latest image and point cloud 
        self.latest_image = cv_img
        self.latest_pc = pc_msg

        # Generate a timestamp key from the image message 
        ts_key = (img_msg.header.stamp.sec, img_msg.header.stamp.nanosec)
        
        # Retrieve detection data closest to the current timestamp
        detection_dict = self.get_nearest_detection_data(ts_key)

        # Publish an image showing raw detections
        self.publish_raw_detections_image(cv_img, detection_dict)
        
        # Process the frame to update tracking and perform data fusion
        self.process_frame(cv_img, detection_dict)

    # ============================================================
    #  Detection Callbacks (using Float32MultiArrayStamped)
    # ============================================================
    def detection_2d_car_cb(self, msg: Float32MultiArrayStamped):
        # Extract and log the timestamp from the incoming message
        ts_key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        # Length of the data in the message
        arr_len = len(msg.data)
        self.get_logger().info(f"[2D_CAR] Received Float32MultiArrayStamped with {arr_len} floats, ts_key={ts_key}")

        # Check if there is any detection data in the message
        if arr_len > 0:
        # Convert the flat data array into a structured numpy array, each row containing 6 elements
            arr = np.array(msg.data, dtype=np.float32).reshape(-1, 6)
            self.get_logger().info(f"[2D_CAR] detection array shape: {arr.shape}")
            self.get_logger().info(f"[2D_CAR] detection data:\n{arr}")
        # Handle cases where no detections are received
        else:
            arr = np.empty((0, 6))
            self.get_logger().info(f"[2D_CAR] No detections received.")
            
        # Store the received detection data in the detection cache using the timestamp as the key
        if ts_key not in self.detection_cache:
            self.detection_cache[ts_key] = {}
        self.detection_cache[ts_key]['2d_car'] = arr

    def detection_2d_ped_cb(self, msg: Float32MultiArrayStamped):
        ts_key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        arr_len = len(msg.data)
        self.get_logger().info(f"[2D_PED] Received Float32MultiArrayStamped with {arr_len} floats, ts_key={ts_key}")

        if arr_len > 0:
            arr = np.array(msg.data, dtype=np.float32).reshape(-1, 6)
            self.get_logger().info(f"[2D_PED] detection array shape: {arr.shape}")
            self.get_logger().info(f"[2D_PED] detection data:\n{arr}")
        else:
            arr = np.empty((0, 6))
            self.get_logger().info(f"[2D_PED] No detections received.")

        if ts_key not in self.detection_cache:
            self.detection_cache[ts_key] = {}
        self.detection_cache[ts_key]['2d_ped'] = arr

    def detection_3d_car_cb(self, msg: Float32MultiArrayStamped):
        ts_key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        arr_len = len(msg.data)
        self.get_logger().info(f"[3D_CAR] Received Float32MultiArrayStamped with {arr_len} floats, ts_key={ts_key}")

        if arr_len > 0:
            arr = np.array(msg.data, dtype=np.float32).reshape(-1, 15)
            self.get_logger().info(f"[3D_CAR] detection array shape: {arr.shape}")
            self.get_logger().info(f"[3D_CAR] detection data:\n{arr}")
        else:
            arr = np.empty((0, 15))
            self.get_logger().info(f"[3D_CAR] No detections received.")

        if ts_key not in self.detection_cache:
            self.detection_cache[ts_key] = {}
        self.detection_cache[ts_key]['3d_car'] = arr

    def detection_3d_ped_cb(self, msg: Float32MultiArrayStamped):
        ts_key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        arr_len = len(msg.data)
        self.get_logger().info(f"[3D_PED] Received Float32MultiArrayStamped with {arr_len} floats, ts_key={ts_key}")

        if arr_len > 0:
            arr = np.array(msg.data, dtype=np.float32).reshape(-1, 15)
            self.get_logger().info(f"[3D_PED] detection array shape: {arr.shape}")
            self.get_logger().info(f"[3D_PED] detection data:\n{arr}")
        else:
            arr = np.empty((0, 15))
            self.get_logger().info(f"[3D_PED] No detections received.")

        if ts_key not in self.detection_cache:
            self.detection_cache[ts_key] = {}
        self.detection_cache[ts_key]['3d_ped'] = arr

    # ============================================================
    #  Find Nearest Detection Data
    # ============================================================
    def get_nearest_detection_data(self, ts_key, max_delta=0.05):
        def to_sec(key):
            return key[0] + key[1]*1e-9

        t_query = to_sec(ts_key)
        best_key = None
        best_diff = float('inf')

        for k in self.detection_cache.keys():
            t_k = to_sec(k)
            diff = abs(t_k - t_query)
            if diff < best_diff:
                best_diff = diff
                best_key = k

        if best_key is not None and best_diff <= max_delta:
            self.get_logger().info(
                f"Found detection data close to timestamp (diff={best_diff:.5f}s)."
            )
            return self.detection_cache[best_key]
        else:
            self.get_logger().info("No detection data within max_delta seconds.")
            return {}

    # ============================================================
    #  Processing / Fusion
    # ============================================================
    def process_frame(self, cv_img, detection_dict):
        if cv_img is None:
            self.get_logger().warn("No image to process.")
            return

        combined_trackers = []

        for cat in self.cat_list:
            dets_2d = np.empty((0, 6))
            dets_3d = np.empty((0, 15))

            if cat == 'Car':
                if '2d_car' in detection_dict:
                    dets_2d = detection_dict['2d_car']
                if '3d_car' in detection_dict:
                    dets_3d = detection_dict['3d_car']
            else:  # 'Pedestrian'
                if '2d_ped' in detection_dict:
                    dets_2d = detection_dict['2d_ped']
                if '3d_ped' in detection_dict:
                    dets_3d = detection_dict['3d_ped']

            self.get_logger().info(
                f"Category={cat}, #2D={dets_2d.shape[0]}, #3D={dets_3d.shape[0]}"
            )

            trackers = self.run_fusion_and_update(self.trackers[cat], dets_2d, dets_3d, cat)
            if trackers is not None and trackers.size > 0:
                trackers[:, 0] = self.assign_unique_ids(cat, trackers[:, 0])
                combined_trackers.append(trackers)

        if combined_trackers:
            combined_trackers = np.vstack(combined_trackers)
        else:
            combined_trackers = np.empty((0, 10))

        self.get_logger().info(f"Synchronized Fusion => {combined_trackers.shape[0]} objects tracked.")
        self.publish_3d_annotated_image(cv_img, combined_trackers)

    def run_fusion_and_update(self, tracker, dets_2d, dets_3d, category_name):
        num_2d = dets_2d.shape[0]
        num_3d = dets_3d.shape[0]
        self.get_logger().info(f"Fusion/update '{category_name}': {num_2d} 2D, {num_3d} 3D")

        if num_2d == 0 and num_3d == 0:
            return np.empty((0, 10))

        # data_fusion(...) => trackers_output = tracker.update(...)
        trackers_output = np.empty((0, 10))
        return trackers_output

    # ============================================================
    #  Publish Raw 2D Detections Overlay
    # ============================================================
    def publish_raw_detections_image(self, cv_img, detection_dict):
        if cv_img is None:
            return

        img_raw = cv_img.copy()

        for cat in self.cat_list:
            if cat == 'Car':
                dets_2d = detection_dict.get('2d_car', np.empty((0, 6)))
                color = (0, 255, 0)
            else:
                dets_2d = detection_dict.get('2d_ped', np.empty((0, 6)))
                color = (255, 0, 0)

            label = cat
            self.get_logger().info(f"Drawing {dets_2d.shape[0]} bounding boxes for category={cat}.")
            for det in dets_2d:
                try:
                    x1, y1, x2, y2 = map(int, det[1:5])
                    cv2.rectangle(img_raw, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_raw, label, (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except Exception as e:
                    self.get_logger().error(f"Error drawing {label} detection: {e}")

        try:
            raw_msg = self.bridge.cv2_to_imgmsg(img_raw, encoding='bgr8')
            raw_msg.header.stamp = self.get_clock().now().to_msg()
            raw_msg.header.frame_id = "camera_link"
            self.raw_detections_image_pub.publish(raw_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish raw detections image: {e}")

    # ============================================================
    #  Publish 3D-Annotated Image
    # ============================================================
    def publish_3d_annotated_image(self, cv_img, trackers):
        annotated = cv_img.copy()

        for row in trackers:
            if len(row) < 9:
                continue
            track_id = int(row[0])
            h, w, l, x, y, z, theta = row[1:8]
            bbox3d_tmp = np.array([h, w, l, x, y, z, theta], dtype=np.float32)
            color = self.compute_color_for_id(track_id)
            label_str = f"ID:{track_id}"

            annotated = self.show_image_with_boxes_3d(annotated, bbox3d_tmp, color, label_str)

        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            annotated_msg.header.stamp = self.get_clock().now().to_msg()
            annotated_msg.header.frame_id = "camera_link"
            self.annotated_image_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish 3D-annotated image: {e}")

    def show_image_with_boxes_3d(self, img, bbox3d_tmp, color=(255, 255, 255), label_str="", line_thickness=2):
        corners_2d = self.project_3d_box(bbox3d_tmp)
        if corners_2d is not None and corners_2d.shape == (8, 2):
            img = self.draw_projected_box3d(img, corners_2d, color, line_thickness)
        return img

    def draw_projected_box3d(self, image, qs, color=(255, 255, 255), thickness=2):
        qs = qs.astype(int)
        for k in range(4):
            i, j = k, (k + 1) % 4
            image = cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k + 4, ((k + 1) % 4) + 4
            image = cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k, k + 4
            image = cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        return image

    def project_3d_box(self, box3d_tmp):
        # Suppose (h, w, l, x, y, z, theta)
        corners_2d = np.zeros((8, 2), dtype=np.float32)
        return corners_2d

    def assign_unique_ids(self, category, ids):
        category_mapping = {cat: idx + 1 for idx, cat in enumerate(self.cat_list)}
        prefix = category_mapping.get(category, 0) * 1000
        unique_ids = prefix + ids
        return unique_ids

    def compute_color_for_id(self, idx):
        palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
        color = [int((p * (idx**2 - idx + 1)) % 255) for p in palette]
        return tuple(color)

def main(args=None):
    config_file = '/home/prabuddhi/ros2_ws1/src/deepfusionmot/config/kitti_real_time.yaml'
    rclpy.init(args=args)

    cfg, _ = Config(config_file)
    node = RealTimeFusionNode(cfg)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down the node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

