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

# Use ApproximateTimeSynchronizer for approximate timestamp matching
from message_filters import Subscriber, ApproximateTimeSynchronizer

# Local/Project imports
from deepfusionmot.data_fusion import data_fusion
from deepfusionmot.DeepFusionMOT import DeepFusionMOT
from deepfusionmot.coordinate_transformation import convert_x1y1x2y2_to_tlwh
from deepfusionmot.config import Config  # Configuration loader

class RealTimeFusionNode(Node):  # ROS2 Node
    def __init__(self, cfg):
        super().__init__('real_time_fusion_node')
        self.cfg = cfg

        self.cat_list = self.cfg.get('cat_list', ['Car', 'Pedestrian'])
        self.get_logger().info(f"Configured categories: {self.cat_list}")

        self.trackers = {}
        for cat in self.cat_list:
            self.trackers[cat] = DeepFusionMOT(cfg, cat)

        self.bridge = CvBridge()
        self.detection_cache = {}

        image_sub = Subscriber(self, Image, '/camera/image_raw')
        lidar_sub = Subscriber(self, PointCloud2, '/lidar/points')
        self.ts = ApproximateTimeSynchronizer([image_sub, lidar_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.synced_sensor_callback)

        if 'Car' in self.cat_list:
            self.create_subscription(Float32MultiArrayStamped, '/detection_2d/car', self.detection_2d_car_cb, 10)
            self.create_subscription(Float32MultiArrayStamped, '/detection_3d/car', self.detection_3d_car_cb, 10)
        if 'Pedestrian' in self.cat_list:
            self.create_subscription(Float32MultiArrayStamped, '/detection_2d/pedestrian', self.detection_2d_ped_cb, 10)
            self.create_subscription(Float32MultiArrayStamped, '/detection_3d/pedestrian', self.detection_3d_ped_cb, 10)

        self.annotated_image_pub = self.create_publisher(Image, '/annotated_image', 10)
        self.raw_detections_image_pub = self.create_publisher(Image, '/raw_detections_image', 10)

        self.get_logger().info("RealTimeFusionNode initialized with ApproximateTimeSynchronizer.")

    def synced_sensor_callback(self, img_msg: Image, pc_msg: PointCloud2):
        self.get_logger().info(
            f"Received synchronized Image + PointCloud2\n"
            f"  Image timestamp: sec={img_msg.header.stamp.sec}, nanosec={img_msg.header.stamp.nanosec}\n"
            f"  PointCloud timestamp: sec={pc_msg.header.stamp.sec}, nanosec={pc_msg.header.stamp.nanosec}"
        )

        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        self.get_logger().info(f"Image shape: {cv_img.shape[0]} x {cv_img.shape[1]}")
        self.latest_image = cv_img
        self.latest_pc = pc_msg

        ts_key = (img_msg.header.stamp.sec, img_msg.header.stamp.nanosec)
        detection_dict = self.get_nearest_detection_data(ts_key, max_delta=0.1)

        # Print detection data
        #self.get_logger().info(f"Detections at {ts_key}: {detection_dict}")

        self.publish_raw_detections_image(cv_img, detection_dict)
        self.process_frame(cv_img, detection_dict)

    def detection_2d_car_cb(self, msg: Float32MultiArrayStamped):
        ts_key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        arr = (np.array(msg.data, dtype=np.float32).reshape(-1, 6) 
               if len(msg.data) > 0 else np.empty((0, 6)))
        if ts_key not in self.detection_cache:
            self.detection_cache[ts_key] = {}
        if '2d_car' in self.detection_cache[ts_key]:
            existing = self.detection_cache[ts_key]['2d_car']
            if existing.size and arr.size:
                self.detection_cache[ts_key]['2d_car'] = np.vstack([existing, arr])
            elif arr.size:
                self.detection_cache[ts_key]['2d_car'] = arr
        else:
            self.detection_cache[ts_key]['2d_car'] = arr

    def detection_2d_ped_cb(self, msg: Float32MultiArrayStamped):
        ts_key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        arr = (np.array(msg.data, dtype=np.float32).reshape(-1, 6) 
               if len(msg.data) > 0 else np.empty((0, 6)))
        if ts_key not in self.detection_cache:
            self.detection_cache[ts_key] = {}
        if '2d_ped' in self.detection_cache[ts_key]:
            existing = self.detection_cache[ts_key]['2d_ped']
            if existing.size and arr.size:
                self.detection_cache[ts_key]['2d_ped'] = np.vstack([existing, arr])
            elif arr.size:
                self.detection_cache[ts_key]['2d_ped'] = arr
        else:
            self.detection_cache[ts_key]['2d_ped'] = arr

    def detection_3d_car_cb(self, msg: Float32MultiArrayStamped):
        ts_key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        arr = (np.array(msg.data, dtype=np.float32).reshape(-1, 15) 
               if len(msg.data) > 0 else np.empty((0, 15)))
        if ts_key not in self.detection_cache:
            self.detection_cache[ts_key] = {}
        if '3d_car' in self.detection_cache[ts_key]:
            existing = self.detection_cache[ts_key]['3d_car']
            if existing.size and arr.size:
                self.detection_cache[ts_key]['3d_car'] = np.vstack([existing, arr])
            elif arr.size:
                self.detection_cache[ts_key]['3d_car'] = arr
        else:
            self.detection_cache[ts_key]['3d_car'] = arr

    def detection_3d_ped_cb(self, msg: Float32MultiArrayStamped):
        ts_key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        arr = (np.array(msg.data, dtype=np.float32).reshape(-1, 15) 
               if len(msg.data) > 0 else np.empty((0, 15)))
        if ts_key not in self.detection_cache:
            self.detection_cache[ts_key] = {}
        if '3d_ped' in self.detection_cache[ts_key]:
            existing = self.detection_cache[ts_key]['3d_ped']
            if existing.size and arr.size:
                self.detection_cache[ts_key]['3d_ped'] = np.vstack([existing, arr])
            elif arr.size:
                self.detection_cache[ts_key]['3d_ped'] = arr
        else:
            self.detection_cache[ts_key]['3d_ped'] = arr

    def get_nearest_detection_data(self, ts_key, max_delta=0.2):
        def to_sec(key):
            return key[0] + key[1] * 1e-9

        t_query = to_sec(ts_key)
        merged_data_lists = {}

        for k, data in self.detection_cache.items():
            t_k = to_sec(k)
            if abs(t_k - t_query) <= max_delta:
                for det_type, arr in data.items():
                    if arr.size == 0:
                        continue
                    if det_type not in merged_data_lists:
                        merged_data_lists[det_type] = []
                    merged_data_lists[det_type].append(arr)

        merged_data = {}
        for det_type, list_of_arrays in merged_data_lists.items():
            try:
                merged_data[det_type] = np.vstack(list_of_arrays) if list_of_arrays else np.empty((0,))
            except Exception as e:
                self.get_logger().error(f"Error stacking arrays for {det_type}: {e}")
                merged_data[det_type] = np.empty((0,))

        if merged_data:
            self.get_logger().info("Merged detection data found within max_delta.")
        else:
            self.get_logger().info("No detection data within max_delta seconds.")

        # Purge old entries from the detection cache
        keys_to_delete = []
        for k in list(self.detection_cache.keys()):
            if to_sec(k) < t_query - max_delta:
                keys_to_delete.append(k)
        for k in keys_to_delete:
            del self.detection_cache[k]

        return merged_data

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
            else:
                if '2d_ped' in detection_dict:
                    dets_2d = detection_dict['2d_ped']
                if '3d_ped' in detection_dict:
                    dets_3d = detection_dict['3d_ped']

            trackers = self.run_fusion_and_update(self.trackers[cat], dets_2d, dets_3d, cat)
            if trackers is not None and trackers.size > 0:
                trackers[:, 0] = self.assign_unique_ids(cat, trackers[:, 0])
                combined_trackers.append(trackers)

        if combined_trackers:
            combined_trackers = np.vstack(combined_trackers)
        else:
            combined_trackers = np.empty((0, 10))

        self.publish_3d_annotated_image(cv_img, combined_trackers)

    def run_fusion_and_update(self, tracker, dets_2d, dets_3d, category_name):
        num_2d = dets_2d.shape[0]
        num_3d = dets_3d.shape[0]
        if num_2d == 0 and num_3d == 0:
            return np.empty((0, 10))
        trackers_output = np.empty((0, 10))
        return trackers_output

    def publish_raw_detections_image(self, cv_img, detection_dict):
        if cv_img is None:
            return

        img_raw = cv_img.copy()
        drawn_boxes = []  # To track drawn bounding boxes

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
                    # Log the box coordinates
                    self.get_logger().info(f"Box coordinates: ({x1}, {y1}), ({x2}, {y2})")
                    
                    duplicate_found = False
                    for bx1, by1, bx2, by2 in drawn_boxes:
                        if abs(x1 - bx1) < 5 and abs(y1 - by1) < 5 and abs(x2 - bx2) < 5 and abs(y2 - by2) < 5:
                            duplicate_found = True
                            break
                    if duplicate_found:
                        continue

                    drawn_boxes.append((x1, y1, x2, y2))
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

