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
from std_msgs.msg import Float32MultiArray, String

# Local/Project imports
from deepfusionmot.data_fusion import data_fusion
from deepfusionmot.DeepFusionMOT import DeepFusionMOT
from deepfusionmot.coordinate_transformation import convert_x1y1x2y2_to_tlwh
from deepfusionmot.config import Config # Configuration loader

class RealTimeFusionNode(Node): # ROS2 Node
    def __init__(self, cfg):
        super().__init__('real_time_fusion_node')  
        self.cfg = cfg  # Store configuration passed to the node

        # Read and store categories from config
        self.cat_list = self.cfg.get('cat_list', ['Car', 'Pedestrian'])
        self.get_logger().info(f"Configured categories: {self.cat_list}")

        # Initialize dictionaries to store trackers and detection data by category
        self.trackers = {}
        self.detections_2d = {}
        self.detections_3d = {}
        
        # Initialize trackers and detection arrays for each category
        for cat in self.cat_list:
            self.trackers[cat] = DeepFusionMOT(cfg, cat)
            self.detections_2d[cat] = np.empty((0, 6))
            self.detections_3d[cat] = np.empty((0, 15))
            self.get_logger().info(f"Initialized tracker and detections for category: {cat}")

        self.bridge = CvBridge()

        # Store calibration data from the /camera/calibration topic
        self.calib_data = None

        # --------------- Subscribers ---------------
        # 1) Camera image
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10) # Subscribe to camera images

        # 2) LiDAR
        self.create_subscription(PointCloud2, '/lidar/points', self.lidar_callback, 10)  # Subscribe to LiDAR data
        
        # Conditional subscriptions for 2D and 3D detections based on categories
        # 3) 2D Detections (Car/Ped)  
        if 'Car' in self.cat_list:
            self.create_subscription(Float32MultiArray, '/detection_2d/car', self.detection_2d_car_cb, 10) # 2D car detections
        if 'Pedestrian' in self.cat_list:
            self.create_subscription(Float32MultiArray, '/detection_2d/pedestrian', self.detection_2d_ped_cb, 10) # 2D pedestrian detections 

        # 4) 3D Detections (Car/Ped)  
        if 'Car' in self.cat_list:
            self.create_subscription(Float32MultiArray, '/detection_3d/car', self.detection_3d_car_cb, 10) # 3D car detections
        if 'Pedestrian' in self.cat_list:
            self.create_subscription(Float32MultiArray, '/detection_3d/pedestrian', self.detection_3d_ped_cb, 10)# 3D pedestrian detections

        # 5) Calibration data as a std_msgs/String
        self.create_subscription(String, '/camera/calibration', self.calibration_cb, 10) # Subscribe to calibration data

        # --------------- Publishers ---------------
        self.annotated_image_pub = self.create_publisher(Image, '/annotated_image', 10) # Publisher for annotated images
        self.raw_detections_image_pub = self.create_publisher(Image, '/raw_detections_image', 10) # Publisher for annotated images

        self.get_logger().info("Node is initialized. Waiting for data")

    # ------------------ Calibration Callback ---------------------
    def calibration_cb(self, msg: String):
        """
        This callback is triggered when the /camera/calibration topic publishes a string.
        It parses the calibration string and stores the relevant matrices in self.calib_data.
        """
        self.get_logger().info("Calibration callback triggered!")
        calib_str = msg.data
        self.calib_data = self.parse_calib_string(calib_str)  # Process and store calibration data
        self.get_logger().info("Received and parsed calibration data from /camera/calibration.")
        
    def parse_calib_string(self, calib_str):
        """
        Parses the raw calibration string into a dictionary of NumPy arrays.
        Example calibration string format:
            P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 ...
            P1: ...
            P2: ...
            R_rect: ...
            Tr_velo_cam: ...
        """
        lines = calib_str.strip().split('\n')
        calib_dict = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Split by ":", then parse numbers
            if ':' not in line:
                # Skip lines without key:value format
                continue

            key, vals = line.split(':', 1)
            key = key.strip()  # e.g., "P0"
            vals = vals.strip()  # e.g., "7.215377e+02 0.000000..."

            # Parse the numbers
            num_strs = vals.split()
            try:
                floats = [float(x) for x in num_strs]
            except ValueError as e:
                self.get_logger().error(f"Error parsing calibration values for {key}: {e}")
                continue

            # Store as NumPy arrays, reshaping based on key
            if key.startswith('P'):
                # "P0", "P1", "P2", "P3" => assume 3x4
                if len(floats) == 12:
                    matrix = np.array(floats).reshape(3, 4)
                    calib_dict[key] = matrix
                else:
                    calib_dict[key] = np.array(floats)
            elif key.startswith('R_rect'):
                # "R_rect" => assume 3x3
                if len(floats) == 9:
                    matrix = np.array(floats).reshape(3, 3)
                    calib_dict[key] = matrix
                else:
                    calib_dict[key] = np.array(floats)
            elif key.startswith('Tr_'):
                # "Tr_velo_cam" or similar => assume 3x4
                if len(floats) == 12:
                    matrix = np.array(floats).reshape(3, 4)
                    calib_dict[key] = matrix
                else:
                    calib_dict[key] = np.array(floats)
            else:
                # Unknown key, store as is
                calib_dict[key] = np.array(floats)

        return calib_dict

    # ------------------ Image & LiDAR Callbacks ---------------------
    def image_callback(self, msg: Image):
        """Triggered on new camera image."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  # Convert ROS image to OpenCV format
            self.get_logger().info(f"Received an image")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # 1) Publish the raw 2D detections
        self.publish_raw_detections_image()
        # 2) Then run the fusion + tracking + 3D-annotated image
        self.process_frame()  # Process the latest frame

    def lidar_callback(self, msg: PointCloud2):
        """Triggered on new LiDAR point cloud."""
        self.latest_pc = msg # Store latest point cloud data
        self.get_logger().info("Received a new point cloud.")

    # ------------------ Detection Callbacks ---------------------
    def detection_2d_car_cb(self, msg: Float32MultiArray):
        try:
            arr = np.array(msg.data).reshape(-1, 6) # Handle 2D car detections
            self.detections_2d['Car'] = arr
            self.get_logger().info(f"Received {arr.shape[0]} 2D Car detections: {arr}")
        except Exception as e:
            self.get_logger().error(f"Error processing 2D Car detections: {e}")

    def detection_2d_ped_cb(self, msg: Float32MultiArray):
        try:
            arr = np.array(msg.data).reshape(-1, 6) # Handle 2D pedestrian detections
            self.detections_2d['Pedestrian'] = arr
            self.get_logger().info(f"Received {arr.shape[0]} 2D Pedestrian detections: {arr}")
        except Exception as e:
            self.get_logger().error(f"Error processing 2D Pedestrian detections: {e}")

    def detection_3d_car_cb(self, msg: Float32MultiArray):
        try:
            arr = np.array(msg.data).reshape(-1, 15) # Handle 3D car detections
            self.detections_3d['Car'] = arr
            self.get_logger().info(f"Received {arr.shape[0]} 3D Car detections: {arr}")
        except Exception as e:
            self.get_logger().error(f"Error processing 3D Car detections: {e}")

    def detection_3d_ped_cb(self, msg: Float32MultiArray):
        try:
            arr = np.array(msg.data).reshape(-1, 15) # Handle 2D pedestrian detections
            self.detections_3d['Pedestrian'] = arr
            self.get_logger().info(f"Received {arr.shape[0]} 3D Pedestrian detections: {arr}")
        except Exception as e:
            self.get_logger().error(f"Error processing 3D Pedestrian detections: {e}")

    # ------------------ Main Processing ---------------------
    def process_frame(self):
        """Perform data fusion + tracking and publish results."""
        if self.latest_image is None:
            self.get_logger().info("No image yet")
            return

        # If we haven't gotten calibration yet
        if self.calib_data is None:
            self.get_logger().warn("No camera calibration yet")
            return

        combined_trackers = []

        # Iterate over each category in cat_list
        for cat in self.cat_list:
            # Ensure that both 2D and 3D detections are available for the category
            dets_2d = self.detections_2d.get(cat, np.empty((0, 6)))
            dets_3d = self.detections_3d.get(cat, np.empty((0, 15)))

            # Category-level processing
            trackers = self.run_fusion_and_update(
                self.trackers[cat],
                dets_2d,
                dets_3d,
                cat
            )

            if trackers is not None and trackers.size > 0:
                # Prefix track IDs with category to ensure uniqueness
                trackers[:, 0] = self.assign_unique_ids(cat, trackers[:, 0])
                combined_trackers.append(trackers)

        # Combine all trackers into a single array
        if combined_trackers:
            combined_trackers = np.vstack(combined_trackers)
        else:
            combined_trackers = np.empty((0, 10))  # Adjust shape as needed

        self.get_logger().info(f"Combined trackers: {combined_trackers.shape[0]} objects total.")

        # Publish the 3D-projected image with bounding boxes & track IDs
        self.publish_3d_annotated_image(combined_trackers)

    def run_fusion_and_update(self, tracker, dets_2d, dets_3d, category_name):
        """Fuse 2D & 3D, then update the specified tracker."""
        num_2d = dets_2d.shape[0]
        num_3d = dets_3d.shape[0]
        self.get_logger().info(f"Running fusion/update for '{category_name}' with {num_2d} 2D and {num_3d} 3D detections.")

        if num_2d == 0 and num_3d == 0:
            return np.empty((0, 10))

        # Basic parsing (adapt to your detection format!)
        dets_3d_camera = dets_3d[:, 7:14] if dets_3d.shape[1] >= 14 else np.empty((0, 7))
        ori_array = dets_3d[:, -1].reshape((-1, 1)) if dets_3d.shape[1] >= 15 else np.empty((0, 1))
        other_array = dets_3d[:, 1:7] if dets_3d.shape[1] >= 7 else np.empty((0, 6))
        additional_info = (
            np.concatenate((ori_array, other_array), axis=1)
            if (len(ori_array) > 0 and len(other_array) > 0)
            else np.empty((0, 7))
        )
        dets_3dto2d_image = dets_3d[:, 2:6] if dets_3d.shape[1] >= 6 else np.empty((0, 4))

        # 2D bounding boxes
        dets_2d_frame = dets_2d[:, 1:5] if dets_2d.shape[1] >= 5 else np.empty((0, 4))

        # data_fusion
        dets_fusion, dets_only_3d, dets_only_2d = data_fusion(
            dets_3d_camera,
            dets_2d_frame,
            dets_3dto2d_image,
            additional_info
        )
        self.get_logger().info(
            f"Fusion results for {category_name}: "
            f"{len(dets_fusion['dets_3d_fusion'])} fused, "
            f"{len(dets_only_2d)} 2D-only, "
            f"{len(dets_only_3d['dets_3d_only'])} 3D-only."
        )

        # Convert 2D-only to top-left format
        dets_only_2d_tlwh = np.array([convert_x1y1x2y2_to_tlwh(i) for i in dets_only_2d])

        # Update the tracker
        trackers_output = tracker.update(dets_fusion, dets_only_2d_tlwh, dets_only_3d, self.cfg, 0, 0)
        if trackers_output is not None and trackers_output.size > 0:
            self.get_logger().info(f"{category_name} tracker output shape: {trackers_output.shape}.")
        else:
            self.get_logger().info(f"{category_name} tracker output is empty.")

        return trackers_output

    def assign_unique_ids(self, category, ids):
        """
        Prefix IDs with category to ensure uniqueness across categories.
        Since IDs are integers, we'll convert them to floats where the integer part represents the category.
        For example:
            - Category 'Car' might be assigned a unique integer, say 1.
            - Category 'Pedestrian' might be assigned 2.
            - ID 3 for 'Car' becomes 1003.0
            - ID 4 for 'Pedestrian' becomes 2004.0
        """
        category_mapping = {cat: idx+1 for idx, cat in enumerate(self.cat_list)}
        prefix = category_mapping.get(category, 0) * 1000  # Adjust multiplier as needed

        unique_ids = prefix + ids
        return unique_ids

    # ------------------ Image Publishing ---------------------
    def publish_raw_detections_image(self):
        """Publish an image containing only the raw 2D detections (no tracking)."""
        if self.latest_image is None:
            return

        img_raw = self.latest_image.copy()

        # Iterate over each category in cat_list
        for cat in self.cat_list:
            detections = self.detections_2d.get(cat, np.empty((0, 6)))
            color = (0, 255, 0) if cat == 'Car' else (255, 0, 0)  # Customize colors as needed
            label = "Car" if cat == 'Car' else "Pedestrian"

            for det in detections:
                try:
                    x1, y1, x2, y2 = map(int, det[1:5])
                    cv2.rectangle(img_raw, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_raw, label, (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except Exception as e:
                    self.get_logger().error(f"Error drawing detections for {cat}: {e}")

        # Convert and publish
        try:
            raw_msg = self.bridge.cv2_to_imgmsg(img_raw, encoding='bgr8')
            raw_msg.header.stamp = self.get_clock().now().to_msg()
            raw_msg.header.frame_id = "camera_link"
            self.raw_detections_image_pub.publish(raw_msg)
            self.get_logger().debug("Published raw detections image.")
        except Exception as e:
            self.get_logger().error(f"Failed to publish raw detections image: {e}")

    def publish_3d_annotated_image(self, trackers):
        """
        1) For each tracker row => parse [ID, h, w, l, x, y, z, theta, ...].
        2) Project the 3D box into the image based on self.calib_data.
        3) Draw the lines + text label.
        4) Publish the final annotated image as a ROS topic.
        """
        if self.latest_image is None:
            return

        # If no calibration data yet, skip
        if self.calib_data is None:
            self.get_logger().warn("No camera calibration yet, cannot project 3D boxes.")
            return

        annotated = self.latest_image.copy()

        for row in trackers:
            if len(row) < 9:
                continue  # Need at least ID + 3D box + orientation
            track_id = int(row[0])
            h, w, l = row[1], row[2], row[3]
            x, y, z = row[4], row[5], row[6]
            theta = row[7]
            # row[8] might be orientation or confidence, etc.

            bbox3d_tmp = np.array([h, w, l, x, y, z, theta], dtype=np.float32)
            color = self.compute_color_for_id(track_id)
            label_str = f"ID:{track_id}"

            annotated = self.show_image_with_boxes_3d(
                annotated,
                bbox3d_tmp,
                color,
                label_str
            )

        # Finally, publish the annotated image
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            annotated_msg.header.stamp = self.get_clock().now().to_msg()
            annotated_msg.header.frame_id = "camera_link"
            self.annotated_image_pub.publish(annotated_msg)
            self.get_logger().debug("Published 3D-annotated image.")
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")

    # ------------------ 3D Projection Utility ---------------------
    def show_image_with_boxes_3d(self, img, bbox3d_tmp, color=(255,255,255), label_str="", line_thickness=2):
        """
        Create 8 corners in 3D space, transform + project them with your stored calibration,
        then draw lines & label on the image.
        """
        corners_2d = self.project_3d_box(bbox3d_tmp)
        if corners_2d is not None and corners_2d.shape == (8, 2):
            img = self.draw_projected_box3d(img, corners_2d, color, line_thickness)

            # Draw label near corner #4
            c1 = (int(corners_2d[4, 0]), int(corners_2d[4, 1]))
            tf = max(line_thickness - 1, 1)  # font thickness
            t_size = cv2.getTextSize(str(label_str), 0, fontScale=line_thickness / 3, thickness=tf)[0]
            c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, str(label_str), (c1[0], c1[1] - 2),
                        0, line_thickness / 3, (225, 255, 255),
                        thickness=tf, lineType=cv2.LINE_AA)
        return img

    def draw_projected_box3d(self, image, qs, color=(255,255,255), thickness=2):
        """Draw lines between the 8 corners (qs) of the 3D box."""
        qs = qs.astype(int)
        for k in range(4):
            i, j = k, (k + 1) % 4
            image = cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k+4, ((k+1) % 4) + 4
            image = cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k, k+4
            image = cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        return image

    def project_3d_box(self, box3d_tmp):
        """
        box3d_tmp = (h, w, l, x, y, z, theta).
        Projects the 3D bounding box into the 2D image using the calibration data.
        Returns:
            corners_2d: NumPy array of shape (8, 2) containing the 2D pixel coordinates.
            Returns None if projection fails (e.g., box is behind the camera).
        """
        # Attempt to use 'P2', fallback to 'P0' if 'P2' is unavailable
        if 'P2' in self.calib_data:
            P = self.calib_data['P2']
        elif 'P0' in self.calib_data:
            P = self.calib_data['P0']
            self.get_logger().warn("Using 'P0' matrix as 'P2' was not found.")
        else:
            self.get_logger().warn("No suitable projection matrix found (P0/P2). Skipping.")
            return None

        h, w, l, x, y, z, theta = box3d_tmp
        # Build 8 corners in local coordinates (KITTI style)
        corner_x = [l/2, l/2, -l/2, -l/2, l/2,  l/2,  -l/2, -l/2]
        corner_y = [0,    0,    0,     0,   -h,   -h,   -h,   -h ]
        corner_z = [w/2, -w/2, -w/2,  w/2, w/2, -w/2, -w/2,  w/2]

        corners_3D = np.vstack([corner_x, corner_y, corner_z]).T  # (8,3)

        # Rotation around Y axis
        R_y = np.array([
            [ math.cos(theta), 0, math.sin(theta)],
            [ 0,               1, 0              ],
            [-math.sin(theta), 0, math.cos(theta)]
        ])
        corners_3D = corners_3D @ R_y.T

        # Translate
        corners_3D[:, 0] += x
        corners_3D[:, 1] += y
        corners_3D[:, 2] += z

        # Project
        corners_3D_hom = np.hstack((corners_3D, np.ones((8,1))))
        corners_2D = corners_3D_hom @ P.T  # shape (8,3)

        # Normalize
        with np.errstate(divide='ignore', invalid='ignore'):
            corners_2D[:, 0] /= corners_2D[:, 2]
            corners_2D[:, 1] /= corners_2D[:, 2]

        # Check for any corners behind the camera
        if np.any(corners_2D[:, 2] < 0.1):
            self.get_logger().debug("Some corners are behind the camera. Skipping this box.")
            return None

        return corners_2D[:, :2]

    # ------------------ Unique ID Assignment ---------------------
    def assign_unique_ids(self, category, ids):
        """
        Prefix IDs with category to ensure uniqueness across categories.
        For example:
            - Category 'Car' is assigned a prefix, e.g., 1000
            - Category 'Pedestrian' is assigned a prefix, e.g., 2000
            - ID 1 for 'Car' becomes 1001
            - ID 1 for 'Pedestrian' becomes 2001
        """
        category_mapping = {cat: idx+1 for idx, cat in enumerate(self.cat_list)}  # e.g., {'Car':1, 'Pedestrian':2}
        prefix = category_mapping.get(category, 0) * 1000  # Adjust multiplier as needed

        unique_ids = prefix + ids
        return unique_ids

    # ------------------ Image Publishing ---------------------
    def publish_raw_detections_image(self):
        """Publish an image containing only the raw 2D detections (no tracking)."""
        if self.latest_image is None:
            return

        img_raw = self.latest_image.copy()

        # Iterate over each category in cat_list
        for cat in self.cat_list:
            detections = self.detections_2d.get(cat, np.empty((0, 6)))
            color = (0, 255, 0) if cat == 'Car' else (255, 0, 0)  # Customize colors as needed
            label = "Car" if cat == 'Car' else "Pedestrian"

            for det in detections:
                try:
                    x1, y1, x2, y2 = map(int, det[1:5])
                    cv2.rectangle(img_raw, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_raw, label, (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except Exception as e:
                    self.get_logger().error(f"Error drawing detections for {cat}: {e}")

        # Convert and publish
        try:
            raw_msg = self.bridge.cv2_to_imgmsg(img_raw, encoding='bgr8')
            raw_msg.header.stamp = self.get_clock().now().to_msg()
            raw_msg.header.frame_id = "camera_link"
            self.raw_detections_image_pub.publish(raw_msg)
            self.get_logger().debug("Published raw detections image.")
        except Exception as e:
            self.get_logger().error(f"Failed to publish raw detections image: {e}")

    def publish_3d_annotated_image(self, trackers):
        """
        1) For each tracker row => parse [ID, h, w, l, x, y, z, theta, ...].
        2) Project the 3D box into the image based on self.calib_data.
        3) Draw the lines + text label.
        4) Publish the final annotated image as a ROS topic.
        """
        if self.latest_image is None:
            return

        # If no calibration data yet, skip
        if self.calib_data is None:
            self.get_logger().warn("No camera calibration yet, cannot project 3D boxes.")
            return

        annotated = self.latest_image.copy()

        for row in trackers:
            if len(row) < 9:
                continue  # Need at least ID + 3D box + orientation
            track_id = int(row[0])
            h, w, l = row[1], row[2], row[3]
            x, y, z = row[4], row[5], row[6]
            theta = row[7]
            # row[8] might be orientation or confidence, etc.

            bbox3d_tmp = np.array([h, w, l, x, y, z, theta], dtype=np.float32)
            color = self.compute_color_for_id(track_id)
            label_str = f"ID:{track_id}"

            annotated = self.show_image_with_boxes_3d(
                annotated,
                bbox3d_tmp,
                color,
                label_str
            )

        # Finally, publish the annotated image
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            annotated_msg.header.stamp = self.get_clock().now().to_msg()
            annotated_msg.header.frame_id = "camera_link"
            self.annotated_image_pub.publish(annotated_msg)
            self.get_logger().debug("Published 3D-annotated image.")
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")

    # ------------------ 3D Projection Utility ---------------------
    def show_image_with_boxes_3d(self, img, bbox3d_tmp, color=(255,255,255), label_str="", line_thickness=2):
        """
        Create 8 corners in 3D space, transform + project them with your stored calibration,
        then draw lines & label on the image.
        """
        corners_2d = self.project_3d_box(bbox3d_tmp)
        if corners_2d is not None and corners_2d.shape == (8, 2):
            img = self.draw_projected_box3d(img, corners_2d, color, line_thickness)

            # Draw label near corner #4
            c1 = (int(corners_2d[4, 0]), int(corners_2d[4, 1]))
            tf = max(line_thickness - 1, 1)  # font thickness
            t_size = cv2.getTextSize(str(label_str), 0, fontScale=line_thickness / 3, thickness=tf)[0]
            c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, str(label_str), (c1[0], c1[1] - 2),
                        0, line_thickness / 3, (225, 255, 255),
                        thickness=tf, lineType=cv2.LINE_AA)
        return img

    def draw_projected_box3d(self, image, qs, color=(255,255,255), thickness=2):
        """Draw lines between the 8 corners (qs) of the 3D box."""
        qs = qs.astype(int)
        for k in range(4):
            i, j = k, (k + 1) % 4
            image = cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k+4, ((k+1) % 4) + 4
            image = cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k, k+4
            image = cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        return image

    def project_3d_box(self, box3d_tmp):
        """
        box3d_tmp = (h, w, l, x, y, z, theta).
        Projects the 3D bounding box into the 2D image using the calibration data.
        Returns:
            corners_2d: NumPy array of shape (8, 2) containing the 2D pixel coordinates.
            Returns None if projection fails (e.g., box is behind the camera).
        """
        # Attempt to use 'P2', fallback to 'P0' if 'P2' is unavailable
        if 'P2' in self.calib_data:
            P = self.calib_data['P2']
        elif 'P0' in self.calib_data:
            P = self.calib_data['P0']
            self.get_logger().warn("Using 'P0' matrix as 'P2' was not found.")
        else:
            self.get_logger().warn("No suitable projection matrix found (P0/P2). Skipping.")
            return None

        h, w, l, x, y, z, theta = box3d_tmp
        # Build 8 corners in local coordinates (KITTI style)
        corner_x = [l/2, l/2, -l/2, -l/2, l/2,  l/2,  -l/2, -l/2]
        corner_y = [0,    0,    0,     0,   -h,   -h,   -h,   -h ]
        corner_z = [w/2, -w/2, -w/2,  w/2, w/2, -w/2, -w/2,  w/2]

        corners_3D = np.vstack([corner_x, corner_y, corner_z]).T  # (8,3)

        # Rotation around Y axis
        R_y = np.array([
            [ math.cos(theta), 0, math.sin(theta)],
            [ 0,               1, 0              ],
            [-math.sin(theta), 0, math.cos(theta)]
        ])
        corners_3D = corners_3D @ R_y.T

        # Translate
        corners_3D[:, 0] += x
        corners_3D[:, 1] += y
        corners_3D[:, 2] += z

        # Project
        corners_3D_hom = np.hstack((corners_3D, np.ones((8,1))))
        corners_2D = corners_3D_hom @ P.T  # shape (8,3)

        # Normalize
        with np.errstate(divide='ignore', invalid='ignore'):
            corners_2D[:, 0] /= corners_2D[:, 2]
            corners_2D[:, 1] /= corners_2D[:, 2]

        # Check for any corners behind the camera
        if np.any(corners_2D[:, 2] < 0.1):
            self.get_logger().debug("Some corners are behind the camera. Skipping this box.")
            return None

        return corners_2D[:, :2]

    # ------------------ Unique ID Assignment ---------------------
    def assign_unique_ids(self, category, ids):
        """
        Prefix IDs with category to ensure uniqueness across categories.
        For example:
            - Category 'Car' is assigned a prefix, e.g., 1000
            - Category 'Pedestrian' is assigned a prefix, e.g., 2000
            - ID 1 for 'Car' becomes 1001
            - ID 1 for 'Pedestrian' becomes 2001
        """
        category_mapping = {cat: idx+1 for idx, cat in enumerate(self.cat_list)}  # e.g., {'Car':1, 'Pedestrian':2}
        prefix = category_mapping.get(category, 0) * 1000  # Adjust multiplier as needed

        unique_ids = prefix + ids
        return unique_ids

    # ------------------ Misc Helpers ---------------------
    def compute_color_for_id(self, idx):
        """
        Generates a unique color for each tracker ID.
        """
        palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
        color = [int((p * (idx**2 - idx + 1)) % 255) for p in palette]
        return tuple(color)

def main(args=None):
    config_file = '/home/prabuddhi/ros2_ws1/src/deepfusionmot/config/kitti_real_time.yaml' # Path of the config file
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

