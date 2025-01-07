#!/usr/bin/env python3
import os
import math
import numpy as np
import cv2
import time

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

# ROS 2 message imports
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32MultiArray, String

# Message filters for synchronization
import message_filters

# Local/Project imports
from deepfusionmot.data_fusion import data_fusion
from deepfusionmot.DeepFusionMOT import DeepFusionMOT
from deepfusionmot.coordinate_transformation import convert_x1y1x2y2_to_tlwh

class RealTimeFusionNode(Node):
    def __init__(self, cfg):
        super().__init__('real_time_fusion_node')
        self.cfg = cfg

        # Read categories from config
        self.cat_list = self.cfg.get('cat_list', ['Car', 'Pedestrian'])
        self.get_logger().info(f"Configured categories: {self.cat_list}")

        # Initialize trackers and detections dictionaries
        self.trackers = {}
        self.detections_2d = {cat: np.empty((0, 6)) for cat in self.cat_list}
        self.detections_3d = {cat: np.empty((0, 15)) for cat in self.cat_list}

        for cat in self.cat_list:
            self.trackers[cat] = DeepFusionMOT(cfg, cat)
            self.get_logger().info(f"Initialized tracker for category: {cat}")

        self.bridge = CvBridge()

        # Store calibration data from the /camera/calibration topic
        self.calib_data = None

        # --------------- Subscribers with message_filters ---------------
        image_sub = message_filters.Subscriber(self, Image, '/camera/image_raw')
        lidar_sub = message_filters.Subscriber(self, PointCloud2, '/lidar/points')
        det_2d_car_sub = message_filters.Subscriber(self, Float32MultiArray, '/detection_2d/car')
        det_2d_ped_sub = message_filters.Subscriber(self, Float32MultiArray, '/detection_2d/pedestrian')
        det_3d_car_sub = message_filters.Subscriber(self, Float32MultiArray, '/detection_3d/car')
        det_3d_ped_sub = message_filters.Subscriber(self, Float32MultiArray, '/detection_3d/pedestrian')
        calib_sub = message_filters.Subscriber(self, String, '/camera/calibration')

        # ApproximateTimeSynchronizer with allow_headerless=True
        ats = message_filters.ApproximateTimeSynchronizer(
            [image_sub, lidar_sub, det_2d_car_sub, det_2d_ped_sub, det_3d_car_sub, det_3d_ped_sub, calib_sub],
            queue_size=10,
            slop=0.1,             # Adjust based on your data's timestamp precision
            allow_headerless=True # Enable handling of headerless messages
        )
        ats.registerCallback(self.synced_callback)

        # --------------- Publishers ---------------
        self.annotated_image_pub = self.create_publisher(Image, '/annotated_image', 10)
        self.raw_detections_image_pub = self.create_publisher(Image, '/raw_detections_image', 10)

        self.get_logger().info("RealTimeFusionNode initialized. Waiting for synchronized data...")

    def synced_callback(self, image_msg, lidar_msg, det2d_car_msg, det2d_ped_msg, det3d_car_msg, det3d_ped_msg, calib_msg):
        # Calculate the frame time based on image timestamp
        frame_time = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9
        self.get_logger().debug(f"Synchronized Callback at time: {frame_time}")

        # Parse calibration data
        self.calibration_cb(calib_msg)

        # Parse image
        try:
            img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            self.get_logger().debug(f"Image received with shape: {img.shape}")
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        # Parse LiDAR data (if needed)
        point_cloud = lidar_msg  # Further processing as needed
        self.get_logger().debug("LiDAR data received.")

        # Parse detections
        detections_2d_car = self.parse_detections(det2d_car_msg, category='Car')
        detections_2d_ped = self.parse_detections(det2d_ped_msg, category='Pedestrian')
        detections_3d_car = self.parse_detections(det3d_car_msg, category='Car', is_3d=True)
        detections_3d_ped = self.parse_detections(det3d_ped_msg, category='Pedestrian', is_3d=True)

        # Process frame with synchronized data
        self.process_frame(img, point_cloud, detections_2d_car, detections_2d_ped, detections_3d_car, detections_3d_ped)

    def parse_detections(self, msg, category, is_3d=False):
        """
        Parse detection messages by removing the frame_idx prepended in the detection data.
        """
        try:
            if is_3d:
                # Reshape to (-1, 16) and remove the first column (frame_idx)
                detections = np.array(msg.data).reshape(-1, 16)[:, 1:16]
                self.get_logger().debug(f"Parsed {detections.shape[0]} 3D {category} detections.")
            else:
                # Reshape to (-1, 7) and remove the first column (frame_idx)
                detections = np.array(msg.data).reshape(-1, 7)[:, 1:7]
                self.get_logger().debug(f"Parsed {detections.shape[0]} 2D {category} detections.")
            return detections
        except Exception as e:
            self.get_logger().error(f"Detection parsing error for {category}: {e}")
            return np.empty((0, 15)) if is_3d else np.empty((0, 6))

    def calibration_cb(self, msg: String):
        """
        Parse and store calibration data.
        """
        calib_str = msg.data
        self.calib_data = self.parse_calib_string(calib_str)
        self.get_logger().debug("Calibration data updated.")

    def parse_calib_string(self, calib_str):
        """
        Parses the raw calibration string into a dictionary of NumPy arrays.
        """
        lines = calib_str.strip().split('\n')
        calib_dict = {}
        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, vals = line.split(':', 1)
            key = key.strip()
            vals = vals.strip()
            num_strs = vals.split()
            try:
                floats = [float(x) for x in num_strs]
            except ValueError as e:
                self.get_logger().error(f"Error parsing calibration values for {key}: {e}")
                continue
            if key.startswith('P'):
                matrix = np.array(floats).reshape(3, 4) if len(floats) == 12 else np.array(floats)
                calib_dict[key] = matrix
            elif key.startswith('R_rect'):
                matrix = np.array(floats).reshape(3, 3) if len(floats) == 9 else np.array(floats)
                calib_dict[key] = matrix
            elif key.startswith('Tr_'):
                matrix = np.array(floats).reshape(3, 4) if len(floats) == 12 else np.array(floats)
                calib_dict[key] = matrix
            else:
                calib_dict[key] = np.array(floats)
        return calib_dict

    def process_frame(self, img, point_cloud, dets_2d_car, dets_2d_ped, dets_3d_car, dets_3d_ped):
        """
        Perform data fusion, tracking, and publish annotated images.
        """
        if self.calib_data is None:
            self.get_logger().warn("Calibration data not available. Skipping frame processing.")
            return

        combined_trackers = []

        start_time = time.time()

        for category in self.cat_list:
            dets_2d = dets_2d_car if category == 'Car' else dets_2d_ped
            dets_3d = dets_3d_car if category == 'Car' else dets_3d_ped

            # Perform data fusion
            dets_fusion, dets_only_3d, dets_only_2d = data_fusion(
                dets_3d[:, 7:14] if dets_3d.size else np.empty((0, 7)),
                dets_2d[:, 1:5] if dets_2d.size else np.empty((0, 4)),
                dets_3d[:, 2:6] if dets_3d.size else np.empty((0, 4)),
                dets_3d[:, [14, 1, 2, 3, 4, 5, 6]] if dets_3d.size else np.empty((0, 7))
            )

            self.get_logger().debug(
                f"Category '{category}': {len(dets_fusion)} fused, {len(dets_only_2d)} 2D-only, {len(dets_only_3d)} 3D-only detections."
            )

            # Convert 2D-only detections to TLWH format
            dets_only_2d_tlwh = np.array([convert_x1y1x2y2_to_tlwh(det) for det in dets_only_2d])

            # Update tracker
            trackers_output = self.trackers[category].update(
                dets_fusion,
                dets_only_2d_tlwh,
                dets_only_3d,
                self.cfg,
                frame=0,  # Modify if frame index is available
                seq_id=0  # Modify if sequence ID is available
            )

            if trackers_output is not None and trackers_output.size > 0:
                # Assign unique IDs
                trackers_output[:, 0] = self.assign_unique_ids(category, trackers_output[:, 0])
                combined_trackers.append(trackers_output)

        # Combine all trackers
        if combined_trackers:
            combined_trackers = np.vstack(combined_trackers)
        else:
            combined_trackers = np.empty((0, 10))  # Adjust shape as needed

        self.get_logger().debug(f"Combined trackers: {combined_trackers.shape[0]} objects.")

        # Publish results
        self.publish_3d_annotated_image(combined_trackers, img)

        processing_time = time.time() - start_time
        self.get_logger().info(f"Frame processed in {processing_time:.4f} seconds.")

    def assign_unique_ids(self, category, ids):
        """
        Prefix IDs with category to ensure uniqueness across categories.
        """
        category_mapping = {cat: idx + 1 for idx, cat in enumerate(self.cat_list)}
        prefix = category_mapping.get(category, 0) * 1000  # Example: Car=1000, Pedestrian=2000
        unique_ids = prefix + ids
        return unique_ids

    def publish_3d_annotated_image(self, trackers, img):
        """
        Draw 3D bounding boxes and IDs on the image and publish.
        """
        annotated = img.copy()

        for row in trackers:
            if len(row) < 9:
                continue  # Ensure sufficient data
            track_id = int(row[0])
            h, w, l = row[1], row[2], row[3]
            x, y, z = row[4], row[5], row[6]
            theta = row[7]

            bbox3d = np.array([h, w, l, x, y, z, theta], dtype=np.float32)
            color = self.compute_color_for_id(track_id)
            label = f"ID:{track_id}"

            annotated = self.draw_3d_box(annotated, bbox3d, color, label)

        # Publish annotated image
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            annotated_msg.header.stamp = self.get_clock().now().to_msg()
            annotated_msg.header.frame_id = "camera_link"
            self.annotated_image_pub.publish(annotated_msg)
            self.get_logger().debug("Published annotated image.")
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")

    def draw_3d_box(self, img, bbox3d, color, label, thickness=2):
        """
        Project 3D bounding box to 2D and draw on the image.
        """
        corners_2d = self.project_3d_box(bbox3d)
        if corners_2d is not None and corners_2d.shape == (8, 2):
            img = self.draw_projected_box3d(img, corners_2d, color, thickness)
            # Draw label near a specific corner
            c1 = (int(corners_2d[4, 0]), int(corners_2d[4, 1]))
            tf = max(thickness - 1, 1)  # Font thickness
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                     fontScale=thickness / 3, thickness=tf)[0]
            c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, thickness / 3, (225, 255, 255),
                        thickness=tf, lineType=cv2.LINE_AA)
        return img

    def draw_projected_box3d(self, image, qs, color=(255, 255, 255), thickness=2):
        """Draw lines between the 8 corners (qs) of the 3D box."""
        qs = qs.astype(int)
        for k in range(4):
            i, j = k, (k + 1) % 4
            image = cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k + 4, ((k + 1) % 4) + 4
            image = cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            image = cv2.line(image, (qs[k, 0], qs[k, 1]), (qs[k + 4, 0], qs[k + 4, 1]), color, thickness)
        return image

    def project_3d_box(self, bbox3d):
        """
        Projects the 3D bounding box into the 2D image using calibration data.
        Returns:
            corners_2d: NumPy array of shape (8, 2) containing the 2D pixel coordinates.
            Returns None if projection fails.
        """
        if 'P2' in self.calib_data:
            P = self.calib_data['P2']
        elif 'P0' in self.calib_data:
            P = self.calib_data['P0']
            self.get_logger().warn("Using 'P0' matrix as 'P2' was not found.")
        else:
            self.get_logger().warn("No suitable projection matrix found (P0/P2). Skipping box projection.")
            return None

        h, w, l, x, y, z, theta = bbox3d
        # Define 8 corners in the object's local coordinate system
        corners = np.array([
            [ l/2,  0,  w/2],
            [ l/2,  0, -w/2],
            [-l/2,  0, -w/2],
            [-l/2,  0,  w/2],
            [ l/2, -h,  w/2],
            [ l/2, -h, -w/2],
            [-l/2, -h, -w/2],
            [-l/2, -h,  w/2]
        ])

        # Rotation matrix around Y-axis
        R = np.array([
            [ math.cos(theta), 0, math.sin(theta)],
            [ 0,               1, 0             ],
            [-math.sin(theta), 0, math.cos(theta)]
        ])
        rotated_corners = corners @ R.T

        # Translate to global coordinates
        rotated_corners += np.array([x, y, z])

        # Convert to homogeneous coordinates
        corners_hom = np.hstack((rotated_corners, np.ones((8, 1))))
        corners_2d_hom = corners_hom @ P.T  # Shape: (8, 3)

        # Normalize to get pixel coordinates
        with np.errstate(divide='ignore', invalid='ignore'):
            corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2].reshape(-1, 1)

        # Check for valid projections
        if np.any(corners_2d_hom[:, 2] < 0.1):
            self.get_logger().debug("Some corners are behind the camera. Skipping box.")
            return None

        return corners_2d

    def compute_color_for_id(self, idx):
        """
        Generates a unique color for each tracker ID.
        """
        palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
        color = [int((p * (idx**2 - idx + 1)) % 255) for p in palette]
        return tuple(color)

def main(args=None):
    rclpy.init(args=args)

    from deepfusionmot.config import Config
    config_file = '/home/prabuddhi/ros2_ws1/src/deepfusionmot/config/kitti_real_time.yaml'
    cfg, _ = Config(config_file)

    node = RealTimeFusionNode(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down RealTimeFusionNode.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

