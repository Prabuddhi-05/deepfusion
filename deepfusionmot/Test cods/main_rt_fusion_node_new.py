#!/usr/bin/env python3
import os
import math
import numpy as np
import cv2
from collections import defaultdict
import threading

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

# ROS 2 message imports
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32MultiArray

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

        # Initialize dictionaries to store trackers for each category
        self.trackers = {}
        for cat in self.cat_list:
            self.trackers[cat] = DeepFusionMOT(cfg, cat)
            self.get_logger().info(f"Initialized tracker for category: {cat}")

        self.bridge = CvBridge()

        # Store frame data indexed by frame_idx
        self.frames_data = {}  # key: frame_idx, value: dict with 'image', 'point_cloud', 'dets_2d_car', 'dets_3d_car', etc.
        self.frames_lock = threading.Lock()  # To ensure thread-safe access to frames_data

        # Frame management
        self.frame_counter = 0  # To assign frame_idx to images and point clouds
        self.frame_timestamps = {}  # key: frame_idx, value: time when frame was first received
        self.frame_timeout_seconds = 1.0  # Timeout for processing frames with missing data

        # --------------- Subscribers ---------------
        # 1) Camera image
        self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # 2) LiDAR
        self.create_subscription(
            PointCloud2,
            '/lidar/points',
            self.lidar_callback,
            10
        )

        # Conditional subscriptions for 2D and 3D detections based on categories
        # 3) 2D Detections (Car/Ped)
        if 'Car' in self.cat_list:
            self.create_subscription(Float32MultiArray, '/detection_2d/car', self.detection_2d_car_cb, 10)  # 2D car detections
        if 'Pedestrian' in self.cat_list:
            self.create_subscription(Float32MultiArray, '/detection_2d/pedestrian', self.detection_2d_ped_cb, 10)  # 2D pedestrian detections 

        # 4) 3D Detections (Car/Ped)
        if 'Car' in self.cat_list:
            self.create_subscription(Float32MultiArray, '/detection_3d/car', self.detection_3d_car_cb, 10)  # 3D car detections
        if 'Pedestrian' in self.cat_list:
            self.create_subscription(Float32MultiArray, '/detection_3d/pedestrian', self.detection_3d_ped_cb, 10)  # 3D pedestrian detections

        # --------------- Publishers ---------------
        #self.annotated_image_pub = self.create_publisher(Image, '/annotated_image', 10) # Publisher for annotated images
        self.raw_detections_image_pub = self.create_publisher(Image, '/raw_detections_image', 10)  # Publisher for raw detections images

        # --------------- Timer for Processing Frames ---------------
        self.create_timer(
            0.05,  # 50 ms
            self.process_frames_timer_callback
        )

        self.get_logger().info("Node is initialized. Waiting for data")

    # ------------------ Image Callback ---------------------
    def image_callback(self, msg: Image):
        """Triggered on new camera image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  # Convert ROS image to OpenCV format
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        with self.frames_lock:
            frame_idx = self.frame_counter
            self.frames_data.setdefault(frame_idx, {})['image'] = cv_image
            self.frame_timestamps[frame_idx] = self.get_clock().now().nanoseconds
            self.get_logger().info(f"Received image for frame {frame_idx}")
            self.frame_counter += 1

    # ------------------ LiDAR Callback ---------------------
    def lidar_callback(self, msg: PointCloud2):
        """Triggered on new LiDAR point cloud."""
        with self.frames_lock:
            frame_idx = self.frame_counter - 1  # Assuming LiDAR corresponds to the latest image
            if frame_idx < 0:
                self.get_logger().warn("Received point cloud before any image. Ignoring.")
                return
            self.frames_data.setdefault(frame_idx, {})['point_cloud'] = msg
            self.get_logger().info(f"Received point cloud for frame {frame_idx}")

    # ------------------ 2D Car Detection Callback ---------------------
    def detection_2d_car_cb(self, msg: Float32MultiArray):
        """Callback for 2D Car detections."""
        try:
            # Each detection has 6 elements: [frame_idx, x1, y1, x2, y2, confidence]
            arr = np.array(msg.data).reshape(-1, 6)
            with self.frames_lock:
                for det in arr:
                    frame_idx = int(det[0])
                    detection = det[1:]  # [x1, y1, x2, y2, confidence]
                    self.frames_data.setdefault(frame_idx, {}).setdefault('dets_2d_car', []).append(detection)
                    self.get_logger().info(f"Received 2D Car detection for frame {frame_idx}: {detection.tolist()}")
        except Exception as e:
            self.get_logger().error(f"Error processing 2D Car detections: {e}")

    # ------------------ 2D Pedestrian Detection Callback ---------------------
    def detection_2d_ped_cb(self, msg: Float32MultiArray):
        """Callback for 2D Pedestrian detections."""
        try:
            # Each detection has 6 elements: [frame_idx, x1, y1, x2, y2, confidence]
            arr = np.array(msg.data).reshape(-1, 6)
            with self.frames_lock:
                for det in arr:
                    frame_idx = int(det[0])
                    detection = det[1:]  # [x1, y1, x2, y2, confidence]
                    self.frames_data.setdefault(frame_idx, {}).setdefault('dets_2d_ped', []).append(detection)
                    self.get_logger().info(f"Received 2D Pedestrian detection for frame {frame_idx}: {detection.tolist()}")
        except Exception as e:
            self.get_logger().error(f"Error processing 2D Pedestrian detections: {e}")

    # ------------------ 3D Car Detection Callback ---------------------
    def detection_3d_car_cb(self, msg: Float32MultiArray):
        """Callback for 3D Car detections."""
        try:
            # Each detection has 15 elements: [frame_idx, ...14 other elements...]
            arr = np.array(msg.data).reshape(-1, 15)
            with self.frames_lock:
                for det in arr:
                    frame_idx = int(det[0])
                    detection = det[1:]  # 14 elements excluding frame_idx
                    self.frames_data.setdefault(frame_idx, {}).setdefault('dets_3d_car', []).append(detection)
                    self.get_logger().info(f"Received 3D Car detection for frame {frame_idx}: {detection.tolist()}")
        except Exception as e:
            self.get_logger().error(f"Error processing 3D Car detections: {e}")

    # ------------------ 3D Pedestrian Detection Callback ---------------------
    def detection_3d_ped_cb(self, msg: Float32MultiArray):
        """Callback for 3D Pedestrian detections."""
        try:
            # Each detection has 15 elements: [frame_idx, ...14 other elements...]
            arr = np.array(msg.data).reshape(-1, 15)
            with self.frames_lock:
                for det in arr:
                    frame_idx = int(det[0])
                    detection = det[1:]  # 14 elements excluding frame_idx
                    self.frames_data.setdefault(frame_idx, {}).setdefault('dets_3d_ped', []).append(detection)
                    self.get_logger().info(f"Received 3D Pedestrian detection for frame {frame_idx}: {detection.tolist()}")
        except Exception as e:
            self.get_logger().error(f"Error processing 3D Pedestrian detections: {e}")

    # ------------------ Timer Callback for Processing Frames ---------------------
    def process_frames_timer_callback(self):
        """
        Periodically called to process frames that have all required data and have been buffered
        long enough to allow all detections to arrive.
        """
        with self.frames_lock:
            frames_to_process = []
            current_time = self.get_clock().now().nanoseconds
            for frame_idx, frame in list(self.frames_data.items()):
                # Check if the frame has all required data
                has_image = 'image' in frame
                has_point_cloud = 'point_cloud' in frame
                has_dets_2d_car = 'dets_2d_car' in frame
                has_dets_2d_ped = 'dets_2d_ped' in frame
                has_dets_3d_car = 'dets_3d_car' in frame
                has_dets_3d_ped = 'dets_3d_ped' in frame

                # Determine if all categories have detections
                has_dets_2d = True
                has_dets_3d = True
                for cat in self.cat_list:
                    if cat == 'Car' and not has_dets_2d_car:
                        has_dets_2d = False
                    if cat == 'Pedestrian' and not has_dets_2d_ped:
                        has_dets_2d = False
                    if cat == 'Car' and not has_dets_3d_car:
                        has_dets_3d = False
                    if cat == 'Pedestrian' and not has_dets_3d_ped:
                        has_dets_3d = False

                # Check if frame has all data or has timed out
                frame_age = (current_time - self.frame_timestamps.get(frame_idx, current_time)) / 1e9  # seconds
                if has_image and has_point_cloud and has_dets_2d and has_dets_3d:
                    frames_to_process.append(frame_idx)
                elif frame_age > self.frame_timeout_seconds:
                    self.get_logger().warn(f"Frame {frame_idx} timed out. Processing with available data.")
                    frames_to_process.append(frame_idx)

            for frame_idx in sorted(frames_to_process):
                self.get_logger().info(f"Processing frame {frame_idx}")
                self.process_frame(frame_idx)
                # Remove the frame data and timestamp to free memory
                del self.frames_data[frame_idx]
                del self.frame_timestamps[frame_idx]

        # Optional: Log the current state of frames_data
        self.get_logger().debug(f"Current frames_data keys: {list(self.frames_data.keys())}")

    # ------------------ Frame Processing ---------------------
    def process_frame(self, frame_idx):
        """
        Perform data fusion + tracking and publish results for the specified frame_idx.
        """
        frame = self.frames_data.get(frame_idx, {})
        if not frame:
            self.get_logger().warn(f"No data found for frame {frame_idx}. Skipping processing.")
            return

        image = frame.get('image')
        point_cloud = frame.get('point_cloud')

        if image is None:
            self.get_logger().warn(f"No image data for frame {frame_idx}. Skipping processing.")
            return

        combined_trackers = []

        # Iterate over each category in cat_list
        for cat in self.cat_list:
            # Retrieve detections for the current category and frame_idx
            dets_2d_key = f'dets_2d_{cat.lower()}'
            dets_3d_key = f'dets_3d_{cat.lower()}'

            dets_2d = np.array(frame.get(dets_2d_key, []))
            dets_3d = np.array(frame.get(dets_3d_key, []))

            # Check if detections have the expected number of columns
            expected_dets_3d_cols = 14  # Adjust based on your detection data structure
            if dets_3d.size > 0 and dets_3d.shape[1] != expected_dets_3d_cols:
                self.get_logger().error(
                    f"3D detections for category '{cat}' in frame {frame_idx} have incorrect number of columns: "
                    f"{dets_3d.shape[1]} expected {expected_dets_3d_cols}"
                )
                continue  # Skip processing for this category

            # Category-level processing
            trackers = self.run_fusion_and_update(
                self.trackers[cat],
                dets_2d,
                dets_3d,
                cat,
                frame_idx
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

        self.get_logger().info(f"Combined trackers for frame {frame_idx}: {combined_trackers.shape[0]} objects total.")

        # Publish the raw 2D detections
        self.publish_raw_detections_image(frame_idx)

        # Optionally, publish other processed data here

    def run_fusion_and_update(self, tracker, dets_2d, dets_3d, category_name, frame_idx):
        """Fuse 2D & 3D, then update the specified tracker."""
        num_2d = dets_2d.shape[0]
        num_3d = dets_3d.shape[0]
        self.get_logger().info(
            f"Running fusion/update for '{category_name}' with {num_2d} 2D and {num_3d} 3D detections for frame {frame_idx}."
        )

        if num_2d == 0 and num_3d == 0:
            return np.empty((0, 10))

        # Basic parsing (adapt to your detection format!)
        dets_3d_camera = dets_3d[:, 7:14] if dets_3d.shape[1] >= 14 else np.empty((0, 7))
        ori_array = dets_3d[:, -1].reshape((-1, 1)) if dets_3d.shape[1] >= 14 else np.empty((0, 1))
        other_array = dets_3d[:, 1:7] if dets_3d.shape[1] >= 7 else np.empty((0, 6))
        additional_info = (
            np.concatenate((ori_array, other_array), axis=1)
            if (len(ori_array) > 0 and len(other_array) > 0)
            else np.empty((0, 7))
        )
        dets_3dto2d_image = dets_3d[:, 2:6] if dets_3d.shape[1] >= 6 else np.empty((0, 4))

        # 2D bounding boxes
        dets_2d_frame = dets_2d[:, :4] if dets_2d.shape[1] >= 4 else np.empty((0, 4))

        # Check if additional_info is not empty before proceeding
        if additional_info.size == 0 and num_3d > 0:
            self.get_logger().error(
                f"Additional info for 3D detections in category '{category_name}' for frame {frame_idx} is empty. "
                f"Skipping fusion for this category."
            )
            return np.empty((0, 10))

        # data_fusion
        dets_fusion, dets_only_3d, dets_only_2d = data_fusion(
            dets_3d_camera,
            dets_2d_frame,
            dets_3dto2d_image,
            additional_info
        )
        self.get_logger().info(
            f"Fusion results for {category_name} in frame {frame_idx}: "
            f"{len(dets_fusion['dets_3d_fusion'])} fused, "
            f"{len(dets_only_2d)} 2D-only, "
            f"{len(dets_only_3d['dets_3d_only'])} 3D-only."
        )

        # Convert 2D-only to top-left format
        dets_only_2d_tlwh = np.array([convert_x1y1x2y2_to_tlwh(i) for i in dets_only_2d])

        # Update the tracker
        trackers_output = tracker.update(
            dets_fusion,
            dets_only_2d_tlwh,
            dets_only_3d,
            self.cfg,
            frame_idx,
            0
        )
        if trackers_output is not None and trackers_output.size > 0:
            self.get_logger().info(f"{category_name} tracker output shape: {trackers_output.shape}.")
        else:
            self.get_logger().info(f"{category_name} tracker output is empty.")

        return trackers_output

    def assign_unique_ids(self, category, ids):
        """
        Prefix IDs with category to ensure uniqueness across categories.
        Example:
            - Category 'Car' is assigned a prefix, e.g., 1000
            - Category 'Pedestrian' is assigned 2000
            - ID 1 for 'Car' becomes 1001
            - ID 1 for 'Pedestrian' becomes 2001
        """
        category_mapping = {cat: (idx + 1) * 1000 for idx, cat in enumerate(self.cat_list)}  # e.g., {'Car':1000, 'Pedestrian':2000}
        prefix = category_mapping.get(category, 0)  # Default prefix 0 if category not found

        unique_ids = prefix + ids
        return unique_ids

    # ------------------ Raw Detections Publishing ---------------------
    def publish_raw_detections_image(self, frame_idx):
        """Publish an image containing only the raw 2D detections (no tracking) for a specific frame."""
        frame = self.frames_data.get(frame_idx, {})
        image = frame.get('image')
        if image is None:
            self.get_logger().warn(f"No image data for frame {frame_idx}. Cannot publish raw detections image.")
            return

        img_raw = image.copy()

        # Iterate over each category in cat_list
        for cat in self.cat_list:
            detections = frame.get(f'dets_2d_{cat.lower()}', [])
            if not detections:
                continue
            color = (0, 255, 0) if cat == 'Car' else (255, 0, 0)  # Customize colors as needed
            label = "Car" if cat == 'Car' else "Pedestrian"

            for det in detections:
                try:
                    x1, y1, x2, y2 = map(int, det[:4])  # Assuming det = [x1, y1, x2, y2, confidence]
                    cv2.rectangle(img_raw, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        img_raw,
                        label,
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )
                except Exception as e:
                    self.get_logger().error(f"Error drawing detections for {cat}: {e}")

        # Convert and publish
        try:
            raw_msg = self.bridge.cv2_to_imgmsg(img_raw, encoding='bgr8')
            raw_msg.header.stamp = self.get_clock().now().to_msg()
            raw_msg.header.frame_id = "camera_link"
            self.raw_detections_image_pub.publish(raw_msg)
            self.get_logger().debug(f"Published raw detections image for frame {frame_idx}.")
        except Exception as e:
            self.get_logger().error(f"Failed to publish raw detections image for frame {frame_idx}: {e}")

# ------------------ Main Function ---------------------
def main(args=None):
    """
    Main function to initialize and spin the RealTimeFusionNode.
    """
    # Path to the configuration file
    config_file = '/home/prabuddhi/ros2_ws1/src/deepfusionmot/config/kitti_real_time.yaml'  

    rclpy.init(args=args)

    try:
        cfg, _ = Config(config_file)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return

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


#!/usr/bin/env python3
import os
import math
import numpy as np
import cv2
from collections import defaultdict
import threading

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
from deepfusionmot.config import Config  # Configuration loader

class RealTimeFusionNode(Node):  # ROS2 Node
    def __init__(self, cfg):
        super().__init__('real_time_fusion_node')  
        self.cfg = cfg  # Store configuration passed to the node

        # Read and store categories from config
        self.cat_list = self.cfg.get('cat_list', ['Car', 'Pedestrian'])
        self.get_logger().info(f"Configured categories: {self.cat_list}")

        # Initialize dictionaries to store trackers and detection data by category
        self.trackers = {}
        for cat in self.cat_list:
            self.trackers[cat] = DeepFusionMOT(cfg, cat)
            self.get_logger().info(f"Initialized tracker for category: {cat}")

        self.bridge = CvBridge()

        # Synchronization structures
        self.frames_data = {}  # key: frame_idx, value: dict with 'image', 'point_cloud', 'dets_2d_car', 'dets_3d_car', etc.
        self.frames_lock = threading.Lock()  # To ensure thread-safe access to frames_data

        # Frame management
        self.frame_counter = 0  # To assign frame_idx to images and point clouds
        self.frame_timestamps = {}  # key: frame_idx, value: time when frame was first received
        self.frame_timeout_seconds = 1.0  # Timeout for processing frames with missing data

        # --------------- Subscribers ---------------
        # 1) Camera image
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)  # Subscribe to camera images

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
        self.raw_detections_image_pub = self.create_publisher(Image, '/raw_detections_image', 10) # Publisher for raw detections images

        # --------------- Timer for Processing Frames ---------------
        self.create_timer(
            0.05,  # 50 ms
            self.process_frames_timer_callback
        )

        self.get_logger().info("Node is initialized. Waiting for data")

        # Calibration data
        self.calib_data = None

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

    # ------------------ Image Callback ---------------------
    def image_callback(self, msg: Image):
        """Triggered on new camera image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  # Convert ROS image to OpenCV format
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        with self.frames_lock:
            frame_idx = self.frame_counter
            self.frames_data.setdefault(frame_idx, {})['image'] = cv_image
            self.frame_timestamps[frame_idx] = self.get_clock().now().nanoseconds
            self.get_logger().info(f"Received image for frame {frame_idx}")
            self.frame_counter += 1

    # ------------------ LiDAR Callback ---------------------
    def lidar_callback(self, msg: PointCloud2):
        """Triggered on new LiDAR point cloud."""
        with self.frames_lock:
            frame_idx = self.frame_counter - 1  # Assuming LiDAR corresponds to the latest image
            if frame_idx < 0:
                self.get_logger().warn("Received point cloud before any image. Ignoring.")
                return
            self.frames_data.setdefault(frame_idx, {})['point_cloud'] = msg
            self.get_logger().info(f"Received point cloud for frame {frame_idx}")

    # ------------------ 2D Car Detection Callback ---------------------
    def detection_2d_car_cb(self, msg: Float32MultiArray):
        """Callback for 2D Car detections."""
        try:
            # Each detection has 6 elements: [frame_idx, x1, y1, x2, y2, confidence]
            arr = np.array(msg.data).reshape(-1, 6)
            with self.frames_lock:
                for det in arr:
                    frame_idx = int(det[0])
                    detection = det[1:]  # [x1, y1, x2, y2, confidence]
                    self.frames_data.setdefault(frame_idx, {}).setdefault('dets_2d_car', []).append(detection)
                    self.get_logger().info(f"Received 2D Car detection for frame {frame_idx}: {detection.tolist()}")
        except Exception as e:
            self.get_logger().error(f"Error processing 2D Car detections: {e}")

    # ------------------ 2D Pedestrian Detection Callback ---------------------
    def detection_2d_ped_cb(self, msg: Float32MultiArray):
        """Callback for 2D Pedestrian detections."""
        try:
            # Each detection has 6 elements: [frame_idx, x1, y1, x2, y2, confidence]
            arr = np.array(msg.data).reshape(-1, 6)
            with self.frames_lock:
                for det in arr:
                    frame_idx = int(det[0])
                    detection = det[1:]  # [x1, y1, x2, y2, confidence]
                    self.frames_data.setdefault(frame_idx, {}).setdefault('dets_2d_pedestrian', []).append(detection)
                    self.get_logger().info(f"Received 2D Pedestrian detection for frame {frame_idx}: {detection.tolist()}")
        except Exception as e:
            self.get_logger().error(f"Error processing 2D Pedestrian detections: {e}")

    # ------------------ 3D Car Detection Callback ---------------------
    def detection_3d_car_cb(self, msg: Float32MultiArray):
        """Callback for 3D Car detections."""
        try:
            # Each detection has 15 elements: [frame_idx, ...14 other elements...]
            arr = np.array(msg.data).reshape(-1, 15)
            with self.frames_lock:
                for det in arr:
                    frame_idx = int(det[0])
                    detection = det[1:]  # 14 elements excluding frame_idx
                    self.frames_data.setdefault(frame_idx, {}).setdefault('dets_3d_car', []).append(detection)
                    self.get_logger().info(f"Received 3D Car detection for frame {frame_idx}: {detection.tolist()}")
        except Exception as e:
            self.get_logger().error(f"Error processing 3D Car detections: {e}")

    # ------------------ 3D Pedestrian Detection Callback ---------------------
    def detection_3d_ped_cb(self, msg: Float32MultiArray):
        """Callback for 3D Pedestrian detections."""
        try:
            # Each detection has 15 elements: [frame_idx, ...14 other elements...]
            arr = np.array(msg.data).reshape(-1, 15)
            with self.frames_lock:
                for det in arr:
                    frame_idx = int(det[0])
                    detection = det[1:]  # 14 elements excluding frame_idx
                    self.frames_data.setdefault(frame_idx, {}).setdefault('dets_3d_pedestrian', []).append(detection)
                    self.get_logger().info(f"Received 3D Pedestrian detection for frame {frame_idx}: {detection.tolist()}")
        except Exception as e:
            self.get_logger().error(f"Error processing 3D Pedestrian detections: {e}")

    # ------------------ Timer Callback for Processing Frames ---------------------
    def process_frames_timer_callback(self):
        """
        Periodically called to process frames that have all required data and have been buffered
        long enough to allow all detections to arrive.
        """
        with self.frames_lock:
            frames_to_process = []
            current_time = self.get_clock().now().nanoseconds
            for frame_idx, frame in list(self.frames_data.items()):
                # Check if the frame has all required data
                has_image = 'image' in frame
                has_point_cloud = 'point_cloud' in frame

                # Determine if all categories have detections
                has_dets_2d = True
                has_dets_3d = True
                for cat in self.cat_list:
                    dets_2d_key = f'dets_2d_{cat.lower()}'
                    dets_3d_key = f'dets_3d_{cat.lower()}'
                    if cat == 'Car' and not frame.get(dets_2d_key):
                        has_dets_2d = False
                    if cat == 'Pedestrian' and not frame.get(dets_2d_key):
                        has_dets_2d = False
                    if cat == 'Car' and not frame.get(dets_3d_key):
                        has_dets_3d = False
                    if cat == 'Pedestrian' and not frame.get(dets_3d_key):
                        has_dets_3d = False

                # Check if frame has all data or has timed out
                frame_age = (current_time - self.frame_timestamps.get(frame_idx, current_time)) / 1e9  # seconds
                if has_image and has_point_cloud and has_dets_2d and has_dets_3d:
                    frames_to_process.append(frame_idx)
                elif frame_age > self.frame_timeout_seconds:
                    self.get_logger().warn(f"Frame {frame_idx} timed out. Processing with available data.")
                    frames_to_process.append(frame_idx)

            for frame_idx in sorted(frames_to_process):
                self.get_logger().info(f"Processing frame {frame_idx}")
                self.process_frame(frame_idx)
                # Remove the frame data and timestamp to free memory
                del self.frames_data[frame_idx]
                del self.frame_timestamps[frame_idx]

        # Optional: Log the current state of frames_data
        self.get_logger().debug(f"Current frames_data keys: {list(self.frames_data.keys())}")

    # ------------------ Frame Processing ---------------------
    def process_frame(self, frame_idx):
        """
        Perform data fusion + tracking and publish results for the specified frame_idx.
        """
        frame = self.frames_data.get(frame_idx, {})
        if not frame:
            self.get_logger().warn(f"No data found for frame {frame_idx}. Skipping processing.")
            return

        image = frame.get('image')
        point_cloud = frame.get('point_cloud')

        if image is None:
            self.get_logger().warn(f"No image data for frame {frame_idx}. Skipping processing.")
            return

        combined_trackers = []

        # Iterate over each category in cat_list
        for cat in self.cat_list:
            # Retrieve detections for the current category and frame_idx
            dets_2d_key = f'dets_2d_{cat.lower()}'
            dets_3d_key = f'dets_3d_{cat.lower()}'

            dets_2d = np.array(frame.get(dets_2d_key, []))
            dets_3d = np.array(frame.get(dets_3d_key, []))

            # Check if detections have the expected number of columns
            expected_dets_3d_cols = 14  # Adjust based on your detection data structure
            if dets_3d.size > 0 and dets_3d.shape[1] != expected_dets_3d_cols:
                self.get_logger().error(
                    f"3D detections for category '{cat}' in frame {frame_idx} have incorrect number of columns: "
                    f"{dets_3d.shape[1]} expected {expected_dets_3d_cols}"
                )
                continue  # Skip processing for this category

            # Category-level processing
            trackers = self.run_fusion_and_update(
                self.trackers[cat],
                dets_2d,
                dets_3d,
                cat,
                frame_idx
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

        self.get_logger().info(f"Combined trackers for frame {frame_idx}: {combined_trackers.shape[0]} objects total.")

        # Publish the raw 2D detections
        self.publish_raw_detections_image(frame_idx)

        # Optionally, publish other processed data here (e.g., annotated images)

    def run_fusion_and_update(self, tracker, dets_2d, dets_3d, category_name, frame_idx):
        """Fuse 2D & 3D, then update the specified tracker."""
        num_2d = dets_2d.shape[0]
        num_3d = dets_3d.shape[0]
        self.get_logger().info(
            f"Running fusion/update for '{category_name}' with {num_2d} 2D and {num_3d} 3D detections for frame {frame_idx}."
        )

        if num_2d == 0 and num_3d == 0:
            return np.empty((0, 10))

        # Basic parsing (adapt to your detection format!)
        dets_3d_camera = dets_3d[:, 7:14] if dets_3d.shape[1] >= 14 else np.empty((0, 7))
        ori_array = dets_3d[:, -1].reshape((-1, 1)) if dets_3d.shape[1] >= 14 else np.empty((0, 1))
        other_array = dets_3d[:, 1:7] if dets_3d.shape[1] >= 7 else np.empty((0, 6))
        additional_info = (
            np.concatenate((ori_array, other_array), axis=1)
            if (len(ori_array) > 0 and len(other_array) > 0)
            else np.empty((0, 7))
        )
        dets_3dto2d_image = dets_3d[:, 2:6] if dets_3d.shape[1] >= 6 else np.empty((0, 4))

        # 2D bounding boxes
        dets_2d_frame = dets_2d[:, :4] if dets_2d.shape[1] >= 4 else np.empty((0, 4))

        # Check if additional_info is not empty before proceeding
        if additional_info.size == 0 and num_3d > 0:
            self.get_logger().error(
                f"Additional info for 3D detections in category '{category_name}' for frame {frame_idx} is empty. "
                f"Skipping fusion for this category."
            )
            return np.empty((0, 10))

        # data_fusion
        dets_fusion, dets_only_3d, dets_only_2d = data_fusion(
            dets_3d_camera,
            dets_2d_frame,
            dets_3dto2d_image,
            additional_info
        )
        self.get_logger().info(
            f"Fusion results for {category_name} in frame {frame_idx}: "
            f"{len(dets_fusion['dets_3d_fusion'])} fused, "
            f"{len(dets_only_2d)} 2D-only, "
            f"{len(dets_only_3d['dets_3d_only'])} 3D-only."
        )

        # Convert 2D-only to top-left format
        dets_only_2d_tlwh = np.array([convert_x1y1x2y2_to_tlwh(i) for i in dets_only_2d])

        # Update the tracker
        trackers_output = tracker.update(
            dets_fusion,
            dets_only_2d_tlwh,
            dets_only_3d,
            self.cfg,
            frame_idx,
            0  # Assuming placeholder for another parameter
        )
        if trackers_output is not None and trackers_output.size > 0:
            self.get_logger().info(f"{category_name} tracker output shape: {trackers_output.shape}.")
        else:
            self.get_logger().info(f"{category_name} tracker output is empty.")

        return trackers_output

    def assign_unique_ids(self, category, ids):
        """
        Prefix IDs with category to ensure uniqueness across categories.
        Example:
            - Category 'Car' is assigned a prefix, e.g., 1000
            - Category 'Pedestrian' is assigned 2000
            - ID 1 for 'Car' becomes 1001
            - ID 1 for 'Pedestrian' becomes 2001
        """
        category_mapping = {cat: (idx + 1) * 1000 for idx, cat in enumerate(self.cat_list)}  # e.g., {'Car':1000, 'Pedestrian':2000}
        prefix = category_mapping.get(category, 0)  # Default prefix 0 if category not found

        unique_ids = prefix + ids
        return unique_ids

    # ------------------ Raw Detections Publishing ---------------------
    def publish_raw_detections_image(self, frame_idx):
        """Publish an image containing only the raw 2D detections (no tracking) for a specific frame."""
        frame = self.frames_data.get(frame_idx, {})
        image = frame.get('image')
        if image is None:
            self.get_logger().warn(f"No image data for frame {frame_idx}. Cannot publish raw detections image.")
            return

        img_raw = image.copy()

        # Iterate over each category in cat_list
        for cat in self.cat_list:
            detections = frame.get(f'dets_2d_{cat.lower()}', [])
            if not detections:
                continue
            color = (0, 255, 0) if cat == 'Car' else (255, 0, 0)  # Customize colors as needed
            label = "Car" if cat == 'Car' else "Pedestrian"

            for det in detections:
                try:
                    x1, y1, x2, y2 = map(int, det[:4])  # Assuming det = [x1, y1, x2, y2, confidence]
                    cv2.rectangle(img_raw, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        img_raw,
                        label,
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )
                except Exception as e:
                    self.get_logger().error(f"Error drawing detections for {cat}: {e}")

        # Convert and publish
        try:
            raw_msg = self.bridge.cv2_to_imgmsg(img_raw, encoding='bgr8')
            raw_msg.header.stamp = self.get_clock().now().to_msg()
            raw_msg.header.frame_id = "camera_link"
            self.raw_detections_image_pub.publish(raw_msg)
            self.get_logger().debug(f"Published raw detections image for frame {frame_idx}.")
        except Exception as e:
            self.get_logger().error(f"Failed to publish raw detections image for frame {frame_idx}: {e}")

    # ------------------ Main Processing ---------------------
    # Removed the original process_frame method since synchronization is now handled via frames_data

    # ------------------ 3D Projection Utility ---------------------
    # Retained existing methods for projecting and annotating 3D boxes
    # You may integrate similar publishing methods here if needed

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

