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
from visualization_msgs.msg import MarkerArray

# QoS imports for transient local
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

# Local/Project imports
from deepfusionmot.data_fusion import data_fusion
from deepfusionmot.DeepFusionMOT import DeepFusionMOT
from deepfusionmot.coordinate_transformation import convert_x1y1x2y2_to_tlwh
# We'll no longer directly call compute_box_3dto2d(...) with a file,
# but instead parse the calibration from the /camera/calibration topic.

class RealTimeFusionNode(Node):
    def __init__(self, cfg):
        super().__init__('real_time_fusion_node')
        self.cfg = cfg

        # Trackers for each category
        self.tracker_car = DeepFusionMOT(cfg, 'Car')
        self.tracker_ped = DeepFusionMOT(cfg, 'Pedestrian')

        self.bridge = CvBridge()

        # Buffers for latest detections & image
        self.latest_image = None
        self.latest_pc = None
        self.detections_2d_car = np.empty((0, 6))
        self.detections_2d_ped = np.empty((0, 6))
        self.detections_3d_car = np.empty((0, 15))
        self.detections_3d_ped = np.empty((0, 15))

        # Store calibration data from the /camera/calibration topic
        self.calib_data = None

        # --------------- Subscribers ---------------
        # 1) Camera image
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        # 2) LiDAR
        self.create_subscription(PointCloud2, '/lidar/points', self.lidar_callback, 10)

        # 3) 2D Detections (Car/Ped)
        self.create_subscription(Float32MultiArray, '/detection_2d/car', self.detection_2d_car_cb, 10)
        self.create_subscription(Float32MultiArray, '/detection_2d/pedestrian', self.detection_2d_ped_cb, 10)

        # 4) 3D Detections (Car/Ped)
        self.create_subscription(Float32MultiArray, '/detection_3d/car', self.detection_3d_car_cb, 10)
        self.create_subscription(Float32MultiArray, '/detection_3d/pedestrian', self.detection_3d_ped_cb, 10)

        # 5) Calibration data as a std_msgs/String
        #    We'll use transient_local in case it's published once and we join late
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        self.create_subscription(String, '/camera/calibration', self.calibration_cb, 10)

        # --------------- Publishers ---------------
        self.tracking_pub = self.create_publisher(MarkerArray, '/tracking_markers', 10)  # optional
        self.annotated_image_pub = self.create_publisher(Image, '/annotated_image', 10)
        self.raw_detections_image_pub = self.create_publisher(Image, '/raw_detections_image', 10)

        self.get_logger().info("RealTimeFusionNode initialized. Waiting for detections...")

    # ------------------ Calibration Callback ---------------------
    def calibration_cb(self, msg: String):
        """
        This callback is triggered once the /camera/calibration topic publishes a string.
        We'll parse that string to extract the relevant calibration items (e.g. P2, R_rect, etc.)
        and store them in self.calib_data as NumPy arrays.
        """
        self.get_logger().info("Calibration callback triggered!")
        calib_str = msg.data
        self.calib_data = self.parse_calib_string(calib_str)
        self.get_logger().info("Received calibration from /camera/calibration (std_msgs/String).")

        # For debugging, you can print or log self.calib_data
        # self.get_logger().info(f"Parsed calibration: {self.calib_data}")

    def parse_calib_string(self, calib_str):
        """
        Parse the raw calibration string into a dictionary of arrays.
        For example, lines might look like:
          P0: 7.215377e+02 0.0 6.095593e+02 0.0 ...
          P1: ...
          P2: ...
          R_rect: ...
          Tr_velo_cam: ...
        We'll store them in a dict e.g. {'P0': np.array(...), 'R_rect': np.array(...)} etc.

        NOTE: Adjust this logic based on your actual format.
        """
        lines = calib_str.strip().split('\n')
        calib_dict = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # e.g. "P0: 7.215377000000e+02 0.000000000000e+00 ..."
            # split by ":", then parse numbers
            if ':' not in line:
                # skip lines that don't have a key:value format
                continue

            key, vals = line.split(':', 1)
            key = key.strip()  # e.g. "P0"
            vals = vals.strip()  # e.g. "7.215377000000e+02 0.000000..."

            # now parse the numbers
            num_strs = vals.split()
            floats = [float(x) for x in num_strs]

            # We'll store them as an np.array. 
            # You can reshape them if you know the dimension (e.g. 3x4 for P0).
            # For instance, if it's "P2" we might do (3,4).
            if key.startswith('P'):
                # For "P0", "P1", "P2", "P3" => we assume 3x4
                # Only do it if we indeed have 12 floats
                if len(floats) == 12:
                    matrix = np.array(floats).reshape(3, 4)
                    calib_dict[key] = matrix
                else:
                    # fallback to 1D array
                    calib_dict[key] = np.array(floats)
            elif key.startswith('R_rect'):
                # might be a 3x3
                if len(floats) == 9:
                    matrix = np.array(floats).reshape(3, 3)
                    calib_dict[key] = matrix
                else:
                    # fallback
                    calib_dict[key] = np.array(floats)
            elif key.startswith('Tr_'):
                # e.g. "Tr_velo_cam"
                # Sometimes it's 3x4 or 4x4. You need to know.
                # Suppose it's 3x4 (like KITTI). Then do:
                if len(floats) == 12:
                    matrix = np.array(floats).reshape(3, 4)
                    calib_dict[key] = matrix
                else:
                    calib_dict[key] = np.array(floats)
            else:
                # fallback for unknown keys
                calib_dict[key] = np.array(floats)

        return calib_dict

    # ------------------ Image & LiDAR Callbacks ---------------------
    def image_callback(self, msg: Image):
        """Triggered on new camera image."""
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.get_logger().info(f"Received an image: shape={self.latest_image.shape}")

        # 1) Publish the raw 2D detections
        self.publish_raw_detections_image()
        # 2) Then run the fusion + tracking + 3D-annotated image
        self.process_frame()

    def lidar_callback(self, msg: PointCloud2):
        """Triggered on new LiDAR point cloud."""
        self.latest_pc = msg
        self.get_logger().info("Received a new point cloud.")

    # ------------------ Detection Callbacks ---------------------
    def detection_2d_car_cb(self, msg: Float32MultiArray):
        arr = np.array(msg.data).reshape(-1, 6)
        self.detections_2d_car = arr
        self.get_logger().info(f"Received {arr.shape[0]} 2D Car detections.")

    def detection_2d_ped_cb(self, msg: Float32MultiArray):
        arr = np.array(msg.data).reshape(-1, 6)
        self.detections_2d_ped = arr
        self.get_logger().info(f"Received {arr.shape[0]} 2D Pedestrian detections.")

    def detection_3d_car_cb(self, msg: Float32MultiArray):
        arr = np.array(msg.data).reshape(-1, 15)
        self.detections_3d_car = arr
        self.get_logger().info(f"Received {arr.shape[0]} 3D Car detections.")

    def detection_3d_ped_cb(self, msg: Float32MultiArray):
        arr = np.array(msg.data).reshape(-1, 15)
        self.detections_3d_ped = arr
        self.get_logger().info(f"Received {arr.shape[0]} 3D Pedestrian detections.")

    # ------------------ Main Processing ---------------------
    def process_frame(self):
        """Perform data fusion + tracking and publish results."""
        if self.latest_image is None:
            self.get_logger().info("No image yet, skipping frame processing.")
            return

        # If we haven't gotten calibration yet, skip 3D stuff
        if self.calib_data is None:
            self.get_logger().warn("No camera calibration yet. Skipping 3D projection.")
            return

        # Category-level processing (Car + Ped)
        trackers_car = self.run_fusion_and_update(
            self.tracker_car,
            self.detections_2d_car,
            self.detections_3d_car,
            'Car'
        )
        trackers_ped = self.run_fusion_and_update(
            self.tracker_ped,
            self.detections_2d_ped,
            self.detections_3d_ped,
            'Pedestrian'
        )

        # Combine results
        if (trackers_car is not None and trackers_car.size > 0) and \
           (trackers_ped is not None and trackers_ped.size > 0):
            combined_trackers = np.vstack([trackers_car, trackers_ped])
        elif (trackers_car is not None and trackers_car.size > 0):
            combined_trackers = trackers_car
        elif (trackers_ped is not None and trackers_ped.size > 0):
            combined_trackers = trackers_ped
        else:
            combined_trackers = np.empty((0, 10))  # shape depends on your tracking data

        self.get_logger().info(f"Combined trackers: {combined_trackers.shape[0]} objects total.")

        # 1) If you still want 3D bounding boxes in RViz, do it:
        marker_array = self.create_marker_array(combined_trackers)
        self.tracking_pub.publish(marker_array)
        self.get_logger().info(f"Published {len(marker_array.markers)} markers for visualization in RViz.")

        # 2) Publish the 3D-projected image with bounding boxes & track IDs
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
        dets_3d_fusion, dets_3d_only, dets_2d_only = data_fusion(
            dets_3d_camera,
            dets_2d_frame,
            dets_3dto2d_image,
            additional_info
        )
        self.get_logger().info(
            f"Fusion results for {category_name}: "
            f"{len(dets_3d_fusion['dets_3d_fusion'])} fused, "
            f"{len(dets_2d_only)} 2D-only, "
            f"{len(dets_3d_only['dets_3d_only'])} 3D-only."
        )

        # Convert 2D-only to top-left format
        dets_2d_only_tlwh = np.array([convert_x1y1x2y2_to_tlwh(i) for i in dets_2d_only])

        # Update the tracker
        trackers_output = tracker.update(dets_3d_fusion, dets_2d_only_tlwh, dets_3d_only, self.cfg, 0, 0)
        if trackers_output is not None and trackers_output.size > 0:
            self.get_logger().info(f"{category_name} tracker output shape: {trackers_output.shape}.")
        else:
            self.get_logger().info(f"{category_name} tracker output is empty.")

        return trackers_output

    # ------------------ Image Publishing ---------------------
    def publish_raw_detections_image(self):
        """Publish an image containing only the raw 2D detections (no tracking)."""
        if self.latest_image is None:
            return

        img_raw = self.latest_image.copy()

        # Draw raw 2D Car (green)
        for det in self.detections_2d_car:
            x1, y1, x2, y2 = map(int, det[1:5])
            cv2.rectangle(img_raw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_raw, "Car", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw raw 2D Pedestrian (blue)
        for det in self.detections_2d_ped:
            x1, y1, x2, y2 = map(int, det[1:5])
            cv2.rectangle(img_raw, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_raw, "Pedestrian", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Convert and publish
        raw_msg = self.bridge.cv2_to_imgmsg(img_raw, encoding='bgr8')
        raw_msg.header.stamp = self.get_clock().now().to_msg()
        raw_msg.header.frame_id = "camera_link"
        self.raw_detections_image_pub.publish(raw_msg)

    def publish_3d_annotated_image(self, trackers):
        """
        1) For each tracker row => parse [ID, h, w, l, x, y, z, theta, orientation, ...].
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
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        annotated_msg.header.stamp = self.get_clock().now().to_msg()
        annotated_msg.header.frame_id = "camera_link"
        self.annotated_image_pub.publish(annotated_msg)

    # ------------------ RViz Markers (optional) ---------------------
    def create_marker_array(self, trackers):
        from visualization_msgs.msg import Marker, MarkerArray
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for row in trackers:
            track_id = int(row[0])
            h, w, l = row[1], row[2], row[3]
            x, y, z = row[4], row[5], row[6]
            yaw = row[7]

            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = "velodyne"  # or "camera_link"
            marker.ns = "tracked_objects"
            marker.id = track_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z + (h / 2.0)

            from geometry_msgs.msg import Quaternion
            half_yaw = yaw * 0.5
            marker.pose.orientation.z = math.sin(half_yaw)
            marker.pose.orientation.w = math.cos(half_yaw)

            marker.scale.x = l
            marker.scale.y = w
            marker.scale.z = h

            color = self.compute_color_for_id(track_id)
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 0.6

            marker_array.markers.append(marker)

            # Optional: add TEXT marker
            text_marker = Marker()
            text_marker.header.stamp = now
            text_marker.header.frame_id = marker.header.frame_id
            text_marker.ns = "tracked_id"
            text_marker.id = track_id + 100000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = x
            text_marker.pose.position.y = y
            text_marker.pose.position.z = z + h + 0.3
            text_marker.scale.z = 0.8
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = f"ID: {track_id}"
            text_marker.pose.orientation = marker.pose.orientation
            marker_array.markers.append(text_marker)

        return marker_array

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
        We'll look for 'P2' or similar in self.calib_data
        (or if you prefer 'P0' as your default).
        Then build corners, rotate, translate, project.
        """
        if 'P2' not in self.calib_data:
            self.get_logger().warn("No 'P2' matrix in calib_data. Check your parsed keys. Skipping.")
            return None

        P = self.calib_data['P2']  # shape (3,4) from parse_calib_string

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

        # normalize
        corners_2D[:, 0] /= corners_2D[:, 2]
        corners_2D[:, 1] /= corners_2D[:, 2]

        # any corner behind camera?
        if np.any(corners_2D[:, 2] < 0.1):
            return None

        return corners_2D[:, :2]

    # ------------------ Misc Helpers ---------------------
    def compute_color_for_id(self, idx):
        palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
        color = [int((p * (idx**2 - idx + 1)) % 255) for p in palette]
        return tuple(color)

def main(args=None):
    # Example config path - adjust as needed
    config_file = '/home/prabuddhi/ros2_ws1/src/deepfusionmot/config/kitti_real_time.yaml'
    rclpy.init(args=args)

    from deepfusionmot.config import Config
    cfg, _ = Config(config_file)

    node = RealTimeFusionNode(cfg)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

