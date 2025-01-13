#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node

import numpy as np
import cv2
from cv_bridge import CvBridge

# ROS 2 message imports
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

# Local/Project imports
from deepfusionmot.data_fusion import data_fusion
from deepfusionmot.DeepFusionMOT import DeepFusionMOT
from deepfusionmot.coordinate_transformation import convert_x1y1x2y2_to_tlwh

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

        # Subscribers
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(PointCloud2, '/lidar/points', self.lidar_callback, 10)
        self.create_subscription(Float32MultiArray, '/detection_2d/car', self.detection_2d_car_cb, 10)
        self.create_subscription(Float32MultiArray, '/detection_2d/pedestrian', self.detection_2d_ped_cb, 10)
        self.create_subscription(Float32MultiArray, '/detection_3d/car', self.detection_3d_car_cb, 10)
        self.create_subscription(Float32MultiArray, '/detection_3d/pedestrian', self.detection_3d_ped_cb, 10)

        # Publishers
        self.tracking_pub = self.create_publisher(MarkerArray, '/tracking_markers', 10)
        # Annotated image with tracking results
        self.annotated_image_pub = self.create_publisher(Image, '/annotated_image', 10)
        # NEW: raw detections only
        self.raw_detections_image_pub = self.create_publisher(Image, '/raw_detections_image', 10)

        self.get_logger().info("RealTimeFusionNode initialized. Waiting for detections...")

    def image_callback(self, msg):
        """Triggered on new camera image."""
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.get_logger().info(f"Received an image: shape={self.latest_image.shape}")

        # If you'd like to publish the raw detections image right away:
        self.publish_raw_detections_image()
        # Then do the rest (fusion + tracking + annotated image)
        self.process_frame()

    def lidar_callback(self, msg):
        """Triggered on new LiDAR point cloud."""
        self.latest_pc = msg
        self.get_logger().info("Received a new point cloud.")

    def detection_2d_car_cb(self, msg):
        arr = np.array(msg.data).reshape(-1, 6)
        self.detections_2d_car = arr
        self.get_logger().info(f"Received {arr.shape[0]} 2D Car detections.")

    def detection_2d_ped_cb(self, msg):
        arr = np.array(msg.data).reshape(-1, 6)
        self.detections_2d_ped = arr
        self.get_logger().info(f"Received {arr.shape[0]} 2D Pedestrian detections.")

    def detection_3d_car_cb(self, msg):
        arr = np.array(msg.data).reshape(-1, 15)
        self.detections_3d_car = arr
        self.get_logger().info(f"Received {arr.shape[0]} 3D Car detections.")

    def detection_3d_ped_cb(self, msg):
        arr = np.array(msg.data).reshape(-1, 15)
        self.detections_3d_ped = arr
        self.get_logger().info(f"Received {arr.shape[0]} 3D Pedestrian detections.")

    def process_frame(self):
        """Perform data fusion + tracking and publish results."""
        if self.latest_image is None:
            self.get_logger().info("No image yet, skipping frame processing.")
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
            combined_trackers = np.empty((0, 10))  # Adjust shape if needed

        self.get_logger().info(f"Combined trackers: {combined_trackers.shape[0]} objects total.")

        # 1) Publish 3D bounding boxes to RViz
        marker_array = self.create_marker_array(combined_trackers)
        self.tracking_pub.publish(marker_array)
        self.get_logger().info(f"Published {len(marker_array.markers)} markers for visualization.")

        # 2) Publish a 2D-annotated image with bounding boxes + track ID, if you store them
        self.publish_annotated_image()

    def run_fusion_and_update(self, tracker, dets_2d, dets_3d, category_name):
        """Fuse 2D & 3D, then update the specified tracker."""
        num_2d = dets_2d.shape[0]
        num_3d = dets_3d.shape[0]
        self.get_logger().info(f"Running fusion/update for '{category_name}' with {num_2d} 2D and {num_3d} 3D detections.")

        if num_2d == 0 and num_3d == 0:
            return np.empty((0, 10))

        # Parse relevant columns for 3D
        if dets_3d.shape[1] >= 14:
            dets_3d_camera = dets_3d[:, 7:14]
        else:
            dets_3d_camera = np.empty((0,7))

        if dets_3d.shape[1] >= 15:
            ori_array = dets_3d[:, -1].reshape((-1,1))
        else:
            ori_array = np.empty((0,1))

        if dets_3d.shape[1] >= 7:
            other_array = dets_3d[:, 1:7]
        else:
            other_array = np.empty((0,6))

        if len(ori_array) > 0 and len(other_array) > 0:
            additional_info = np.concatenate((ori_array, other_array), axis=1)
        else:
            additional_info = np.empty((0,7))

        if dets_3d.shape[1] >= 6:
            dets_3dto2d_image = dets_3d[:, 2:6]
        else:
            dets_3dto2d_image = np.empty((0,4))

        # 2D bounding boxes
        if dets_2d.shape[1] >= 5:
            dets_2d_frame = dets_2d[:, 1:5]
        else:
            dets_2d_frame = np.empty((0,4))

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

    def publish_raw_detections_image(self):
        """Publish an image containing only the raw 2D detections, no tracking info."""
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

    def publish_annotated_image(self):
        """Draw bounding boxes from the final trackers (or raw) with ID, etc. 
           Currently draws the raw 2D boxes, but you can adapt to use track IDs.
        """
        if self.latest_image is None:
            return
        annotated = self.latest_image.copy()

        # Example: same as original, drawing raw detection boxes
        for det in self.detections_2d_car:
            x1, y1, x2, y2 = map(int, det[1:5])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, "Car", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for det in self.detections_2d_ped:
            x1, y1, x2, y2 = map(int, det[1:5])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated, "Pedestrian", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Possibly also overlay track ID bounding boxes if your final tracker 
        # includes 2D bounding coords in columns [8..11].
        # e.g.:
        # for row in combined_trackers:
        #    track_id = int(row[0])
        #    x1,y1,x2,y2 = map(int, row[8:12])
        #    # draw with a different color + label "ID: {track_id}"

        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        annotated_msg.header.stamp = self.get_clock().now().to_msg()
        annotated_msg.header.frame_id = "camera_link"
        self.annotated_image_pub.publish(annotated_msg)

    def create_marker_array(self, trackers):
        """Same as before: produce 3D bounding boxes (CUBE) with ID text in RViz."""
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for row in trackers:
            track_id = int(row[0])
            h, w, l = row[1], row[2], row[3]
            x, y, z = row[4], row[5], row[6]
            yaw = row[7]

            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = "velodyne"
            marker.ns = "tracked_objects"
            marker.id = track_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()

            # Position
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z + (h / 2.0)

            # Orientation about Z axis (yaw)
            import math
            from geometry_msgs.msg import Quaternion
            q_yaw = Quaternion()
            half_yaw = yaw * 0.5
            q_yaw.z = math.sin(half_yaw)
            q_yaw.w = math.cos(half_yaw)
            marker.pose.orientation = q_yaw

            # Scale
            marker.scale.x = l
            marker.scale.y = w
            marker.scale.z = h

            color = self.compute_color_for_id(track_id)
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 0.6
            marker_array.markers.append(marker)

            # Text marker
            text_marker = Marker()
            text_marker.header.stamp = now
            text_marker.header.frame_id = marker.header.frame_id
            text_marker.ns = "tracked_id"
            text_marker.id = track_id + 100000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.lifetime = marker.lifetime
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

    def compute_color_for_id(self, idx):
        palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
        color = [int((p * (idx**2 - idx + 1)) % 255) for p in palette]
        return tuple(color)

def main(args=None):
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

