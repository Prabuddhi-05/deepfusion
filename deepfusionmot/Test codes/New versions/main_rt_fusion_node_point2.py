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
from std_msgs.msg import String, ColorRGBA
from my_msgs.msg import Float32MultiArrayStamped  # aggregated bounding boxes

# For exact-time matching
from message_filters import Subscriber, TimeSynchronizer

# For Markers in RViz
from visualization_msgs.msg import Marker, MarkerArray

# Coordinate transformations (must define in coordinate_transformation.py)
from deepfusionmot.coordinate_transformation import (
    convert_3dbox_to_8corner,
    compute_box_3dto2d,
    transform_cam_to_velo
)


# Fusion + tracking
from deepfusionmot.data_fusion import data_fusion
from deepfusionmot.DeepFusionMOT import DeepFusionMOT
from deepfusionmot.config import Config


def compute_color_for_id(track_id):
    """
    Assign a color based on track_id (deterministic).
    """
    palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
    color = [int((p * (track_id**2 - track_id + 1)) % 255) for p in palette]
    return tuple(color)


class RealTimeFusionNode(Node):
    def __init__(self, cfg):
        super().__init__('real_time_fusion_node')
        self.cfg = cfg
        self.cat_list = self.cfg.get('cat_list', ['Car', 'Pedestrian'])
        self.get_logger().info(f"Configured categories: {self.cat_list}")

        self.bridge = CvBridge()
        # Create one tracker per category
        self.trackers = {}
        for cat in self.cat_list:
            self.trackers[cat] = DeepFusionMOT(cfg, cat)

        # Subscribers for image, pointcloud, detection topics
        self.img_sub = Subscriber(self, Image, '/camera/image_raw')
        self.pc_sub  = Subscriber(self, PointCloud2, '/lidar/points')
        self.det2_car_sub = Subscriber(self, Float32MultiArrayStamped, '/detection_2d/car')
        self.det3_car_sub = Subscriber(self, Float32MultiArrayStamped, '/detection_3d/car')
        self.det2_ped_sub = Subscriber(self, Float32MultiArrayStamped, '/detection_2d/pedestrian')
        self.det3_ped_sub = Subscriber(self, Float32MultiArrayStamped, '/detection_3d/pedestrian')

        # Use TimeSynchronizer to ensure all 6 messages have the same timestamp
        self.ts = TimeSynchronizer([
            self.img_sub,
            self.pc_sub,
            self.det2_car_sub,
            self.det3_car_sub,
            self.det2_ped_sub,
            self.det3_ped_sub
        ], queue_size=10)
        self.ts.registerCallback(self.sync_cb)

        # Calibration subscriber (string messages with P2, R_rect, Tr_velo_cam lines)
        self.calib_str = None
        self.calib_sub = self.create_subscription(
            String,
            '/camera/calibration',
            self.calib_callback,
            10
        )

        # Publishers: raw detection overlay, annotated image, 3D markers
        self.raw_detections_image_pub = self.create_publisher(Image, '/raw_detections_image', 10)
        self.annotated_image_pub      = self.create_publisher(Image, '/annotated_image', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/boxes_3d', 10)

        self.get_logger().info("Node ready: TimeSync + calibration + 3D markers.")

    def calib_callback(self, msg: String):
        """
        Called whenever a new /camera/calibration message arrives.
        We store the entire multi-line calibration text in memory.
        """
        self.calib_str = msg.data
        self.get_logger().info(f"Got calibration data, length={len(self.calib_str)}")

    def sync_cb(self,
                img_msg, pc_msg,
                det2_car_msg, det3_car_msg,
                det2_ped_msg, det3_ped_msg):
        """
        Called whenever all 6 topics have the same (sec, nsec) time (via TimeSynchronizer).
        """
        stamp_sec  = img_msg.header.stamp.sec
        stamp_nsec = img_msg.header.stamp.nanosec
        self.get_logger().info(f"TimeSync => stamp=({stamp_sec},{stamp_nsec})")

        # Convert the ROS Image into a CV image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed image conversion: {e}")
            return

        # Parse 2D/3D detections for Car/Pedestrian
        arr_2d_car = self.parse_2d_detections(det2_car_msg, 'Car')
        arr_3d_car = self.parse_3d_detections(det3_car_msg, 'Car')
        arr_2d_ped = self.parse_2d_detections(det2_ped_msg, 'Pedestrian')
        arr_3d_ped = self.parse_3d_detections(det3_ped_msg, 'Pedestrian')

        # Build a dictionary for fusion/tracking
        detection_dict = {
            'Car': {
                '2d': arr_2d_car,
                '3d': arr_3d_car
            },
            'Pedestrian': {
                '2d': arr_2d_ped,
                '3d': arr_3d_ped
            }
        }

        # 1) Publish 2D bounding-box overlay
        self.publish_raw_detections_image(cv_img, detection_dict)

        # 2) Fusion + Tracking
        cat_trackers_list = self.fuse_and_track(cv_img, detection_dict)
        if cat_trackers_list:
            combined_3d_trackers = np.vstack(cat_trackers_list)
        else:
            combined_3d_trackers = np.empty((0,9))

        # 3) 2D annotated image with 3D boxes drawn in the camera plane
        self.publish_3d_annotated_image(cv_img, combined_3d_trackers)

        # 4) 3D markers in the "velodyne" frame
        self.publish_3d_markers(combined_3d_trackers, stamp_sec, stamp_nsec)

    def parse_2d_detections(self, msg, cat='Car'):
        if len(msg.data) > 0:
            arr = np.array(msg.data, dtype=np.float32).reshape(-1,6)
            self.get_logger().info(f"[2D {cat}] shape={arr.shape}")
        else:
            arr = np.empty((0,6))
            self.get_logger().info(f"[2D {cat}] => empty")
        return arr

    def parse_3d_detections(self, msg, cat='Car'):
        if len(msg.data) > 0:
            arr = np.array(msg.data, dtype=np.float32).reshape(-1,15)
            self.get_logger().info(f"[3D {cat}] shape={arr.shape}")
        else:
            arr = np.empty((0,15))
            self.get_logger().info(f"[3D {cat}] => empty")
        return arr

    def publish_raw_detections_image(self, cv_img, detection_dict):
        """
        Draw 2D bounding boxes for Car/Pedestrian onto 'cv_img' and publish to /raw_detections_image.
        """
        if cv_img is None:
            return
        img_draw = cv_img.copy()

        # Example: Car => green
        if 'Car' in self.cat_list:
            arr_2d = detection_dict['Car']['2d']
            color  = (0,255,0)
            for det in arr_2d:
                # Format: [frame_idx, x1, y1, x2, y2, score]
                x1, y1, x2, y2 = map(int, det[1:5])
                cv2.rectangle(img_draw, (x1,y1), (x2,y2), color, 2)
                cv2.putText(img_draw, "Car", (x1, max(0,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Pedestrian => blue
        if 'Pedestrian' in self.cat_list:
            arr_2d = detection_dict['Pedestrian']['2d']
            color  = (255,0,0)
            for det in arr_2d:
                x1, y1, x2, y2 = map(int, det[1:5])
                cv2.rectangle(img_draw, (x1,y1), (x2,y2), color, 2)
                cv2.putText(img_draw, "Ped", (x1, max(0,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Publish
        try:
            msg_out = self.bridge.cv2_to_imgmsg(img_draw, encoding='bgr8')
            msg_out.header.stamp = self.get_clock().now().to_msg()
            msg_out.header.frame_id = "camera_link"
            self.raw_detections_image_pub.publish(msg_out)
        except Exception as e:
            self.get_logger().error(f"Failed to publish raw detection image: {e}")

    def fuse_and_track(self, cv_img, detection_dict):
        """
        Example placeholder method for data_fusion + tracker.update(...).
        Returns a list of Nx9 arrays, one per category.
        """
        cat_trackers_list = []
        # For each category in self.cat_list:
        #   1) extract 2D, 3D arrays
        #   2) call data_fusion(...)
        #   3) run self.trackers[cat].update(...)
        #   4) ensure output is shape (*, 9)
        #   5) cat_trackers_list.append(...)
        return cat_trackers_list

    def publish_3d_annotated_image(self, cv_img, trackers):
        """
        Draw 3D bounding boxes in the 2D camera image plane (for debugging).
        You presumably have some show_image_with_boxes_3d(...) function in your code.
        """
        if cv_img is None:
            return

        self.get_logger().info(f"publish_3d_annotated_image => trackers shape={trackers.shape}")
        if self.calib_str is None:
            self.get_logger().warn("No calibration data => skipping 3D image overlay.")
            return

        # For each row in trackers, draw the 3D box into cv_img
        # ...

    def publish_3d_markers(self, trackers, stamp_sec, stamp_nsec):
        """
        Publishes MarkerArray for the 3D bounding boxes in the 'velodyne' frame.
        We'll:
         1) compute corners in camera coords
         2) transform corners -> velodyne
         3) build line segments
        """
        from geometry_msgs.msg import Point
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        # If no calibration => can't transform
        if self.calib_str is None:
            self.get_logger().warn("No calibration => can't transform boxes to velodyne. Skipping.")
            return

        marker_id = 0
        for row in trackers:
            if row.size < 9:
                continue

            track_id = int(row[0])
            # [h, w, l, x_cam, y_cam, z_cam, yaw] => row[1:8]
            h, w, l, cx, cy, cz, yaw = row[1:8]

            # Step 1) corners in camera coords
            corners_cam  = convert_3dbox_to_8corner([h,w,l, x_cam, y_cam, z_cam, yaw])  # shape (8,3)

            # Step 2) transform corners to velodyne coords
            corners_velo = transform_cam_to_velo(corners_cam, self.calib_str)
           

            # Step 3) Build line segments for a wireframe box
            edges = [
                (0,1),(1,2),(2,3),(3,0),
                (4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7)
            ]

            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = "velodyne"
            marker.ns = "3d_bboxes"
            marker.id = marker_id
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            # Keep markers forever
            marker.lifetime = rclpy.duration.Duration(seconds=0,nanoseconds=0).to_msg()

            marker.scale.x = 0.06  # line thickness
            color_bgr = compute_color_for_id(track_id)
            marker.color = ColorRGBA(r=float(color_bgr[2])/255.0,
                                     g=float(color_bgr[1])/255.0,
                                     b=float(color_bgr[0])/255.0,
                                     a=0.9)

            for (i0, i1) in edges:
                p0 = corners_velo[i0]
                p1 = corners_velo[i1]
                marker.points.append(Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])))
                marker.points.append(Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])))

            marker_array.markers.append(marker)
            marker_id += 1

        self.marker_pub.publish(marker_array)


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

