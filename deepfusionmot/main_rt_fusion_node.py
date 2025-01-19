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
from std_msgs.msg import String, ColorRGBA # ColorRGBA - for markers
from my_msgs.msg import Float32MultiArrayStamped # Customized message type

# For exact-time matching
from message_filters import Subscriber, TimeSynchronizer

# Markers for RViz
from visualization_msgs.msg import Marker, MarkerArray

# DeepFusionMOT imports
from deepfusionmot.data_fusion import data_fusion
from deepfusionmot.DeepFusionMOT import DeepFusionMOT
from deepfusionmot.coordinate_transformation import (
    compute_box_3dto2d,  # Projects 3D bounding boxes onto the 2D image plane
    convert_x1y1x2y2_to_tlwh, 
    roty # Computes 3D rotation matrices  
)
from deepfusionmot.config import Config  # Configuration loader

def compute_color_for_id(track_id):
    """
    Assigns a color based on track_id - to draw bounding boxes or markers in RViz
    """
    palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
    color = [int((p * (track_id**2 - track_id + 1)) % 255) for p in palette]
    return tuple(color)

def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=2):
    """
    Draws a 3D bounding box in 2D image space.
    qs: (8,2) array of vertices for the 8 3D box corners.
    """
    if qs is not None and qs.shape[0] == 8:
        qs = qs.astype(np.int32)
        for k in range(4):
            i, j = k, (k + 1) % 4
            image = cv2.line(image, (qs[i, 0], qs[i, 1]),
                             (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k + 4, ((k + 1) % 4) + 4
            image = cv2.line(image, (qs[i, 0], qs[i, 1]),
                             (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k, k + 4
            image = cv2.line(image, (qs[i, 0], qs[i, 1]),
                             (qs[j, 0], qs[j, 1]), color, thickness)
    return image

def show_image_with_boxes_3d(img, bbox3d_tmp, calib_str,
                             track_id=0, color=(255,255,255),
                             label_str="", line_thickness=2):
    """
    Projects a single 3D bounding box [h,w,l,x,y,z,theta] into the image
    by parsing the calibration from a string
    """
    corners_2d = compute_box_3dto2d(bbox3d_tmp, calib_str, from_string=True)
    if corners_2d is None:
        print(f"[WARN] track_id={track_id} => Invalid bounding box. No corners in image.")
        return img

    # Draw the 3D box in 2D
    img = draw_projected_box3d(img, corners_2d, color=color, thickness=line_thickness)

    # Label near corner 4
    c1 = (int(corners_2d[4, 0]), int(corners_2d[4, 1]))
    tf = max(line_thickness - 1, 1)
    t_size = cv2.getTextSize(str(label_str), 0, fontScale=line_thickness/3, thickness=tf)[0]
    c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  
    cv2.putText(img, str(label_str), (c1[0], c1[1] - 2), 0,
                line_thickness/3, (225,255,255), thickness=tf, lineType=cv2.LINE_AA)

    return img # Annotated image

def compute_3d_corners_xyz(bbox3d):
    """
    Converts 3D box [h,w,l,x,y,z,theta] into an array of shape (8,3)
    containing the 8 corners in 3D (in the same coordinate frame as x,y,z)
    """
    # Similar to 'convert_3dbox_to_8corner' logic
    h, w, l, x, y, z, yaw = bbox3d
    R = roty(yaw)

    # Defines corners in local coordinate frame
    x_corners = [l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [0,    0,    0,    0,   -h,   -h,   -h,   -h]
    z_corners = [w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]

    # Rotates and translates
    corners = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))  # (3,8)
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z

    corners_3d = corners.T  # shape (8,3)
    return corners_3d

class RealTimeFusionNode(Node): # ROS2 Node
    """
    A node that uses a single TimeSynchronizer for:
      - /camera/image_raw
      - /lidar/points
      - /detection_2d/car
      - /detection_3d/car
      - /detection_2d/pedestrian
      - /detection_3d/pedestrian
    plus a calibration subscriber
    """

    def __init__(self, cfg):
        super().__init__('real_time_fusion_node')
        self.cfg = cfg # Store configuration passed to the node
        self.cat_list = self.cfg.get('cat_list', ['Car', 'Pedestrian']) # Read and store categories from config
        self.get_logger().info(f"Configured categories: {self.cat_list}")

        self.bridge = CvBridge()

        # Create a tracker for each category
        self.trackers = {}
        for cat in self.cat_list:
            self.trackers[cat] = DeepFusionMOT(cfg, cat)

        # Subscribers for image, pointcloud, detection topics
        self.img_sub  = Subscriber(self, Image, '/camera/image_raw')
        self.pc_sub   = Subscriber(self, PointCloud2, '/lidar/points')
        self.det2_car_sub = Subscriber(self, Float32MultiArrayStamped, '/detection_2d/car')
        self.det3_car_sub = Subscriber(self, Float32MultiArrayStamped, '/detection_3d/car')
        self.det2_ped_sub = Subscriber(self, Float32MultiArrayStamped, '/detection_2d/pedestrian')
        self.det3_ped_sub = Subscriber(self, Float32MultiArrayStamped, '/detection_3d/pedestrian')

        # TimeSynchronizer 
        self.ts = TimeSynchronizer([   # Synchronizes messages from the ROS2 topics
            self.img_sub,
            self.pc_sub,
            self.det2_car_sub,
            self.det3_car_sub,
            self.det2_ped_sub,
            self.det3_ped_sub
        ], queue_size=10)
        self.ts.registerCallback(self.sync_cb) # Calls whenever a synchronized set of messages is available

        # Calibration subscriber
        self.calib_str = None
        self.calib_sub = self.create_subscription(
            String,
            '/camera/calibration',
            self.calib_callback,
            10
        )

        # Publishers for debug images
        self.raw_detections_image_pub = self.create_publisher(Image, '/raw_detections_image', 10)
        self.annotated_image_pub      = self.create_publisher(Image, '/annotated_image', 10)

        # Publisher for 3D bounding box markers in RViz
        self.marker_pub = self.create_publisher(MarkerArray, '/boxes_3d', 10)

        self.get_logger().info("RealTimeFusionNode: 6-topic TimeSync")

    def calib_callback(self, msg: String):
        """
        Stores the calibration text.
        """
        self.calib_str = msg.data
        self.get_logger().info(f"Received calibration data")
        #self.get_logger().info(f"Received calibration data (len={len(msg.data)} chars)")

    def sync_cb(self,
                img_msg, pc_msg,
                det2_car_msg, det3_car_msg,
                det2_ped_msg, det3_ped_msg):
        """
        Called whenever all 6 topics have a matching timestamp.
        """
        stamp_sec  = img_msg.header.stamp.sec 
        stamp_nsec = img_msg.header.stamp.nanosec
        self.get_logger().info(f"TimeSync => stamp=({stamp_sec},{stamp_nsec})") # Logs timestamp of synchronized messages

        # Converts image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Extracts detection data
        arr_2d_car = self.parse_2d_detections(det2_car_msg, 'Car')
        arr_3d_car = self.parse_3d_detections(det3_car_msg, 'Car')
        arr_2d_ped = self.parse_2d_detections(det2_ped_msg, 'Pedestrian')
        arr_3d_ped = self.parse_3d_detections(det3_ped_msg, 'Pedestrian')

        # Creates a dictionary organizing detections by category
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

        # Publishes 2D bounding boxes
        self.publish_raw_detections_image(cv_img, detection_dict)

        # Fusion + Tracking 
        cat_trackers_list = self.fuse_and_track(cv_img, detection_dict)

        # Flattens results for visualization
        if cat_trackers_list:
            combined_3d_trackers = np.vstack(cat_trackers_list)
        else:
            combined_3d_trackers = np.empty((0,9))

        # Publishes annotated image with 3D bounding boxes
        self.publish_3d_annotated_image(cv_img, combined_3d_trackers)

        # Publishes the 3D bounding boxes as markers in RViz
        self.publish_3d_markers(combined_3d_trackers, stamp_sec, stamp_nsec)

    # Extracts 2D detection data
    def parse_2d_detections(self, msg: Float32MultiArrayStamped, cat='Car'):
        if len(msg.data) > 0:
            arr = np.array(msg.data, dtype=np.float32).reshape(-1, 6)
            self.get_logger().info(f"[2D {cat}] stamp=({msg.header.stamp.sec},{msg.header.stamp.nanosec}), shape={arr.shape}")
        else:
            arr = np.empty((0,6))
            self.get_logger().info(f"[2D {cat}] (empty) => stamp=({msg.header.stamp.sec},{msg.header.stamp.nanosec})")
        return arr

    # Extracts 3D detection data
    def parse_3d_detections(self, msg: Float32MultiArrayStamped, cat='Car'):
        if len(msg.data) > 0:
            arr = np.array(msg.data, dtype=np.float32).reshape(-1, 15)
            self.get_logger().info(f"[3D {cat}] stamp=({msg.header.stamp.sec},{msg.header.stamp.nanosec}), shape={arr.shape}")
        else:
            arr = np.empty((0,15))
            self.get_logger().info(f"[3D {cat}] (empty) => stamp=({msg.header.stamp.sec},{msg.header.stamp.nanosec})")
        return arr

    def publish_raw_detections_image(self, cv_img, detection_dict):
        if cv_img is None:
            return
        img_draw = cv_img.copy()

        # Car => Green
        if 'Car' in self.cat_list:
            arr_2d = detection_dict['Car']['2d']
            color  = (0,255,0)
            for det in arr_2d:
                x1, y1, x2, y2 = map(int, det[1:5])
                cv2.rectangle(img_draw, (x1,y1), (x2,y2), color, 2)
                cv2.putText(img_draw, "Car", (x1, max(0,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Pedestrian => Blue
        if 'Pedestrian' in self.cat_list:
            arr_2d = detection_dict['Pedestrian']['2d']
            color  = (255,0,0)
            for det in arr_2d:
                x1, y1, x2, y2 = map(int, det[1:5])
                cv2.rectangle(img_draw, (x1,y1), (x2,y2), color, 2)
                cv2.putText(img_draw, "Ped", (x1, max(0,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Publishes
        try:
            msg_out = self.bridge.cv2_to_imgmsg(img_draw, encoding='bgr8') # Converts the annotated OpenCV image back to a ROS Image message
            msg_out.header.stamp = self.get_clock().now().to_msg()
            msg_out.header.frame_id = "camera_link"
            self.raw_detections_image_pub.publish(msg_out)
        except Exception as e:
            self.get_logger().error(f"Failed to publish raw detection image: {e}")

    def fuse_and_track(self, cv_img, detection_dict): #  Fusion and multi-object tracking for each category
        cat_trackers_list = [] # Stores tracking results for each category after fusion and tracking

        for cat in self.cat_list:
            arr_2d = detection_dict[cat]['2d'] # A NumPy array of 2D detections
            arr_3d = detection_dict[cat]['3d'] # A NumPy array of 3D detections

            n2d, n3d = arr_2d.shape[0], arr_3d.shape[0] # Number of 2D and 3D detections
            self.get_logger().info(f"Fusion => {cat}, #2D={n2d}, #3D={n3d}")

            if n2d == 0 and n3d == 0:
                cat_trackers_list.append(np.empty((0,9)))
                continue

            # Frame index
            if n2d > 0:
                frame_idx = int(arr_2d[0, 0])
            else:
                frame_idx = int(arr_3d[0, 0])

            # 3D columns => [7..14] => [h,w,l,x,y,z,theta]
            dets_3d_camera = arr_3d[:, 7:14]
            ori_array = arr_3d[:, 14].reshape(-1,1) # Orientation
            other_array = arr_3d[:, 1:7]  # x1_3D,y1_3D,x2_3D,y2_3D,score_3D
            dets_3dto2d_image = arr_3d[:, 2:6] # 2D bounding box data projected from 3D detections
            additional_info = np.concatenate((ori_array, other_array), axis=1)

            # 2D => [x1,y1,x2,y2]
            dets_2d_frame = arr_2d[:, 1:5] if n2d else np.empty((0,4))

            # Data fusion
            fused_3d, only3d, only2d = data_fusion(dets_3d_camera,
                                                   dets_2d_frame,
                                                   dets_3dto2d_image,
                                                   additional_info)
            if len(only2d) > 0:
                only2d_tlwh = np.array([convert_x1y1x2y2_to_tlwh(b) for b in only2d]) # If there are any 2D-only detections, convert to the relevant format
            else:
                only2d_tlwh = np.empty((0,4))

            start_t = time.time()
            trackers_out = self.trackers[cat].update(
                fused_3d, only2d_tlwh, only3d,
                self.cfg,
                frame_idx
            )
            elapsed = time.time() - start_t # Calculates the time taken by the tracker update
            self.get_logger().info(
                f"Tracking {cat}: took {elapsed:.4f}s, #tracks_out={len(trackers_out)}"
            )

            # Some trackers might return >9 columns and they are sliced to 9 for consistency
            if trackers_out.size > 0:
                cols = trackers_out.shape[1]
                if cols > 9:
                    self.get_logger().info(f"Slicing from {cols} to 9 columns for {cat}")
                    trackers_out = trackers_out[:, :9]

            cat_trackers_list.append(trackers_out)

        return cat_trackers_list

    def publish_3d_annotated_image(self, cv_img, trackers):
        """
        Publishes an image annotated with projected 3D bounding boxes
        """
        if cv_img is None:
            return
        self.get_logger().info(f"publish_3d_annotated_image => trackers shape={trackers.shape}")

        if self.calib_str is None:
            self.get_logger().warn("No calibration data => skipping 3D image overlay.")
            return

        annotated = cv_img.copy()
        for row_idx, row in enumerate(trackers):
            self.get_logger().info(f"  row[{row_idx}] = {row}")
            if row.size < 9:
                self.get_logger().warn(f"Skipping row[{row_idx}], <9 elements.")
                continue

            track_id = int(row[0])
            # [h,w,l,x,y,z,theta] => row[1:8]
            bbox3d_tmp = row[1:8]
            color = compute_color_for_id(track_id)
            label_str = f"ID:{track_id}"

            annotated = show_image_with_boxes_3d(
                annotated,
                bbox3d_tmp,
                self.calib_str,
                track_id=track_id,
                color=color,
                label_str=label_str,
                line_thickness=2
            )

        # Publishes annotated image
        try:
            ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            ann_msg.header.stamp = self.get_clock().now().to_msg()
            ann_msg.header.frame_id = "camera_link"
            self.annotated_image_pub.publish(ann_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish 3D-annotated image: {e}")

    def publish_3d_markers(self, trackers, stamp_sec, stamp_nsec):
        """
        Publishes MarkerArray to draw the bounding boxes as lines or cubes in 3D
        """
        marker_array = MarkerArray()
        header_frame_id = "velodyne"  
        
        now_sec = stamp_sec + stamp_nsec * 1e-9
        current_time = self.get_clock().now().to_msg()

        marker_id = 0
        for row_idx, row in enumerate(trackers):
            if row.size < 9:
                continue

            track_id = int(row[0])
            h, w, l, x, y, z, theta = row[1:8]
            color_bgr = compute_color_for_id(track_id)
            # Converts BGR -> RGBA
            color_rgba = ColorRGBA(r=float(color_bgr[2])/255.0,
                                   g=float(color_bgr[1])/255.0,
                                   b=float(color_bgr[0])/255.0,
                                   a=0.8)

            # Uses LINE_LIST marker with corners
            corners_3d = compute_3d_corners_xyz([h,w,l,x,y,z,theta])
            # corners_3d is (8,3) in the same coordinate system as (x,y,z).

            # Defines the marker as a set of line segments connecting corners
            # Typical box edges: 0->1,1->2,2->3,3->0, 4->5,5->6,6->7,7->4, 0->4,1->5,2->6,3->7
            edges = [
                (0,1),(1,2),(2,3),(3,0),
                (4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7)
            ]

            marker = Marker()
            marker.header.stamp = current_time
            marker.header.frame_id = header_frame_id
            marker.ns = "3d_bboxes"
            marker.id = marker_id
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.lifetime = rclpy.duration.Duration(seconds=0, nanoseconds=0).to_msg()  # forever

            marker.scale.x = 0.05  # Line width
            marker.color = color_rgba

            # Adds the line segments
            for (i0, i1) in edges:
                p0 = corners_3d[i0]
                p1 = corners_3d[i1]
                marker.points.append(self.point_xyz(*p0))
                marker.points.append(self.point_xyz(*p1))

            marker_array.markers.append(marker)
            marker_id += 1
            
        # Publishes the array
        self.marker_pub.publish(marker_array)

    def point_xyz(self, x, y, z):
        from geometry_msgs.msg import Point
        return Point(x=float(x), y=float(y), z=float(z))

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

