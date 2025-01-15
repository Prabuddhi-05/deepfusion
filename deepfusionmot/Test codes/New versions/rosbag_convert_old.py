import os  
import rclpy  
from rclpy.serialization import serialize_message
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from sensor_msgs.msg import Image, PointCloud2, PointField  
from std_msgs.msg import String  
import numpy as np  
import cv2
import struct 
from cv_bridge import CvBridge

# Import the custom message
from my_msgs.msg import Float32MultiArrayStamped

def kitti_to_rosbag(base_path, output_bag_path):
    """
    Convert KITTI dataset sequences into ROS 2 bag files with 
    detection messages that include a Header (and thus a timestamp).
    """
    rclpy.init()  

    for seq_id in range(1):
        seq_str = str(seq_id).zfill(4)
        bag_file_path = os.path.join(output_bag_path, f"{seq_str}.db3")
        metadata_file_path = os.path.join(output_bag_path, f"{seq_str}_metadata.yaml")

        storage_options = StorageOptions(uri=bag_file_path, storage_id='sqlite3')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        writer = SequentialWriter()
        writer.open(storage_options, converter_options)

        topics = {
            "calibration": '/camera/calibration',
            "image": '/camera/image_raw',
            "point_cloud": '/lidar/points',
            "detection_2d_car": '/detection_2d/car',
            "detection_2d_pedestrian": '/detection_2d/pedestrian',
            "detection_3d_car": '/detection_3d/car',
            "detection_3d_pedestrian": '/detection_3d/pedestrian',
            "odometry": '/sensor/odometry',
        }

        for topic_name, ros_type in [
            (topics["calibration"], 'std_msgs/msg/String'),
            (topics["image"], 'sensor_msgs/msg/Image'),
            (topics["point_cloud"], 'sensor_msgs/msg/PointCloud2'),
            (topics["detection_2d_car"], 'my_msgs/msg/Float32MultiArrayStamped'),
            (topics["detection_2d_pedestrian"], 'my_msgs/msg/Float32MultiArrayStamped'),
            (topics["detection_3d_car"], 'my_msgs/msg/Float32MultiArrayStamped'),
            (topics["detection_3d_pedestrian"], 'my_msgs/msg/Float32MultiArrayStamped'),
            (topics["odometry"], 'std_msgs/msg/String'),
        ]:
            metadata = TopicMetadata(
                name=topic_name,
                type=ros_type,
                serialization_format='cdr'
            )
            writer.create_topic(metadata)

        bridge = CvBridge()
        base_time_ns = int(1e9)  # start at 1 second

        calib_file = os.path.join(base_path, 'calib', f'{seq_str}.txt')
        image_dir = os.path.join(base_path, 'image_02', seq_str)
        point_cloud_dir = os.path.join(base_path, 'velodyne', seq_str)

        detection_2d_car_file = os.path.join(
            '/home/prabuddhi/DeepFusionMOT/data/detections/2D/rrc/training/Car',
            f'{seq_str}.txt'
        )
        detection_2d_pedestrian_file = os.path.join(
            '/home/prabuddhi/DeepFusionMOT/data/detections/2D/rrc/training/Pedestrian',
            f'{seq_str}.txt'
        )
        detection_3d_car_file = os.path.join(
            '/home/prabuddhi/DeepFusionMOT/data/detections/3D/pointrcnn/training/Car',
            f'{seq_str}.txt'
        )
        detection_3d_pedestrian_file = os.path.join(
            '/home/prabuddhi/DeepFusionMOT/data/detections/3D/pointrcnn/training/Pedestrian',
            f'{seq_str}.txt'
        )

        odometry_file = os.path.join(base_path, 'oxts', f'{seq_str}.txt')

        image_files = sorted(os.listdir(image_dir))
        point_cloud_files = sorted(os.listdir(point_cloud_dir))

        images_written = 0
        point_clouds_written = 0

        for idx, image_file in enumerate(image_files):
            timestamp_ns = base_time_ns + idx * int(1e8)

            # --- Write Image ---
            image_path = os.path.join(image_dir, image_file)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Failed to read image {image_path}. Skipping.")
                continue

            try:
                img_msg = bridge.cv2_to_imgmsg(img, encoding='bgr8')
                img_msg.header.stamp.sec = timestamp_ns // int(1e9)
                img_msg.header.stamp.nanosec = timestamp_ns % int(1e9)
                img_msg.header.frame_id = 'camera_link'
                writer.write(topics["image"], serialize_message(img_msg), timestamp_ns)
                images_written += 1
            except Exception as e:
                print(f"Error converting image {image_path}: {e}")
                continue

            # --- Write PointCloud2 (if exists) ---
            if idx < len(point_cloud_files):
                pc_file = os.path.join(point_cloud_dir, point_cloud_files[idx])
                pc_data = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)
                pc_msg = create_pointcloud2(pc_data, timestamp_ns)
                writer.write(topics["point_cloud"], serialize_message(pc_msg), timestamp_ns)
                point_clouds_written += 1

            write_2d_detections_stamped(
                writer,
                detection_2d_car_file,
                topics["detection_2d_car"],
                timestamp_ns,
                frame_idx=idx,
                seq_str=seq_str,
                category='car'
            )
            write_2d_detections_stamped(
                writer,
                detection_2d_pedestrian_file,
                topics["detection_2d_pedestrian"],
                timestamp_ns,
                frame_idx=idx,
                seq_str=seq_str,
                category='pedestrian'
            )

            write_3d_detections_stamped(
                writer,
                detection_3d_car_file,
                topics["detection_3d_car"],
                timestamp_ns,
                frame_idx=idx,
                seq_str=seq_str,
                category='car'
            )
            write_3d_detections_stamped(
                writer,
                detection_3d_pedestrian_file,
                topics["detection_3d_pedestrian"],
                timestamp_ns,
                frame_idx=idx,
                seq_str=seq_str,
                category='pedestrian'
            )

            write_odometry(writer, odometry_file, topics["odometry"], timestamp_ns, idx)

            if os.path.exists(calib_file):
                with open(calib_file, 'r') as f:
                    calib_msg = String(data=f.read())
                    writer.write(topics["calibration"], serialize_message(calib_msg), timestamp_ns)

        print(f"Sequence {seq_str}: Processed {images_written} images and {point_clouds_written} point clouds.")

        metadata_generated_path = bag_file_path + ".metadata.yaml"
        if os.path.exists(metadata_generated_path):
            os.rename(metadata_generated_path, metadata_file_path)

    rclpy.shutdown()

def write_2d_detections_stamped(writer, det_file, topic, timestamp_ns, frame_idx, seq_str, category):
    if not os.path.exists(det_file):
        return
    with open(det_file, 'r') as f:
        for line in f:
            det_data = list(map(float, line.strip().split(',')))
            frame_num = int(det_data[0])
            if frame_num == frame_idx:
                print(f"Writing 2D detection for seq={seq_str}, frame={frame_idx}, cat={category}, data={det_data}")
                det_msg = Float32MultiArrayStamped()
                det_msg.data = det_data
                det_msg.header.stamp.sec = timestamp_ns // int(1e9)
                det_msg.header.stamp.nanosec = timestamp_ns % int(1e9)
                det_msg.header.frame_id = "detection_2d_link"
                writer.write(topic, serialize_message(det_msg), timestamp_ns)

def write_3d_detections_stamped(writer, det_file, topic, timestamp_ns, frame_idx, seq_str, category):
    if not os.path.exists(det_file):
        return
    with open(det_file, 'r') as f:
        for line in f:
            det_data = list(map(float, line.strip().split(',')))
            frame_num = int(det_data[0])
            if frame_num == frame_idx:
                print(f"Writing 3D detection for seq={seq_str}, frame={frame_idx}, cat={category}, data={det_data}")
                det_msg = Float32MultiArrayStamped()
                det_msg.data = det_data
                det_msg.header.stamp.sec = timestamp_ns // int(1e9)
                det_msg.header.stamp.nanosec = timestamp_ns % int(1e9)
                det_msg.header.frame_id = "detection_3d_link"
                writer.write(topic, serialize_message(det_msg), timestamp_ns)

def create_pointcloud2(points, timestamp_ns):
    from sensor_msgs.msg import PointCloud2, PointField
    msg = PointCloud2()
    msg.header.stamp.sec = timestamp_ns // int(1e9)
    msg.header.stamp.nanosec = timestamp_ns % int(1e9)
    msg.header.frame_id = 'velodyne'
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * points.shape[0]
    msg.data = b''.join([struct.pack('ffff', *p) for p in points])
    msg.is_dense = True
    return msg

def write_odometry(writer, odo_file, topic, timestamp_ns, frame_idx):
    from std_msgs.msg import String
    if not os.path.exists(odo_file):
        print(f"Warning: no odometry file at {odo_file}")
        return
    with open(odo_file, 'r') as f:
        lines = f.readlines()
    total_frames = len(lines)
    if frame_idx >= total_frames:
        print(f"Warning: frame idx {frame_idx} out of range (max {total_frames}).")
        return
    raw_oxts_data = lines[frame_idx].strip()
    oxts_msg = String()
    oxts_msg.data = raw_oxts_data
    writer.write(topic, serialize_message(oxts_msg), timestamp_ns)

def main():
    kitti_to_rosbag(
        '/home/prabuddhi/DeepFusionMOT_test/data/kitti/tracking/training',
        '/home/prabuddhi/ros2_ws1/src/deepfusionmot/data/kitti/tracking/training_new_det'
    )

if __name__ == '__main__':
    main()

