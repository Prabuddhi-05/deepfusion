# Real-Time Multimodal Fusion Node (DeepFusion)

This ROS 2 node performs **real-time fusion** of 2D and 3D detections using synchronized camera and LiDAR inputs. It supports object categories like **Car** and **Pedestrian**, and publishes fused and unmatched detections for downstream use and visualization.

---

## Features

- Real-time fusion of 2D + 3D detections (no tracking)
- Supports both Car and Pedestrian categories
- Publishes:
  - `/fusion/fused_3d`: Fused 3D detections
  - `/fusion/only_3d`: Unmatched 3D detections
  - `/fusion/only_2d`: Unmatched 2D detections
  - `/raw_detections_image`: Camera image with raw 2D bounding boxes drawn
- Calibration subscriber (`/camera/calibration`) that accepts KITTI-style calibration text
- Works with synchronized camera, LiDAR, and detection topics

---

## Topics

| Topic Name               | Type                          | Description                                |
|--------------------------|-------------------------------|--------------------------------------------|
| `/fusion/fused_3d`       | `Float32MultiArrayStamped`    | Fused 3D bounding boxes                    |
| `/fusion/only_3d`        | `Float32MultiArrayStamped`    | 3D detections not matched with 2D         |
| `/fusion/only_2d`        | `Float32MultiArrayStamped`    | 2D detections not matched with 3D         |
| `/raw_detections_image`  | `sensor_msgs/Image`           | Image with raw 2D bounding boxes overlayed|
| `/camera/calibration`    | `std_msgs/String`             | KITTI-style calibration input              |

---

## Configuration: `kitti_real_time.yaml`

You can configure which object categories are used for fusion using the `cat_list` parameter in the config file located at:

```
deepfusion/config/kitti_real_time.yaml
```

Example:
```yaml
dataset: 'kitti'

# Use either one or both:
# cat_list: ['Car']
# cat_list: ['Pedestrian']
cat_list: ['Car', 'Pedestrian']

Car:
  metric_3d: iou_3d
  metric_2d: iou_2d
  max_ages: 25
  min_frames: 3
  cost_function:
    iou_2d: 0.5
    sdiou_2d: 1.1
    giou_3d: -0.2
    iou_3d: 0.01
    dist_3d: 1.1

Pedestrian:
  metric_3d: iou_3d
  metric_2d: iou_2d
  max_ages: 25
  min_frames: 3
  cost_function:
    iou_2d: 0.5
    sdiou_2d: 0.2
    giou_3d: -0.2
    iou_3d: 0.01
    dist_3d: 1.1
```

> 🛠️ You can modify `cat_list` to control which object classes are included in the fusion process (Car, Pedestrian, or both). Tracking-related fields (like `max_ages`, `min_frames`, etc.) are included for future use and currently do not affect fusion.

---

## Usage Instructions

### 1. Clone the Repository

```bash
cd <your_ros2_workspace>/src
git clone https://github.com/Prabuddhi-05/deepfusion.git
```

### 2. Build the Package

```bash
cd <your_ros2_workspace>
colcon build --packages-select deepfusion
source install/setup.bash
```

### 3. Launch the Node

Make sure your ROS 2 bag with the required topics is playing in the background.

Then run:

```bash
ros2 launch deepfusion deepfusion.launch.py
```

---

## Launch File: `deepfusion.launch.py`

This launch file runs the node `main_rt_fusion_node.py` located in the `deepfusion` package:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='deepfusion',
            executable='main_rt_fusion_node',
            name='deepfusion_fusion',
            output='screen',
            emulate_tty=True,
            parameters=[{"use_sim_time": True}]
        )
    ])
```

---

## Notes

- The node uses `message_filters.TimeSynchronizer` to ensure exact-time sync between image, point cloud, and detections.
- Make sure your ROS 2 bag contains the topics:
  - `/camera/image_raw`
  - `/lidar/points`
  - `/detection_2d/car`, `/detection_3d/car`
  - `/detection_2d/pedestrian`, `/detection_3d/pedestrian`
  - `/camera/calibration` (as a `std_msgs/String` with KITTI-format calibration)
