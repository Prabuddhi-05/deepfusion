#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='deepfusionmot',
            executable='main_rt_fusion_node',
            name='deepfusionmot_fusion',
            output='screen',
            emulate_tty=True
        ),
        # Possibly start other nodes or tools, e.g. a detection node, rviz, etc.
    ])

