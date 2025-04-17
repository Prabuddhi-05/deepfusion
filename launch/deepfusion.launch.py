#!/usr/bin/env python3
import os

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
        ),
    ])

