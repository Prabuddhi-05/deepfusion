from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'deepfusion' # Package name

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='prabuddhi',
    maintainer_email='26619055@students.lincoln.ac.uk',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'data_fusion = deepfusion.data_fusion:main',
        'DeepFusionMOT = deepfusion.DeepFusionMOT:main',
        'main_rt_fusion_node = deepfusion.main_rt_fusion_node:main',
        'tracker = deepfusion.tracker:main',
        'calibration = deepfusion.calibration:main',
        'coordinate_transformation = deepfusion.coordinate_transformation:main',
        'cost_function = deepfusion.cost_function:main',
        'kalman_filter_2d = deepfusion.kalman_filter_2d:main',
        'kalman_filter_3d = deepfusion.kalman_filter_3d:main',
        'matching = deepfusion.matching:main',
        'track_2d = deepfusion.track_2d:main',
        'track_3d = deepfusion.track_3d:main',
        'kitti_oxts = deepfusion.kitti_oxts:main',
        'file = deepfusion.file:main',
        'config = deepfusion.config:main',
        'rosbag_convert = deepfusion.rosbag_convert:main',
        ],
    },
)
