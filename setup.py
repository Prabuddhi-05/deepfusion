from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'deepfusionmot' # Package name

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
        'data_fusion = deepfusionmot.data_fusion:main',
        'DeepFusionMOT = deepfusionmot.DeepFusionMOT:main',
        'main_rt_fusion_node = deepfusionmot.main_rt_fusion_node:main',
        'tracker = deepfusionmot.tracker:main',
        'calibration = deepfusionmot.calibration:main',
        'coordinate_transformation = deepfusionmot.coordinate_transformation:main',
        'cost_function = deepfusionmot.cost_function:main',
        'kalman_filter_2d = deepfusionmot.kalman_filter_2d:main',
        'kalman_filter_3d = deepfusionmot.kalman_filter_3d:main',
        'matching = deepfusionmot.matching:main',
        'track_2d = deepfusionmot.track_2d:main',
        'track_3d = deepfusionmot.track_3d:main',
        'kitti_oxts = deepfusionmot.kitti_oxts:main',
        'file = deepfusionmot.file:main',
        'config = deepfusionmot.config:main',
        'rosbag_convert = deepfusionmot.rosbag_convert:main',
        ],
    },
)
