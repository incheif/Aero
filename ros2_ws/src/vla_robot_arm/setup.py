from setuptools import setup
import os
from glob import glob

package_name = 'vla_robot_arm'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Robotics Developer',
    maintainer_email='developer@example.com',
    description='ROS 2 Google VLA Autonomous Arm Manipulation Package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vla_cognitive_node = vla_robot_arm.vla_cognitive_node:main',
            'semantic_spatial_mapper_node = vla_robot_arm.semantic_spatial_mapper_node:main',
            'frontier_explorer_node = vla_robot_arm.frontier_explorer_node:main',
        ],
    },
)
