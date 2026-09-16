from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robot_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name] if os.path.exists('resource/' + package_name) else []),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dhruv Gupta',
    maintainer_email='epost.dhruv@gmail.com',
    description='Target Robot Controller package for AERO',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller_node = robot_controller.controller_node:main',
        ],
    },
)
