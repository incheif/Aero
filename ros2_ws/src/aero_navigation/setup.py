from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'aero_navigation'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name] if os.path.exists('resource/' + package_name) else []),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dhruv Gupta',
    maintainer_email='epost.dhruv@gmail.com',
    description='Cognitive Semantic Navigation with Gemma, Nav2, SLAM, and Frontier Exploration',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gemma_cognitive_node = aero_navigation.gemma_cognitive_node:main',
            'frontier_explorer_node = aero_navigation.frontier_explorer_node:main',
            'semantic_mapper_node = aero_navigation.semantic_mapper_node:main',
        ],
    },
)
