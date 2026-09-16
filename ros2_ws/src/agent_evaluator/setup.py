from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'agent_evaluator'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name] if os.path.exists('resource/' + package_name) else []),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dhruv Gupta',
    maintainer_email='epost.dhruv@gmail.com',
    description='Ground Truth Oracle and Evaluation Supervisor for AERO',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'oracle_node = agent_evaluator.oracle_node:main',
        ],
    },
)
