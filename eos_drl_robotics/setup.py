from setuptools import setup

package_name = 'eos_drl_robotics'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Prithviraj Domadia',
    maintainer_email='pdomadia@robotics.edu',
    description='Edge-Optimized Sensor Fusion: A Unified SLAM-DRL Framework for Autonomous Service Robots',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = eos_drl_robotics.perception_node:main',
            'drl_navigator_node = eos_drl_robotics.drl_navigator_node:main',
            'kinematics_controller = eos_drl_robotics.kinematics_controller:main',
        ],
    },
)
