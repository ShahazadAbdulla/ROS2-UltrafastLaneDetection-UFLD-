from setuptools import find_packages, setup

package_name = 'ufld_lane_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=['ufld_lane_detection', 'ufld_lane_detection.*']),  # Updated to include submodules
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shadow0',
    maintainer_email='jztshadow0@gmail.com',
    description='lane_detection_node for Abaja',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_detection_node = ufld_lane_detection.lane_detection_node:main'
        ],
    },
)
