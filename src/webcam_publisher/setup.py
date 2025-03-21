from setuptools import find_packages, setup

package_name = 'webcam_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shadow0',
    maintainer_email='jztshadow0@gmail.com',
    description='webcam to publish the frames',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'webcam_publisher_node = webcam_publisher.webcam_publisher_node:main'
        ],
    },
)
