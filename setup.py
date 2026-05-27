from setuptools import find_packages, setup

package_name = 'tb4_sensor_reader'

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
    maintainer='jdu853',
    maintainer_email='jdu853@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
		'console_scripts': [
		    'odom_reader = tb4_sensor_reader.odom_reader:main',
		    'motion_controller = tb4_sensor_reader.motion_controller:main',
		    'reactive_controller = tb4_sensor_reader.reactive_controller:main',
		    'avoid_controller = tb4_sensor_reader.avoid_controller:main',
		    'test_node = tb4_sensor_reader.test_node_template:main',
		    'test_odom_linear = tb4_sensor_reader.test_odom_linear:main',
		    'test_odom_square = tb4_sensor_reader.test_odom_square:main',
		    'lidar_b2_node = tb4_sensor_reader.lidar_b2_node:main',
		    'pose_reader = tb4_sensor_reader.pose_reader:main',
		    'physical_motion = tb4_sensor_reader.physical_motion:main',
		    'reactive_physical = tb4_sensor_reader.reactive_physical:main',
		    'avoidance_physical = tb4_sensor_reader.avoidance_physical:main',
		    'camera_viewer = tb4_sensor_reader.camera_viewer:main',
		    'camera_detector = tb4_sensor_reader.camera_detector:main',
		    'detect_and_stop = tb4_sensor_reader.detect_and_stop:main',
		    'autonomous_search = tb4_sensor_reader.autonomous_search:main',
    	],
	},
)
