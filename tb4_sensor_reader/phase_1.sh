#!/bin/bash

source ~/ros2_ws/install/setup.bash

echo "Launching SLAM..."
gnome-terminal -- ros2 launch turtlebot4_navigation slam.launch.py

sleep 3

echo "Launching teleop..."
gnome-terminal -- ros2 run teleop_twist_keyboard teleop_twist_keyboard

# RUN TS TO RUN THE SCRIPT
# chmod +x phase1.sh
#./phase1.sh

# AFTER TELEOP RUN TS TO SAVE MAP
# ros2 run nav2_map_server map_saver_cli -f ~/Desktop/demo_map
# ros2 service call /T<ID>/slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '$HOME/Desktop/demo_map'}"