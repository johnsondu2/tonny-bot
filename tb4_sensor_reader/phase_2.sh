#!/bin/bash
# Before this script runs, close all Phase 1 terminals.
# First time only: chmod +x phase_2.sh

source ~/.bashrc
source ~/ros2_ws/install/setup.bash

cd ~/ros2_ws && colcon build --packages-select tb4_sensor_reader
source ~/ros2_ws/install/setup.bash

clear

read -p "Enter TurtleBot number: " TB_NUM
TB="T$TB_NUM"
echo "Using TurtleBot: $TB"

set-turtlebot $TB_NUM
sleep 1

echo "Undocking..."
ros2 action send_goal /$TB/undock irobot_create_msgs/action/Undock "{}"
sleep 5

echo "Resetting odometry..."
ros2 service call /$TB/reset_pose irobot_create_msgs/srv/ResetPose {}
sleep 1

echo "Verifying reset (must show x: 0.0  y: 0.0):"
ros2 topic echo /$TB/odom --field pose.pose.position --once
sleep 1

echo ""
read -p "Odometry looks correct? Press Enter to start autonomous search, Ctrl+C to abort..." _

source ~/ros2_venv/bin/activate
~/ros2_venv/bin/python3 -m tb4_sensor_reader.autonomous_search