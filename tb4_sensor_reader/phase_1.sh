# Before running this script, must run:
# set-turtlebot XX (whatever number the Turtlebot is)
# sanity && ros2 topic list (optional: to confirm that Turtlebot is connected)

#!/bin/bash

# Source ROS2 environment
source ~/.bashrc
source ~/ros2_ws/install/setup.bash

clear 

# Ask user for TurtleBot number
read -p "Enter TurtleBot number: " TB_NUM

# Create namespace variable
TB="T$TB_NUM"

echo "Using TurtleBot: $TB"

# Optional automatic setup
set-turtlebot $TB_NUM

sleep 1

echo "Undocking..."

ros2 action send_goal /T21/undock irobot_create_msgs/action/Undock "{}"

sleep 1

echo "Launching SLAM..."
gnome-terminal -- bash -c "source ~/.bashrc; source ~/ros2_ws/install/setup.bash; set-turtlebot 21; ros2 service call /T21/reset_pose irobot_create_msgs/srv/ResetPose {}; ros2 launch turtlebot4_navigation slam.launch.py namespace:=/T21; exec bash"

sleep 1

echo "Launching RViz..."
gnome-terminal -- bash -c "source ~/.bashrc; source ~/ros2_ws/install/setup.bash; set-turtlebot 21; ros2 launch turtlebot4_viz view_robot.launch.py namespace:=/T21; exec bash"

sleep 2

echo "Launching teleop..."
gnome-terminal -- bash -c "source ~/.bashrc; source ~/ros2_ws/install/setup.bash; set-turtlebot 21; ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/T21/cmd_vel; exec bash"

# echo ""
# echo "To save the map later:"
# echo "ros2 run nav2_map_server map_saver_cli -f ~/Desktop/demo_map --ros-args -r map:=/$TB/map"

# echo ""
# echo "Serialize pose graph:"
# echo "ros2 service call /$TB/slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph \"{filename: '\$HOME/Desktop/demo_map'}\""

echo ""
echo "================================================="
echo "Mapping running..."
echo "When you are DONE mapping:"
echo "1. Stop teleop (Ctrl+C in teleop terminal)"
echo "2. Come back here"
echo "3. Press ENTER to AUTO SAVE MAP"
echo "================================================="

read -p "Press ENTER to save map..."

echo "Saving map..."

ros2 action send_goal /$TB/dock irobot_create_msgs/action/Dock "{}"

sleep 1

ros2 run nav2_map_server map_saver_cli \
-f ~/Downloads/map/map \
--ros-args -r map:=/T21/map

sleep 5

echo "Map saved!"

echo "Saving SLAM pose graph..."

ros2 service call /T21/slam_toolbox/serialize_map \
slam_toolbox/srv/SerializePoseGraph \
"{filename: '$HOME/Downloads/map/map'}"

sleep 3

cd ~/ros2_ws/src/tb4_sensor_reader/tb4_sensor_reader
python3 path_planner_v2.py

sleep 1

echo "DONE: Map + pose graph + path planning csv saved to Downloads"