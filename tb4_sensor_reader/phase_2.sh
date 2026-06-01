# Before this script runs, the terminals open from Phase 1 must be manually closed.
# Also, just once on PC startup, run "chmod +x phase_2.sh"

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
ros2 action send_goal /$TB/undock irobot_create_msgs/action/Undock "{}"

sleep 1

# This must output x: 0.0, y: 0.0
gnome-terminal -- bash -c "source ~/.bashrc; source ~/ros2_ws/install/setup.bash; set-turtlebot 21; ros2 service call /T21/reset_pose irobot_create_msgs/srv/ResetPose {}; sleep 1; ros2 topic echo /T21/odom --field pose.pose.position --once; exec bash"

sleep 2

gnome-terminal -- bash -c "source ~/.bashrc; source ~/ros2_venv/bin/activate; source ~/ros2_ws/install/setup.bash; set-turtlebot 21; echo ''; read -p 'Press Enter to start autonomous search...' _; ~/ros2_venv/bin/python3 -m tb4_sensor_reader.autonomous_search; exec bash"

echo "Robot is now in autonomous navigation mode."

