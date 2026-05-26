#!/usr/bin/env python3
"""
Test Node Template — Sensor Investigation
COMPSYS732

Use this template as the starting point for your autonomous test nodes.
It combines the publisher pattern from Task 5 (to command motion) with
the subscriber pattern from Task 4 (to read sensor data).

Fill in the sections marked TODO to implement your specific test.

To run:
    ros2 run tb4_sensor_reader test_node
    (after adding 'test_node = tb4_sensor_reader.test_node:main' to setup.py)
"""

import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

# ── TODO: Set your robot namespace ─────────────────────────────────────────────
NAMESPACE = '/T21'           # Change to your robot e.g. /T10

# ── TODO: Define your motion parameters ────────────────────────────────────────
FORWARD_SPEED = 0.3          # m/s  — linear velocity when driving forward
TURN_SPEED    = 0.5          # rad/s — angular velocity when turning
DURATION = 30


# Example durations for known distances/angles:
#   Drive 1.0 m at 0.2 m/s  → duration = 1.0 / 0.2 = 5.0 seconds
#   Turn 90°  at 0.5 rad/s  → duration = (pi/2) / 0.5 = 3.14 seconds
#   Turn 360° at 0.5 rad/s  → duration = (2*pi) / 0.5 = 12.57 seconds






class TestNode(Node):

    def __init__(self):
        super().__init__('test_node')

        # ── Publishers ───────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(
            Twist,
            f'{NAMESPACE}/cmd_vel',
            10)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.odom_sub = self.create_subscription(
            Odometry,
            f'{NAMESPACE}/odom',
            self.odom_callback,
            10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            f'{NAMESPACE}/scan',
            self.scan_callback,
            10)

        # ── State variables ──────────────────────────────────────────────────
        # These store the latest sensor readings so any callback can use them
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_yaw = 0.0   # degrees
        self.nearest_obstacle = float('inf')   # metres

        # ── TODO: Add your own state variables ───────────────────────────────
        # Example: track which phase of your test sequence you are in
        self.phase     = 0
        self.phase_start_time = None
        self.test_done = False

        # ── Control loop timer (runs every 0.1 seconds = 10 Hz) ─────────────
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Test node started')

    # ── Odometry callback ────────────────────────────────────────────────────

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        self.current_x = pos.x
        self.current_y = pos.y

        # Convert quaternion to yaw in degrees
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))

    # ── LiDAR callback ───────────────────────────────────────────────────────

    def scan_callback(self, msg):
        valid = [r for r in msg.ranges if msg.range_min <= r <= msg.range_max]
        self.nearest_obstacle = min(valid) if valid else float('inf')

    # ── Helper: publish a velocity command ───────────────────────────────────

    def drive(self, linear, angular):
        """Publish a Twist command. Call with (0, 0) to stop."""
        msg = Twist()
        msg.linear.x  = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def stop(self):
        self.drive(0.0, 0.0)

    # ── Control loop ─────────────────────────────────────────────────────────

    def control_loop(self):
        """
        Called at 10 Hz. Implement your test sequence here using self.phase
        to track which step of the sequence you are in.

        Pattern:
            Phase 0 → do action A for N seconds → advance to phase 1
            Phase 1 → do action B for N seconds → advance to phase 2
            ...
            Final phase → stop, log results, set self.test_done = True
        """

        if self.test_done:
            self.stop()
            return

        now = self.get_clock().now().nanoseconds / 1e9   # current time in seconds

        # Phase 0 — Wait 2 seconds for connections to establish
        if self.phase == 0:
            if self.phase_start_time is None:
                self.phase_start_time = now
                self.get_logger().info('Phase 0: Waiting for connections...')
            if now - self.phase_start_time >= 2.0:
                self.get_logger().info('Ready — starting test')
                self.phase += 1
                self.phase_start_time = None

        # Phase 1 — Drive forward for DURATION seconds
        elif self.phase == 1:
            if self.phase_start_time is None:
                self.phase_start_time = now
                self.get_logger().info('Phase 1: Driving forward')
            elapsed = now - self.phase_start_time
            if elapsed < DURATION:
                self.drive(FORWARD_SPEED, 0.0)
            else:
                self.stop()
                self.get_logger().info(
                    f'Phase 1 complete | '
                    f'Final position x: {self.current_x:.4f} m  y: {self.current_y:.4f} m')
                self.phase += 1
                self.phase_start_time = None

        # Phase 2 — Test complete
        elif self.phase == 2:
            self.stop()
            self.get_logger().info('Test sequence complete — stopping')
            self.get_logger().info(
                f'Final pose: x={self.current_x:.4f}  y={self.current_y:.4f}  '
                f'yaw={self.current_yaw:.2f} deg')
            self.test_done = True

        # ── TODO: Add more phases for your specific test ──────────────────────


def main(args=None):
    rclpy.init(args=args)
    node = TestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
