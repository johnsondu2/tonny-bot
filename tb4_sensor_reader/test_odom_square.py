#!/usr/bin/env python3

import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from irobot_create_msgs.srv import ResetPose   # ADDED: for true odom reset
from std_msgs.msg import Bool                  # ← NEW: for logger trigger signal

# ── TODO: Set your robot namespace ─────────────────────────────────────────────
NAMESPACE = '/T23'           # Change to your robot e.g. /T10

# ── TODO: Define your motion parameters ────────────────────────────────────────
FORWARD_SPEED = 0.2          # m/s  — linear velocity when driving forward. MAX 0.3
TURN_SPEED    = 1.5         # rad/s — angular velocity when turning. MAX 1.9 (+CCW -CW)
DURATION = 5.0	# ADJUST BASED ON DISTANCE	


# Example durations for known distances/angles:
#   Drive 1.0 m at 0.2 m/s  → duration = 1.0 / 0.2 = 5.0 seconds
#   



class TestNode(Node):

    def __init__(self):
        super().__init__('test_node')

        # ── Publishers ───────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(
            Twist,
            f'{NAMESPACE}/cmd_vel',
            10)

        # ← NEW: publisher that signals odom_logger to start recording
        self.trigger_pub = self.create_publisher(
            Bool,
            f'{NAMESPACE}/logger_trigger',
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

        # ADDED: true pose reset service client so odom_logger sees zeroed /odom
        self.reset_pose_client = self.create_client(
            ResetPose,
            f'{NAMESPACE}/reset_pose'
        )
        self.reset_pose_requested = False
        self.reset_pose_done = False
        self.reset_pose_future = None

        # ← NEW: track whether we've sent the trigger yet
        self.trigger_sent = False

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

    # ADDED: real odometry reset for Create 3 / TurtleBot 4
    def request_reset_pose(self):
        if not self.reset_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f'Reset pose service not available: {NAMESPACE}/reset_pose')
            return False

        req = ResetPose.Request()
        self.reset_pose_future = self.reset_pose_client.call_async(req)
        self.reset_pose_requested = True
        self.get_logger().info('Requested reset_pose service')
        return True

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

            # CHANGED: request true odom reset once
            if not self.reset_pose_requested and now - self.phase_start_time >= 2.0:
                self.request_reset_pose()

            # CHANGED: wait for reset service to complete before starting
            if self.reset_pose_requested and not self.reset_pose_done:
                if self.reset_pose_future.done():
                    try:
                        self.reset_pose_future.result()
                        self.reset_pose_done = True
                        self.phase_start_time = now   # restart timer after reset
                        self.get_logger().info('reset_pose complete')
                    except Exception as e:
                        self.get_logger().error(f'reset_pose failed: {e}')
                        self.test_done = True
                        return

            # CHANGED: small delay after reset so odom_logger gets zeroed samples
            if self.reset_pose_done and now - self.phase_start_time >= 1.0:
                # ← NEW: send trigger to odom_logger so it starts recording now
                if not self.trigger_sent:
                    self.trigger_pub.publish(Bool(data=True))
                    self.trigger_sent = True
                    self.get_logger().info('Logger trigger sent — odom_logger will start recording')
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

        # Phase 2 — Turn 90 degrees (π/2 rad at TURN_SPEED rad/s)
        elif self.phase == 2:
            turn_duration = (math.pi / 2) / abs(TURN_SPEED)
            if self.phase_start_time is None:
                self.phase_start_time = now
                self.get_logger().info('Phase 2: Turning 90 degrees')
            elapsed = now - self.phase_start_time
            if elapsed < turn_duration:
                self.drive(0.0, TURN_SPEED)
            else:
                self.stop()
                self.get_logger().info(
                    f'Phase 2 complete | Heading: {self.current_yaw:.2f} deg')
                self.phase += 1
                self.phase_start_time = None

        # Phase 3 — Drive forward for DURATION seconds
        elif self.phase == 3:
            if self.phase_start_time is None:
                self.phase_start_time = now
                self.get_logger().info('Phase 1: Driving forward')
            elapsed = now - self.phase_start_time
            if elapsed < DURATION:
                self.drive(FORWARD_SPEED, 0.0)
            else:
                self.stop()
                self.get_logger().info(
                    f'Phase 3 complete | '
                    f'Final position x: {self.current_x:.4f} m  y: {self.current_y:.4f} m')
                self.phase += 1
                self.phase_start_time = None

        # Phase 4 — Turn 90 degrees (π/2 rad at TURN_SPEED rad/s)
        elif self.phase == 4:
            turn_duration = (math.pi / 2) / abs(TURN_SPEED)
            if self.phase_start_time is None:
                self.phase_start_time = now
                self.get_logger().info('Phase 4: Turning 90 degrees')
            elapsed = now - self.phase_start_time
            if elapsed < turn_duration:
                self.drive(0.0, TURN_SPEED)
            else:
                self.stop()
                self.get_logger().info(
                    f'Phase 4 complete | Heading: {self.current_yaw:.2f} deg')
                self.phase += 1
                self.phase_start_time = None
                
        # Phase 5 — Drive forward for DURATION seconds
        elif self.phase == 5:
            if self.phase_start_time is None:
                self.phase_start_time = now
                self.get_logger().info('Phase 5: Driving forward')
            elapsed = now - self.phase_start_time
            if elapsed < DURATION:
                self.drive(FORWARD_SPEED, 0.0)
            else:
                self.stop()
                self.get_logger().info(
                    f'Phase 5 complete | '
                    f'Final position x: {self.current_x:.4f} m  y: {self.current_y:.4f} m')
                self.phase += 1
                self.phase_start_time = None

        # Phase 6 — Turn 90 degrees (π/2 rad at TURN_SPEED rad/s)
        elif self.phase == 6:
            turn_duration = (math.pi / 2) / abs(TURN_SPEED)
            if self.phase_start_time is None:
                self.phase_start_time = now
                self.get_logger().info('Phase 6: Turning 90 degrees')
            elapsed = now - self.phase_start_time
            if elapsed < turn_duration:
                self.drive(0.0, TURN_SPEED)
            else:
                self.stop()
                self.get_logger().info(
                    f'Phase 6 complete | Heading: {self.current_yaw:.2f} deg')
                self.phase += 1
                self.phase_start_time = None
                
        # Phase 7 — Drive forward for DURATION seconds
        elif self.phase == 7:
            if self.phase_start_time is None:
                self.phase_start_time = now
                self.get_logger().info('Phase 7: Driving forward')
            elapsed = now - self.phase_start_time
            if elapsed < DURATION:
                self.drive(FORWARD_SPEED, 0.0)
            else:
                self.stop()
                self.get_logger().info(
                    f'Phase 7 complete | '
                    f'Final position x: {self.current_x:.4f} m  y: {self.current_y:.4f} m')
                self.phase += 1
                self.phase_start_time = None

        # Phase 8 — Turn 90 degrees (π/2 rad at TURN_SPEED rad/s)
        elif self.phase == 8:
            turn_duration = (math.pi / 2) / abs(TURN_SPEED)
            if self.phase_start_time is None:
                self.phase_start_time = now
                self.get_logger().info('Phase 8: Turning 90 degrees')
            elapsed = now - self.phase_start_time
            if elapsed < turn_duration:
                self.drive(0.0, TURN_SPEED)
            else:
                self.stop()
                self.get_logger().info(
                    f'Phase 8 complete | Heading: {self.current_yaw:.2f} deg')
                self.phase += 1
                self.phase_start_time = None               
                                
        # Phase 9 — Test complete
        elif self.phase == 9:
            self.stop()
            self.get_logger().info('Test sequence complete — stopping')
            self.get_logger().info(
                f'Final pose: x={self.current_x:.4f}  y={self.current_y:.4f}  '
                f'yaw={self.current_yaw:.2f} deg')
            self.test_done = True


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
