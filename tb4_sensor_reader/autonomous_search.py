import rclpy, cv2, math
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

NAMESPACE      = '/T18'    # change to your robot namespace
FORWARD_SPEED  = 0.15
TURN_SPEED     = 0.5
AVOID_DISTANCE = 0.55
FRONT_ARC_DEG  = 60

CENTRE_KP       = 1.0
CENTRE_MAX_TURN = 0.4

RETURN_TOLERANCE  = 0.30
HEADING_TOLERANCE = 0.12
RETURN_TURN_SPEED = 0.4
TIME_LIMIT_SEC = 300.0

RED_LOW1  = np.array([0,   120, 70])
RED_HIGH1 = np.array([10,  255, 255])
RED_LOW2  = np.array([170, 120, 70])
RED_HIGH2 = np.array([180, 255, 255])
MIN_PIXELS = 500
SNAPSHOT_PATH = 'detection_snapshot.jpg'

# ===== DEBUG SWITCHES =====
DEBUG = True
# Set True ONLY after the debug output proves left/right are swapped:
SWAP_LEFT_RIGHT = False
# Degrees to add to "forward" if index 0 / mount is offset (0 = no correction):
FRONT_OFFSET_DEG = -90.0
# Set True to print scan geometry once at startup:
PRINT_SCAN_INFO = True


class AutonomousSearch(Node):
    def __init__(self):
        super().__init__('autonomous_search')
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Twist, f'{NAMESPACE}/cmd_vel', 10)
        self.create_subscription(LaserScan, f'{NAMESPACE}/scan', self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(CompressedImage,
            f'{NAMESPACE}/oakd/rgb/image_raw/compressed', self.image_callback, 10)
        self.create_subscription(Odometry, f'{NAMESPACE}/odom', self.odom_callback, 10)

        self.nearest_front = float('inf')
        self.nearest_left  = float('inf')
        self.nearest_right = float('inf')
        self.cube_detected = False
        self.cube_side     = None
        self.current_x = 0.0; self.current_y = 0.0; self.current_yaw = 0.0
        self.latest_frame = None
        self.detected_x = None; self.detected_y = None
        self.start_time = self.get_clock().now()
        self._scan_info_printed = False

        self.state = 'SEARCHING'
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Autonomous search (DEBUG) started -- SEARCHING')

    def scan_callback(self, msg):
        inc     = msg.angle_increment
        arc_r   = math.radians(FRONT_ARC_DEG)
        side_r  = math.radians(90)
        offset  = math.radians(FRONT_OFFSET_DEG)
        # index of "forward" (angle 0), plus any manual mount offset
        front_i = int(round((-msg.angle_min + offset) / inc))
        half_a  = int(round(arc_r  / inc))
        side_a  = int(round(side_r / inc))
        n       = len(msg.ranges)

        if PRINT_SCAN_INFO and not self._scan_info_printed:
            self.get_logger().info(
                f'SCAN INFO: angle_min={msg.angle_min:.3f} '
                f'angle_max={msg.angle_max:.3f} inc={inc:.5f} '
                f'n={n} front_i={front_i} half_a={half_a} side_a={side_a} '
                f'range_min={msg.range_min:.2f} range_max={msg.range_max:.2f}')
            self.get_logger().info(
                f'  -> inc {"POSITIVE (CCW, left=+index)" if inc > 0 else "NEGATIVE (CW, right=+index)"}')
            self._scan_info_printed = True

        def arc_min(lo, hi):
            lo = max(0, lo); hi = min(n - 1, hi)
            vals = [r for r in msg.ranges[lo:hi+1]
                    if msg.range_min < r < msg.range_max]
            return min(vals) if vals else float('inf')

        self.nearest_front = arc_min(front_i - half_a, front_i + half_a)
        left_raw  = arc_min(front_i,          front_i + side_a)
        right_raw = arc_min(front_i - side_a, front_i)
        if SWAP_LEFT_RIGHT:
            self.nearest_left, self.nearest_right = right_raw, left_raw
        else:
            self.nearest_left, self.nearest_right = left_raw, right_raw

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny, cosy)

    def image_callback(self, msg):
        if self.state != 'SEARCHING':
            return
        img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        self.latest_frame = img
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
                              cv2.inRange(hsv, RED_LOW2, RED_HIGH2))
        if cv2.countNonZero(mask) >= MIN_PIXELS:
            self.cube_detected = True
            m = cv2.moments(mask)
            if m['m00'] > 0:
                cx = m['m10'] / m['m00']; w = mask.shape[1]
                if   cx < w / 3:     self.cube_side = 'left'
                elif cx < 2 * w / 3: self.cube_side = 'centre'
                else:                self.cube_side = 'right'

    def stop(self):
        self.pub.publish(Twist())

    def _publish_twist(self, lin, ang):
        cmd = Twist(); cmd.linear.x = lin; cmd.angular.z = ang
        self.pub.publish(cmd)

    @staticmethod
    def _angle_diff(target, current):
        d = target - current
        return math.atan2(math.sin(d), math.cos(d))

    def _elapsed(self):
        return (self.get_clock().now() - self.start_time).nanoseconds / 1e9

    def _save_snapshot(self):
        if self.latest_frame is not None:
            cv2.imwrite(SNAPSHOT_PATH, self.latest_frame)
            self.get_logger().info(f'Snapshot saved to {SNAPSHOT_PATH}')

    def _search_cmd(self):
        if self.nearest_front < AVOID_DISTANCE:
            turn = TURN_SPEED if self.nearest_left >= self.nearest_right else -TURN_SPEED
            if DEBUG:
                self.get_logger().info(
                    f'[AVOID] front={self.nearest_front:.2f} '
                    f'L={self.nearest_left:.2f} R={self.nearest_right:.2f} '
                    f'-> turn {"LEFT(+)" if turn > 0 else "RIGHT(-)"} {turn:+.2f}')
            return 0.0, turn
        if math.isinf(self.nearest_left) or math.isinf(self.nearest_right):
            if DEBUG:
                self.get_logger().info(
                    f'[STRAIGHT] a wall missing  L={self.nearest_left:.2f} R={self.nearest_right:.2f}')
            return FORWARD_SPEED, 0.0
        imbalance = self.nearest_left - self.nearest_right
        ang = max(-CENTRE_MAX_TURN, min(CENTRE_MAX_TURN, CENTRE_KP * imbalance))
        if DEBUG:
            self.get_logger().info(
                f'[CENTRE] front={self.nearest_front:.2f} '
                f'L={self.nearest_left:.2f} R={self.nearest_right:.2f} '
                f'imb(L-R)={imbalance:+.2f} -> ang {"LEFT(+)" if ang >= 0 else "RIGHT(-)"} {ang:+.2f}')
        return FORWARD_SPEED, ang

    def control_loop(self):
        if self.state == 'DONE':
            return
        if self.state == 'SEARCHING':
            if self.cube_detected:
                self.state = 'REPORTING'; self.stop(); return
            if self._elapsed() > TIME_LIMIT_SEC:
                self.get_logger().warn('Time limit -- returning')
                self.state = 'RETURNING'; self.stop(); return
            lin, ang = self._search_cmd()
            self._publish_twist(lin, ang)
        elif self.state == 'REPORTING':
            self.stop()
            self.detected_x = self.current_x; self.detected_y = self.current_y
            side = self.cube_side or 'unknown'
            self.get_logger().info(f'RED CUBE DETECTED on {side} pillar')
            self.get_logger().info(
                f'Detected position: x={self.detected_x:.3f} y={self.detected_y:.3f}')
            self._save_snapshot()
            self.state = 'RETURNING'
        elif self.state == 'RETURNING':
            dx = -self.current_x; dy = -self.current_y
            distance = math.sqrt(dx*dx + dy*dy)
            if distance < RETURN_TOLERANCE:
                self.state = 'DONE'; self.stop(); return
            if self.nearest_front < AVOID_DISTANCE:
                turn = RETURN_TURN_SPEED if self.nearest_left >= self.nearest_right else -RETURN_TURN_SPEED
                self._publish_twist(0.0, turn); return
            target_angle = math.atan2(dy, dx)
            heading_err  = self._angle_diff(target_angle, self.current_yaw)
            if abs(heading_err) > HEADING_TOLERANCE:
                self._publish_twist(0.0, RETURN_TURN_SPEED if heading_err > 0 else -RETURN_TURN_SPEED)
            else:
                self._publish_twist(FORWARD_SPEED, 0.0)
        if self.state == 'DONE':
            self.stop()
            self.get_logger().info('=== DONE ===')


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousSearch()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows(); node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()