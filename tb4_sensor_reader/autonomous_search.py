import rclpy, cv2, math
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

NAMESPACE      = '/T7'    # change to your robot namespace
FORWARD_SPEED  = 0.15
TURN_SPEED     = 0.5
AVOID_DISTANCE = 0.55     # obstacle-cylinder avoidance threshold (m)
FRONT_ARC_DEG  = 60

# --- Centre-line steering (drive down the middle of the C) ---
CENTRE_KP       = 1.0     # gain on (left - right) wall-distance imbalance
CENTRE_MAX_TURN = 0.4     # clamp on centring turn rate (rad/s)

# --- Return-to-start tuning ---
RETURN_TOLERANCE  = 0.30  # within this many m of origin -> DONE
HEADING_TOLERANCE = 0.12  # rad; rotate until heading error below this
RETURN_TURN_SPEED = 0.4

# --- Run time limit ---
TIME_LIMIT_SEC = 300.0    # search budget before forced return (tune to demo)

# --- Red HSV thresholds (use your Investigation C values) ---
RED_LOW1  = np.array([0,   120, 70])
RED_HIGH1 = np.array([10,  255, 255])
RED_LOW2  = np.array([170, 120, 70])
RED_HIGH2 = np.array([180, 255, 255])
MIN_PIXELS = 500          # use your calibrated value

SNAPSHOT_PATH = 'detection_snapshot.jpg'


class AutonomousSearch(Node):
    def __init__(self):
        super().__init__('autonomous_search')
        self.bridge = CvBridge()
        self.pub = self.create_publisher(
            Twist, f'{NAMESPACE}/cmd_vel', 10)
        self.create_subscription(
            LaserScan, f'{NAMESPACE}/scan',
            self.scan_callback, 10)
        self.create_subscription(
            CompressedImage,
            f'{NAMESPACE}/oakd/rgb/image_raw/compressed',
            self.image_callback, 10)
        self.create_subscription(
            Odometry, f'{NAMESPACE}/odom',
            self.odom_callback, 10)

        # Shared state
        self.nearest_front = float('inf')
        self.nearest_left  = float('inf')
        self.nearest_right = float('inf')
        self.cube_detected = False
        self.cube_side     = None          # 'left' / 'centre' / 'right'
        self.current_x     = 0.0
        self.current_y     = 0.0
        self.current_yaw   = 0.0
        self.latest_frame  = None

        self.detected_x = None
        self.detected_y = None
        self.start_time = self.get_clock().now()

        self.state = 'SEARCHING'
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Autonomous search started -- SEARCHING (centre-line)')

    # ---------------- Sensor callbacks ----------------

    def scan_callback(self, msg):
        inc     = msg.angle_increment
        arc_r   = math.radians(FRONT_ARC_DEG)
        side_r  = math.radians(90)
        front_i = int(round(-msg.angle_min / inc))
        half_a  = int(round(arc_r  / inc))
        side_a  = int(round(side_r / inc))
        n       = len(msg.ranges)

        def arc_min(lo, hi):
            lo = max(0, lo); hi = min(n - 1, hi)
            vals = [r for r in msg.ranges[lo:hi+1]
                    if msg.range_min < r < msg.range_max]
            return min(vals) if vals else float('inf')

        self.nearest_front = arc_min(front_i - half_a, front_i + half_a)
        self.nearest_left  = arc_min(front_i,          front_i + side_a)
        self.nearest_right = arc_min(front_i - side_a, front_i)

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
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
            cv2.inRange(hsv, RED_LOW2, RED_HIGH2))
        if cv2.countNonZero(mask) >= MIN_PIXELS:
            self.cube_detected = True
            # Classify which pillar by the red blob's horizontal centroid
            m = cv2.moments(mask)
            if m['m00'] > 0:
                cx = m['m10'] / m['m00']
                w  = mask.shape[1]
                if   cx < w / 3:       self.cube_side = 'left'
                elif cx < 2 * w / 3:   self.cube_side = 'centre'
                else:                  self.cube_side = 'right'

    # ---------------- Helpers ----------------

    def stop(self):
        self.pub.publish(Twist())

    def _publish_twist(self, lin, ang):
        cmd = Twist()
        cmd.linear.x  = lin
        cmd.angular.z = ang
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
        else:
            self.get_logger().warn('No frame available to save')

    # ---------------- Search: centre-line following ----------------

    def _search_cmd(self):
        """Drive down the middle of the corridor using LiDAR.
        Steer to keep left and right wall distances equal."""
        # Obstacle dead ahead (wall end or obstacle cylinder): turn away
        if self.nearest_front < AVOID_DISTANCE:
            turn = TURN_SPEED if self.nearest_left >= self.nearest_right \
                   else -TURN_SPEED
            return 0.0, turn

        # Centre between walls: positive imbalance = more room on left -> steer left
        if math.isinf(self.nearest_left) or math.isinf(self.nearest_right):
            return FORWARD_SPEED, 0.0   # one wall missing -- just go straight
        imbalance = self.nearest_left - self.nearest_right
        ang = CENTRE_KP * imbalance
        ang = max(-CENTRE_MAX_TURN, min(CENTRE_MAX_TURN, ang))
        return FORWARD_SPEED, ang

    # ---------------- State machine ----------------

    def control_loop(self):
        if self.state == 'DONE':
            return

        if self.state == 'SEARCHING':
            if self.cube_detected:
                self.state = 'REPORTING'
                self.stop()
                return
            if self._elapsed() > TIME_LIMIT_SEC:
                self.get_logger().warn('Time limit reached -- returning')
                self.state = 'RETURNING'
                self.stop()
                return
            lin, ang = self._search_cmd()
            self._publish_twist(lin, ang)

        elif self.state == 'REPORTING':
            self.stop()
            self.detected_x = self.current_x
            self.detected_y = self.current_y
            side = self.cube_side or 'unknown'
            self.get_logger().info(f'RED CUBE DETECTED on {side} pillar -- stopping')
            self.get_logger().info(
                f'Detected position: x={self.detected_x:.3f} m  '
                f'y={self.detected_y:.3f} m')
            self._save_snapshot()
            self.state = 'RETURNING'

        elif self.state == 'RETURNING':
            dx = 0.0 - self.current_x
            dy = 0.0 - self.current_y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < RETURN_TOLERANCE:
                self.state = 'DONE'
                self.stop()
                return
            # Live obstacle override on the way home
            if self.nearest_front < AVOID_DISTANCE:
                turn = RETURN_TURN_SPEED if self.nearest_left >= self.nearest_right \
                       else -RETURN_TURN_SPEED
                self._publish_twist(0.0, turn)
                return
            target_angle = math.atan2(dy, dx)
            heading_err  = self._angle_diff(target_angle, self.current_yaw)
            if abs(heading_err) > HEADING_TOLERANCE:
                self._publish_twist(
                    0.0, RETURN_TURN_SPEED if heading_err > 0
                    else -RETURN_TURN_SPEED)
            else:
                self._publish_twist(FORWARD_SPEED, 0.0)

        if self.state == 'DONE':
            self.stop()
            det = (f'x={self.detected_x:.3f} y={self.detected_y:.3f} '
                   f'({self.cube_side} pillar)'
                   if self.detected_x is not None else 'none (time limit)')
            self.get_logger().info('=== RUN SUMMARY ===')
            self.get_logger().info(f'Detected position: {det}')
            self.get_logger().info(
                f'Return position:   x={self.current_x:.3f} y={self.current_y:.3f}')
            self.get_logger().info(f'Total run time:    {self._elapsed():.1f} s')


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousSearch()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()