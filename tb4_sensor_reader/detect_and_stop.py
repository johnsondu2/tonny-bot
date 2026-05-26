import rclpy, cv2, math
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

NAMESPACE      = '/T7'    # ← change to your robot namespace
FORWARD_SPEED  = 0.15
TURN_SPEED     = 0.5
AVOID_DISTANCE = 0.55
FRONT_ARC_DEG  = 60

RED_LOW1  = np.array([0,   120, 70])
RED_HIGH1 = np.array([10,  255, 255])
RED_LOW2  = np.array([170, 120, 70])
RED_HIGH2 = np.array([180, 255, 255])
MIN_PIXELS = 500           # ← use your calibrated value

class DetectAndStop(Node):
    def __init__(self):
        super().__init__('detect_and_stop')
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
        self.current_x     = 0.0
        self.current_y     = 0.0
        self.state = 'SEARCHING'
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Detect-and-stop node started — SEARCHING')

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

    def image_callback(self, msg):
        if self.state != 'SEARCHING':
            return   # no need to process once found
        img  = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
            cv2.inRange(hsv, RED_LOW2, RED_HIGH2))
        if cv2.countNonZero(mask) >= MIN_PIXELS:
            self.cube_detected = True

    def stop(self):
        self.pub.publish(Twist())

    def control_loop(self):
        if self.state == 'DONE':
            return
        if self.state == 'SEARCHING':
            if self.cube_detected:
                self.state = 'DETECTED'
                self.stop()
                # Log position at moment of detection
                self.get_logger().info(
                    f'RED CUBE DETECTED — stopping')
                self.get_logger().info(
                    f'Detected position: x={self.current_x:.3f} m  y={self.current_y:.3f} m')
                return
            cmd = Twist()
            if self.nearest_front > AVOID_DISTANCE:
                cmd.linear.x  = FORWARD_SPEED
                cmd.angular.z = 0.0
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = (TURN_SPEED if
                    self.nearest_left >= self.nearest_right
                    else -TURN_SPEED)
            self.pub.publish(cmd)
        elif self.state == 'DETECTED':
            self.stop()
            self.state = 'DONE'

def main(args=None):
    rclpy.init(args=args)
    node = DetectAndStop()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
