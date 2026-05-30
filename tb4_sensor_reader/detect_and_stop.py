import rclpy, cv2, math
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

NAMESPACE      = '/T29'
FORWARD_SPEED  = 0.15
TURN_SPEED     = 0.5
AVOID_DISTANCE = 0.55
FRONT_ARC_DEG  = 60

# Offset in degrees to rotate the "forward" direction.
# +90  = 90° CCW (left side of robot)
# -90  = 90° CW  (right side of robot)
# Change this until the arc points out the physical front of the robot.
FRONT_OFFSET_DEG = -90.0

RED_LOW1  = np.array([0,   120, 70])
RED_HIGH1 = np.array([10,  255, 255])
RED_LOW2  = np.array([170, 120, 70])
RED_HIGH2 = np.array([180, 255, 255])
MIN_PIXELS = 500000

class DetectAndStop(Node):
    def __init__(self):
        super().__init__('detect_and_stop')
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Twist, f'{NAMESPACE}/cmd_vel', 10)
        self.create_subscription(LaserScan, f'{NAMESPACE}/scan',
            self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(CompressedImage,
            f'{NAMESPACE}/oakd/rgb/image_raw/compressed',
            self.image_callback, 10)
        self.create_subscription(Odometry, f'{NAMESPACE}/odom',
            self.odom_callback, 10)

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
        inc    = msg.angle_increment
        arc_r  = math.radians(FRONT_ARC_DEG)
        side_r = math.radians(90)
        n      = len(msg.ranges)

        # Base forward index + offset to align with physical front of robot
        offset_i = int(round(math.radians(FRONT_OFFSET_DEG) / inc))
        front_i  = int(round(-msg.angle_min / inc)) + offset_i

        half_a = int(round(arc_r / 2 / inc))
        side_a = int(round(side_r / inc))

        def arc_min(lo, hi):
            # Use modulo to wrap around the scan array
            indices = [i % n for i in range(lo, hi + 1)]
            vals = [msg.ranges[i] for i in indices
                    if math.isfinite(msg.ranges[i])
                    and msg.range_min < msg.ranges[i] < msg.range_max]
            return min(vals) if vals else float('inf')

        self.nearest_front = arc_min(front_i - half_a, front_i + half_a)
        self.nearest_left  = arc_min(front_i,          front_i + side_a)
        self.nearest_right = arc_min(front_i - side_a, front_i)

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def image_callback(self, msg):
        if self.state != 'SEARCHING':
            return
        img  = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
            cv2.inRange(hsv, RED_LOW2, RED_HIGH2))
        if cv2.countNonZero(mask) >= MIN_PIXELS:
            self.cube_detected = True

    def stop(self):
        self.pub.publish(Twist())

    def _publish_twist(self, lin, ang):
        cmd = Twist()
        cmd.linear.x  = lin
        cmd.angular.z = ang
        self.pub.publish(cmd)

    def control_loop(self):
        if self.state == 'DONE':
            return

        if self.state == 'SEARCHING':
            if self.cube_detected:
                self.state = 'DETECTED'
                self.stop()
                self.get_logger().info('RED CUBE DETECTED — stopping')
                self.get_logger().info(
                    f'Detected position: x={self.current_x:.3f} m  y={self.current_y:.3f} m')
                return

            self.get_logger().info(f'front={self.nearest_front:.2f}')

            if self.nearest_front > AVOID_DISTANCE:
                self._publish_twist(FORWARD_SPEED, 0.0)
            else:
                turn = TURN_SPEED if self.nearest_left >= self.nearest_right else -TURN_SPEED
                self._publish_twist(0.0, turn)

        elif self.state == 'DETECTED':
            self.stop()
            self.state = 'DONE'

        if self.state == 'DONE':
            self.stop()
            self.get_logger().info('=== DONE ===')


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