import rclpy, cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

NAMESPACE = '/T29'          # ← change to your robot namespace

# HSV thresholds for red — use your Investigation C values
# Red wraps around hue 0/180, so two ranges are needed
RED_LOW1  = np.array([0,   120, 70])
RED_HIGH1 = np.array([10,  255, 255])
RED_LOW2  = np.array([170, 120, 70])
RED_HIGH2 = np.array([180, 255, 255])

# Minimum pixel count to count as a detection
# Use the value you established in Investigation C
MIN_PIXELS = 500

class CameraDetector(Node):
    def __init__(self):
        super().__init__('camera_detector')
        self.bridge = CvBridge()
        topic = f'{NAMESPACE}/oakd/rgb/image_raw/compressed'
        self.create_subscription(
            CompressedImage, topic, self.image_callback, 10)
        self.get_logger().info('Camera detector started')

    def image_callback(self, msg):
        img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
            cv2.inRange(hsv, RED_LOW2, RED_HIGH2)
        )
        pixels = cv2.countNonZero(mask)
        overlay = img.copy()
        overlay[mask > 0] = [0, 0, 255]
        cv2.putText(overlay, f'Red pixels: {pixels}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if pixels >= MIN_PIXELS:
            cv2.putText(overlay, 'DETECTED', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            self.get_logger().info(f'Red cube detected — {pixels} pixels')
        cv2.imshow('Detection', overlay)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CameraDetector()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()