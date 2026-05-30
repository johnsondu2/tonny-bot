import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

NAMESPACE = '/T29'   # ← change to your robot namespace

class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        self.bridge = CvBridge()
        topic = f'{NAMESPACE}/oakd/rgb/image_raw/compressed'
        self.create_subscription(
            CompressedImage, topic, self.image_callback, 10)
        self.get_logger().info(f'Camera viewer started — topic: {topic}')

    def image_callback(self, msg):
        img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        cv2.imshow('Camera', img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()