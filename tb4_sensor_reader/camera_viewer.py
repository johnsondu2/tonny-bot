import rclpy, cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import qos_profile_sensor_data

NAMESPACE = '/T21'

class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        topic = f'{NAMESPACE}/oakd/rgb/image_raw/compressed'
        self.create_subscription(
            CompressedImage, topic, self.image_callback, qos_profile_sensor_data)
        self.get_logger().info(f'Camera viewer started — {topic}')

    def image_callback(self, msg):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return
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