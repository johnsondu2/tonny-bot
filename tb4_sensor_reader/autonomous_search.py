import rclpy, cv2, math, csv, os, time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data

NAMESPACE        = '/T21'
FORWARD_SPEED    = 0.15
TURN_SPEED       = 0.5
AVOID_DISTANCE   = 0.45
FRONT_ARC_DEG    = 60
FRONT_OFFSET_DEG = -90.0

WAYPOINT_RADIUS  = 0.1
HOME_RADIUS      = 0.15   # how close to (0,0) counts as home
HEADING_TOL      = 0.08
AVOID_FWD_SECS   = 1.0

WAYPOINTS_CSV    = os.path.expanduser('~/Downloads/map/path_waypoints.csv')

# Cube finding
CUBE_PIXEL_THRESHOLD = 2000
CUBE_STOP_PIXELS     = 25000
CUBE_TURN_SPEED      = 0.1   # slow sweep for accuracy
CUBE_FWD_SPEED       = 0.08
SWEEP_DEG            = 120.0  # sweep 120° CW centred on -X

RED_LOW1  = np.array([0,   120, 70])
RED_HIGH1 = np.array([10,  255, 255])
RED_LOW2  = np.array([170, 120, 70])
RED_HIGH2 = np.array([180, 255, 255])
MIN_PIXELS = 500000


def load_waypoints(path):
    waypoints = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            waypoints.append((float(row['x_m']), float(row['y_m'])))
    return waypoints


class WaypointNav(Node):
    def __init__(self):
        super().__init__('waypoint_nav')
        self.pub = self.create_publisher(Twist, f'{NAMESPACE}/cmd_vel', 10)
        self.create_subscription(LaserScan, f'{NAMESPACE}/scan',
            self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(CompressedImage,
            f'{NAMESPACE}/oakd/rgb/image_raw/compressed',
            self.image_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, f'{NAMESPACE}/odom',
            self.odom_callback, 10)

        # LiDAR
        self.nearest_front = float('inf')
        self.nearest_left  = float('inf')
        self.nearest_right = float('inf')

        # Odometry
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_yaw = 0.0

        # Camera
        self.cube_detected     = False
        self.latest_red_pixels = 0
        self.latest_frame      = None

        # Waypoints
        self.waypoints = load_waypoints(WAYPOINTS_CSV)
        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints')
        self.wp_index     = 0
        self.search_phase = 'TURNING'

        # Avoidance — tracks which state to return to after avoidance
        self.avoid_turn_dir    = 1
        self.avoid_fwd_start   = None
        self.avoid_return_state = 'SEARCHING'  # SEARCHING or RETURNING

        # Odometry trail — split by phase
        self.odom_trail      = []   # full trail (kept for compatibility)
        self.outbound_trail  = []   # SEARCHING phase
        self.return_trail    = []   # RETURNING phase
        self.is_returning    = False

        # Cube finding sub-phases: ALIGN_NEG_X → SWEEP → ALIGN_CUBE → APPROACH
        self.cf_phase             = 'ALIGN_NEG_X'
        self.sweep_start_yaw      = None
        self.sweep_readings       = []
        self.sweep_last_yaw       = None
        self.swept_total          = 0.0
        self.cube_relative_turn   = None
        self.cube_turn_start_yaw  = None
        self.heading_home         = False  # True once first waypoint is reached on return
        self.clearing_dead_end    = False  # True while driving back to dead end after cube
        self.start_time           = time.time()
        self.cube_odom            = None   # (x, y) recorded when cube snapshot taken
        self.home_odom            = None   # (x, y) recorded when home reached

        self.state = 'SEARCHING'
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Waypoint nav started')

    # ── Subscribers ───────────────────────────────────────────────────────────

    def scan_callback(self, msg):
        inc      = msg.angle_increment
        n        = len(msg.ranges)
        offset_i = int(round(math.radians(FRONT_OFFSET_DEG) / inc))
        front_i  = int(round(-msg.angle_min / inc)) + offset_i
        half_a   = int(round(math.radians(FRONT_ARC_DEG / 2) / inc))
        side_a   = int(round(math.radians(90) / inc))

        def arc_min(lo, hi):
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
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny, cosy)
        self.odom_trail.append((self.current_x, self.current_y))
        if self.is_returning:
            self.return_trail.append((self.current_x, self.current_y))
        else:
            self.outbound_trail.append((self.current_x, self.current_y))

    def image_callback(self, msg):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
            cv2.inRange(hsv, RED_LOW2, RED_HIGH2))
        self.latest_red_pixels = cv2.countNonZero(mask)
        self.latest_frame = img
        if self.state not in ('DETECTED', 'DONE', 'CUBE_FINDING', 'RETURNING'):
            if self.latest_red_pixels >= MIN_PIXELS:
                self.cube_detected = True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def stop(self):
        self.pub.publish(Twist())

    def _publish_twist(self, lin, ang):
        cmd = Twist()
        cmd.linear.x  = lin
        cmd.angular.z = ang
        self.pub.publish(cmd)

    def _distance_to_wp(self):
        wx, wy = self.waypoints[self.wp_index]
        return math.sqrt((self.current_x - wx)**2 + (self.current_y - wy)**2)

    def _heading_error_to_wp(self):
        wx, wy = self.waypoints[self.wp_index]
        dx = wx - self.current_x
        dy = wy - self.current_y
        target = math.atan2(dy, dx)
        err    = target - self.current_yaw
        return math.atan2(math.sin(err), math.cos(err))

    def _heading_error_to_xy(self, tx, ty):
        dx = tx - self.current_x
        dy = ty - self.current_y
        target = math.atan2(dy, dx)
        err    = target - self.current_yaw
        return math.atan2(math.sin(err), math.cos(err))

    def _distance_to_home(self):
        return math.sqrt(self.current_x**2 + self.current_y**2)

    # ── Control loop ──────────────────────────────────────────────────────────

    def control_loop(self):
        if self.state == 'DONE':
            return

        # ── SEARCHING ─────────────────────────────────────────────────────
        if self.state == 'SEARCHING':

            if self.cube_detected:
                self.state = 'DETECTED'
                self.stop()
                self.get_logger().info('RED CUBE DETECTED')
                self.get_logger().info(
                    f'Position: x={self.current_x:.3f} y={self.current_y:.3f}')
                return

            if self.nearest_front < AVOID_DISTANCE:
                self.avoid_turn_dir = 1 if self.nearest_left >= self.nearest_right else -1
                self.avoid_return_state = 'SEARCHING'
                self.get_logger().info(
                    f'Obstacle at {self.nearest_front:.2f}m — '
                    f'turning {"LEFT" if self.avoid_turn_dir > 0 else "RIGHT"}')
                self.state = 'AVOIDING'
                return

            if self.search_phase == 'TURNING':
                err = self._heading_error_to_wp()
                self.get_logger().info(
                    f'[SEARCHING/TURNING] wp={self.wp_index+1}/{len(self.waypoints)} '
                    f'err={math.degrees(err):.1f}°')
                if abs(err) < HEADING_TOL:
                    self.stop()
                    self.search_phase = 'DRIVING'
                    self.get_logger().info(f'Aligned — driving to waypoint {self.wp_index+1}')
                else:
                    self._publish_twist(0.0, TURN_SPEED if err > 0 else -TURN_SPEED)

            elif self.search_phase == 'DRIVING':
                dist = self._distance_to_wp()
                self.get_logger().info(
                    f'[SEARCHING/DRIVING] wp={self.wp_index+1}/{len(self.waypoints)} dist={dist:.3f}m')
                if dist < WAYPOINT_RADIUS:
                    self.stop()
                    self.wp_index += 1
                    if self.wp_index >= len(self.waypoints):
                        self.state = 'CUBE_FINDING'
                        self.cf_phase = 'ALIGN_NEG_X'
                        self.get_logger().info('All waypoints reached — CUBE_FINDING')
                    else:
                        self.search_phase = 'TURNING'
                        self.get_logger().info(
                            f'Waypoint reached — turning to waypoint {self.wp_index+1}')
                else:
                    self._publish_twist(FORWARD_SPEED, 0.0)

        # ── AVOIDING ──────────────────────────────────────────────────────
        elif self.state == 'AVOIDING':
            if self.nearest_front >= AVOID_DISTANCE:
                self.avoid_fwd_start = time.time()
                self.state = 'AVOID_FWD'
                self.get_logger().info('Arc clear — forward burst')
            else:
                self._publish_twist(0.0, TURN_SPEED * self.avoid_turn_dir)

        # ── AVOID_FWD ─────────────────────────────────────────────────────
        elif self.state == 'AVOID_FWD':
            if time.time() - self.avoid_fwd_start >= AVOID_FWD_SECS:
                self.stop()
                if self.avoid_return_state == 'RETURNING':
                    # Decrement: skip back one waypoint on return trip
                    if self.wp_index > 0:
                        self.wp_index -= 1
                    self.search_phase = 'TURNING'
                    self.state = 'RETURNING'
                    self.get_logger().info(
                        f'Burst done — re-aligning to return waypoint {self.wp_index+1}')
                else:
                    # Increment: skip forward one waypoint on outbound trip
                    if self.wp_index < len(self.waypoints) - 1:
                        self.wp_index += 1
                    self.search_phase = 'TURNING'
                    self.state = 'SEARCHING'
                    self.get_logger().info(
                        f'Burst done — re-aligning to waypoint {self.wp_index+1}')
            else:
                self._publish_twist(FORWARD_SPEED, 0.0)

        # ── CUBE_FINDING ──────────────────────────────────────────────────
        elif self.state == 'CUBE_FINDING':

            # Phase 1: align to 240° (-2π/3) — 60° CCW from -X, sweep 120° CW through -X
            if self.cf_phase == 'ALIGN_NEG_X':
                target_yaw = -2.0 * math.pi / 3.0
                err = target_yaw - self.current_yaw
                err = math.atan2(math.sin(err), math.cos(err))
                if abs(err) < HEADING_TOL:
                    self.stop()
                    self.sweep_start_yaw = self.current_yaw
                    self.sweep_last_yaw  = self.current_yaw
                    self.swept_total     = 0.0
                    self.sweep_readings  = []
                    self.cf_phase = 'SWEEP'
                    self.get_logger().info('Aligned — starting 120° CW sweep around -X')
                else:
                    self._publish_twist(0.0, CUBE_TURN_SPEED if err > 0 else -CUBE_TURN_SPEED)

            # Phase 2: sweep 120° CW at 0.1 rad/s
            elif self.cf_phase == 'SWEEP':
                delta = self.sweep_last_yaw - self.current_yaw
                if delta > math.pi:
                    delta -= 2 * math.pi
                elif delta < -math.pi:
                    delta += 2 * math.pi
                self.swept_total    += abs(delta)
                self.sweep_last_yaw  = self.current_yaw
                self.sweep_readings.append((self.current_yaw, self.latest_red_pixels))
                self.get_logger().info(
                    f'[SWEEP] swept={math.degrees(self.swept_total):.1f}° '
                    f'red_px={self.latest_red_pixels}')
                if self.swept_total >= math.radians(SWEEP_DEG):
                    self.stop()
                    above = [(yaw, px) for yaw, px in self.sweep_readings
                             if px >= CUBE_PIXEL_THRESHOLD]
                    if above:
                        mid_idx = len(above) // 2
                        cube_yaw = above[mid_idx][0]
                        rel = cube_yaw - self.current_yaw
                        self.cube_relative_turn  = math.atan2(math.sin(rel), math.cos(rel))
                        self.cube_turn_start_yaw = self.current_yaw
                        self.get_logger().info(
                            f'Sweep done — turning {math.degrees(self.cube_relative_turn):.1f}° to face cube')
                        self.cf_phase = 'ALIGN_CUBE'
                    else:
                        self.get_logger().warn('No cube detected in sweep — DONE')
                        self.state = 'DONE'
                else:
                    self._publish_twist(0.0, -CUBE_TURN_SPEED)  # CW = negative

            # Phase 3: turn relative angle to face cube
            elif self.cf_phase == 'ALIGN_CUBE':
                turned = self.current_yaw - self.cube_turn_start_yaw
                turned = math.atan2(math.sin(turned), math.cos(turned))
                remaining = self.cube_relative_turn - turned
                remaining = math.atan2(math.sin(remaining), math.cos(remaining))
                self.get_logger().info(
                    f'[ALIGN_CUBE] remaining={math.degrees(remaining):.1f}°')
                if abs(remaining) < HEADING_TOL:
                    self.stop()
                    self.cf_phase = 'APPROACH'
                    self.get_logger().info('Facing cube — approaching')
                else:
                    self._publish_twist(0.0, CUBE_TURN_SPEED if remaining > 0 else -CUBE_TURN_SPEED)

            # Phase 4: drive forward until red pixels >= CUBE_STOP_PIXELS
            elif self.cf_phase == 'APPROACH':
                self.get_logger().info(f'[APPROACH] red_px={self.latest_red_pixels}')
                if self.latest_red_pixels >= CUBE_STOP_PIXELS:
                    self.stop()
                    self.cube_odom = (self.current_x, self.current_y)
                    self.get_logger().info(
                        f'Cube detected at odometry: x={self.current_x:.4f} y={self.current_y:.4f}')
                    if self.latest_frame is not None:
                        snapshot_path = os.path.expanduser('~/Downloads/map/detection_snapshot.jpg')
                        cv2.imwrite(snapshot_path, self.latest_frame)
                        self.get_logger().info(f'Snapshot saved to {snapshot_path}')
                    # Begin return trip from final waypoint
                    self.wp_index = len(self.waypoints) - 1  # start from dead end waypoint
                    self.search_phase = 'TURNING'
                    self.is_returning = True
                    self.clearing_dead_end = True
                    self.state = 'RETURNING'
                    self.get_logger().info('Cube found — returning to dead end first')
                else:
                    self._publish_twist(CUBE_FWD_SPEED, 0.0)

        # ── RETURNING ─────────────────────────────────────────────────────
        elif self.state == 'RETURNING':

            if self.nearest_front < AVOID_DISTANCE and not self.clearing_dead_end:
                self.avoid_turn_dir = 1 if self.nearest_left >= self.nearest_right else -1
                self.avoid_return_state = 'RETURNING'
                self.get_logger().info(
                    f'[RETURNING] Obstacle at {self.nearest_front:.2f}m — '
                    f'turning {"LEFT" if self.avoid_turn_dir > 0 else "RIGHT"}')
                self.state = 'AVOIDING'
                return

            if self.search_phase == 'TURNING':
                if self.heading_home:
                    err = self._heading_error_to_xy(0.0, 0.0)
                    label = 'home (0,0)'
                else:
                    err = self._heading_error_to_wp()
                    label = f'wp={self.wp_index+1}/{len(self.waypoints)}'
                self.get_logger().info(f'[RETURNING/TURNING] {label} err={math.degrees(err):.1f}°')
                if abs(err) < HEADING_TOL:
                    self.stop()
                    self.search_phase = 'DRIVING'
                    self.get_logger().info(f'Aligned — driving to {label}')
                else:
                    self._publish_twist(0.0, TURN_SPEED if err > 0 else -TURN_SPEED)

            elif self.search_phase == 'DRIVING':
                dist = self._distance_to_home() if self.heading_home else self._distance_to_wp()
                label = 'home' if self.heading_home else f'wp={self.wp_index+1}/{len(self.waypoints)}'
                self.get_logger().info(f'[RETURNING/DRIVING] {label} dist={dist:.3f}m')
                if dist < WAYPOINT_RADIUS:
                    self.stop()
                    if self.heading_home:
                        # Arrived at (0,0)
                        self.home_odom = (self.current_x, self.current_y)
                        self.state = 'DONE'
                        self.get_logger().info('Home reached — DONE')
                    elif self.wp_index == 0:
                        # Reached first waypoint — now target (0,0)
                        self.heading_home = True
                        self.search_phase = 'TURNING'
                        self.get_logger().info('First waypoint reached — heading to (0,0)')
                    else:
                        if self.clearing_dead_end:
                            self.clearing_dead_end = False
                            self.get_logger().info('Dead end reached — beginning return, obstacle avoidance active')
                        self.wp_index -= 1
                        self.search_phase = 'TURNING'
                        self.get_logger().info(
                            f'Waypoint reached — turning to return waypoint {self.wp_index+1}')
                else:
                    self._publish_twist(FORWARD_SPEED, 0.0)

        # ── DETECTED ──────────────────────────────────────────────────────
        elif self.state == 'DETECTED':
            self.stop()
            self.state = 'DONE'

        if self.state == 'DONE':
            self.stop()
            elapsed = time.time() - self.start_time
            mins    = int(elapsed // 60)
            secs    = int(elapsed % 60)
            self.get_logger().info('=== DONE ===')
            self.get_logger().info(f'Total run time: {mins}m {secs}s')
            if self.cube_odom:
                self.get_logger().info(
                    f'Cube snapshot position: x={self.cube_odom[0]:.4f} y={self.cube_odom[1]:.4f}')
            if self.home_odom:
                self.get_logger().info(
                    f'Home return position:   x={self.home_odom[0]:.4f} y={self.home_odom[1]:.4f}')
            self.timer.cancel()
            raise SystemExit


def save_odom_map(trail, map_dir, outbound_trail=None, return_trail=None):
    pgm_path  = os.path.join(map_dir, 'map.pgm')
    yaml_path = os.path.join(map_dir, 'map.yaml')
    out_path  = os.path.join(map_dir, 'odom_overlay.png')

    if not os.path.exists(pgm_path) or not os.path.exists(yaml_path):
        print(f'Map files not found in {map_dir} — skipping odom overlay')
        return

    import yaml
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    resolution = meta['resolution']
    origin     = meta['origin']

    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    scale = 8
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = cv2.resize(vis, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    def world_to_px(wx, wy):
        col = int(round((wx - origin[0]) / resolution))
        row = h - int(round((wy - origin[1]) / resolution))
        return col * scale, row * scale

    OUTBOUND_COLOUR = (0, 200, 255)   # orange
    RETURN_COLOUR   = (120, 255, 120) # green

    # Draw outbound trail
    if outbound_trail and len(outbound_trail) > 1:
        pts = [world_to_px(x, y) for x, y in outbound_trail]
        for i in range(len(pts) - 1):
            cv2.line(vis, pts[i], pts[i+1], OUTBOUND_COLOUR, 2)

    # Draw return trail
    if return_trail and len(return_trail) > 1:
        pts = [world_to_px(x, y) for x, y in return_trail]
        for i in range(len(pts) - 1):
            cv2.line(vis, pts[i], pts[i+1], RETURN_COLOUR, 2)

    # Fall back to full trail if no split trails provided
    if not outbound_trail and not return_trail and len(trail) > 1:
        pts = [world_to_px(x, y) for x, y in trail]
        for i in range(len(pts) - 1):
            cv2.line(vis, pts[i], pts[i+1], (0, 200, 255), 2)

    # Mark start and end
    if trail:
        sx, sy = world_to_px(*trail[0])
        ex, ey = world_to_px(*trail[-1])
        cv2.circle(vis, (sx, sy), 7, (60, 60, 255), -1)
        cv2.putText(vis, 'start', (sx+6, sy-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1)
        cv2.circle(vis, (ex, ey), 7, (255, 255, 255), -1)
        cv2.putText(vis, 'end', (ex+6, ey-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Legend
    lx, ly = 8, 8
    cv2.rectangle(vis, (lx, ly), (lx+160, ly+50), (40, 40, 40), -1)
    cv2.line(vis, (lx+6, ly+15), (lx+26, ly+15), OUTBOUND_COLOUR, 2)
    cv2.putText(vis, 'Outbound', (lx+32, ly+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, OUTBOUND_COLOUR, 1)
    cv2.line(vis, (lx+6, ly+35), (lx+26, ly+35), RETURN_COLOUR, 2)
    cv2.putText(vis, 'Returning', (lx+32, ly+40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, RETURN_COLOUR, 1)

    cv2.imwrite(out_path, vis)
    print(f'Odom overlay saved to {out_path}  ({len(trail)} points)')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointNav()
    try:
        rclpy.spin(node)
    finally:
        save_odom_map(node.odom_trail, os.path.expanduser('~/Downloads/map'), node.outbound_trail, node.return_trail)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()