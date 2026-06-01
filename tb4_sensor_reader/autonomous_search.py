# ─────────────────────────────────────────────────────────────────────────────
# autonomous_search.py
# ─────────────────────────────────────────────────────────────────────────────

import rclpy, cv2, math, os, threading, time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data

from tb4_sensor_reader.path_planner_v2 import (
    load_map, largest_free_region,
    world_to_pixel, pixel_to_world,
    find_dead_end, astar, thin,
    WAYPOINT_STEP, CENTRE_WEIGHT,
)

NAMESPACE        = '/T10'
FORWARD_SPEED    = 0.15
TURN_SPEED       = 0.5
FRONT_ARC_DEG    = 60
FRONT_OFFSET_DEG = -90.0
REPLAN_COOLDOWN_S = 3.0
MAP_DIR          = os.path.expanduser('~/Downloads/map')
WAYPOINT_RADIUS  = 0.1
HOME_RADIUS      = 0.10
HEADING_TOL      = 0.08
EMERGENCY_DIST   = 0.25
ROBOT_RADIUS_PX  = 3
REPLAN_NEARBY_M  = 1.5
REPLAN_LOOKAHEAD = 5
SCAN_SAMPLE_RATE = 4
SCAN_MAX_RANGE_M = 4.0
MIN_NEW_OBS_DIST_PX = 3
HIT_THRESHOLD       = 3
MIN_PIXELS           = 500000
CUBE_PIXEL_THRESHOLD = 2000
CUBE_STOP_PIXELS     = 25000
CUBE_TURN_SPEED      = 0.05
CUBE_FWD_SPEED       = 0.08
SWEEP_DEG            = 120.0
RAMP_ACCEL           = 0.025
RAMP_DECEL           = 0.05
WAYPOINT_TURN_THRESHOLD = math.radians(45)
WAYPOINT_TRACKING_GAIN  = 4.0

# Stuck detector: if the robot hasn't closed the distance to the current
# waypoint by STALL_IMPROVE_M within STALL_TICKS control ticks (~2 s at 10 Hz),
# it is assumed to be circling an obstacle and a fresh replan is forced.
STALL_TICKS     = 20
STALL_IMPROVE_M = 0.02
RED_LOW1  = np.array([0,   120, 70])
RED_HIGH1 = np.array([10,  255, 255])
RED_LOW2  = np.array([170, 120, 70])
RED_HIGH2 = np.array([180, 255, 255])


class WaypointNav(Node):

    def __init__(self):
        super().__init__('waypoint_nav')
        self._last_replan_time = 0.0
        self.start_time        = time.time()

        self.pub = self.create_publisher(Twist, f'{NAMESPACE}/cmd_vel', 10)
        self.create_subscription(LaserScan, f'{NAMESPACE}/scan',
            self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(CompressedImage,
            f'{NAMESPACE}/oakd/rgb/image_raw/compressed',
            self.image_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, f'{NAMESPACE}/odom',
            self.odom_callback, 10)

        self.nearest_front = float('inf')
        self.nearest_left  = float('inf')
        self.nearest_right = float('inf')

        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_z   = 0.0
        self.current_yaw = 0.0
        self.odom_trail      = []
        self.outbound_trail  = []
        self.return_trail    = []
        self.is_returning    = False

        self.cube_detected     = False
        self.latest_red_pixels = 0
        self.latest_img        = None

        self.cube_odom      = None   # (x, y) robot position at detection
        self.cube_world_pos = None   # (x, y) estimated actual cube position
        self.home_odom      = None   # (x, y, z) robot position when home reached

        self.map_resolution   = None
        self.map_origin       = None
        self.map_h            = None
        self.map_w            = None
        self.live_grid        = None
        self.phase1_navigable = None
        self.phase1_dist      = None
        self.hit_counts       = None
        self.map_goal         = None

        self._wp_lock      = threading.Lock()
        self.waypoints     = []
        self.wp_index      = 0
        self.search_phase  = 'TURNING'
        self._cmd_speed    = 0.0
        # Stuck detector state — reset whenever we advance to a new waypoint
        # or receive a fresh replan. See STALL_TICKS / STALL_IMPROVE_M.
        self._wp_min_dist  = float('inf')
        self._stall_count  = 0
        # Numbered live-map saves: incremented each time _replan_thread
        # completes so every replan produces a non-overwriting archive image.
        self._replan_count = 0

        self.replan_needed = False
        self.replanning    = False
        self._replan_lock  = threading.Lock()

        # Cube-finding sub-state:
        #   ALIGN_NEG_X → face -120° (60° CCW from -X)
        #   SWEEP       → 120° CW sweep through -X, recording red pixel counts
        #                 as relative yaw offsets from sweep_start_yaw
        #   ALIGN_CUBE  → rotate to face cube using relative yaw
        #   APPROACH    → creep forward until cube fills the frame
        self.cf_phase            = 'ALIGN_NEG_X'
        self.sweep_readings      = []
        self.cube_target_rel_yaw = None
        self.sweep_start_yaw     = None
        self.sweep_last_yaw      = None
        self.swept_total         = 0.0

        self.state = 'SEARCHING'

        self._load_and_plan()

        # Save the initial plan before the robot moves (live_map_001.png)
        self._replan_count += 1
        self._save_live_map(replan_index=self._replan_count)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('=== Autonomous search node started ===')

    # ── MAP LOADING ───────────────────────────────────────────────────────────

    def _load_and_plan(self):
        pgm_path  = os.path.join(MAP_DIR, 'map.pgm')
        yaml_path = os.path.join(MAP_DIR, 'map.yaml')
        self.get_logger().info(f'Loading Phase 1 map from {MAP_DIR} ...')

        free_grid, resolution, origin, _ = load_map(pgm_path, yaml_path)
        h, w = free_grid.shape
        self.map_resolution = resolution
        self.map_origin     = origin
        self.map_h          = h
        self.map_w          = w

        navigable = largest_free_region(free_grid)
        dist_map  = cv2.distanceTransform(navigable, cv2.DIST_L2, 5)

        self.phase1_navigable = navigable.copy().astype(np.uint8)
        self.phase1_dist      = dist_map
        self.live_grid        = navigable.copy().astype(np.uint8)
        self.hit_counts       = np.zeros((h, w), dtype=np.uint8)

        start_px = world_to_pixel(0.0, 0.0, origin, resolution, h)
        if navigable[start_px[0], start_px[1]] == 0:
            free_coords = np.argwhere(navigable == 1)
            dists = np.sum((free_coords - np.array(start_px))**2, axis=1)
            start_px = tuple(free_coords[np.argmin(dists)])

        self.map_goal = find_dead_end(navigable, dist_map, start_px)
        gx, gy = pixel_to_world(*self.map_goal, origin, resolution, h)
        self.get_logger().info(
            f'Dead-end goal: pixel {self.map_goal} → world ({gx:.2f}, {gy:.2f}) m')

        path = astar(navigable, dist_map, start_px, self.map_goal)
        if path is None:
            self.get_logger().error('Initial A* found no path.')
            return

        start_world = pixel_to_world(*start_px, origin, resolution, h)
        wp_pixels   = thin(path, WAYPOINT_STEP, start_world, origin, resolution, h)
        world_wps   = [pixel_to_world(r, c, origin, resolution, h) for r, c in wp_pixels]

        with self._wp_lock:
            self.waypoints = world_wps
            self.wp_index  = 0

        self.get_logger().info(
            f'Initial plan: {len(world_wps)} waypoints | '
            f'first=({world_wps[0][0]:.2f}, {world_wps[0][1]:.2f}) | '
            f'last=({world_wps[-1][0]:.2f}, {world_wps[-1][1]:.2f})')

    # ── SUBSCRIBER CALLBACKS ──────────────────────────────────────────────────

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

        if self.live_grid is None or self.state not in ('SEARCHING', 'WAITING_REPLAN'):
            return

        new_obstacles_found = []
        for i in range(0, n, SCAN_SAMPLE_RATE):
            r = msg.ranges[i]
            if not math.isfinite(r) or r < msg.range_min or r > SCAN_MAX_RANGE_M:
                continue
            robot_angle = (i - front_i) * inc
            world_angle = robot_angle + self.current_yaw
            wx = self.current_x + r * math.cos(world_angle)
            wy = self.current_y + r * math.sin(world_angle)
            col = int(round((wx - self.map_origin[0]) / self.map_resolution))
            row = self.map_h - int(round((wy - self.map_origin[1]) / self.map_resolution))
            if not (0 <= row < self.map_h and 0 <= col < self.map_w):
                continue
            if self.live_grid[row, col] != 1:
                continue
            if self.phase1_dist[row, col] < MIN_NEW_OBS_DIST_PX:
                continue
            self.hit_counts[row, col] += 1
            if self.hit_counts[row, col] >= HIT_THRESHOLD:
                self.live_grid[row, col] = 0
                new_obstacles_found.append((row, col))

        if not new_obstacles_found:
            return

        should_replan = False
        for row, col in new_obstacles_found:
            ox, oy = pixel_to_world(row, col,
                                    self.map_origin, self.map_resolution, self.map_h)
            dist = math.sqrt((ox - self.current_x)**2 + (oy - self.current_y)**2)
            if dist < REPLAN_NEARBY_M:
                should_replan = True
                break
        if not should_replan:
            should_replan = self._path_blocked()
        if should_replan and not self.replanning:
            self.replan_needed = True

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_z = msg.pose.pose.position.z
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
        self.latest_img = img
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
            cv2.inRange(hsv, RED_LOW2, RED_HIGH2))
        self.latest_red_pixels = cv2.countNonZero(mask)
        if self.state not in ('DETECTED', 'DONE', 'CUBE_FINDING', 'RETURNING'):
            if self.latest_red_pixels >= MIN_PIXELS:
                self.cube_detected = True

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def stop(self):
        self._cmd_speed = 0.0
        self.pub.publish(Twist())

    def _drive_ramped(self, target_speed, steer):
        """
        Publish a forward velocity command with smooth speed ramping.
        Accelerates at RAMP_ACCEL and decelerates at RAMP_DECEL per 100ms tick,
        avoiding the abrupt velocity steps that cause the robot to jerk.
        """
        if target_speed > self._cmd_speed:
            self._cmd_speed = min(target_speed, self._cmd_speed + RAMP_ACCEL)
        elif target_speed < self._cmd_speed:
            self._cmd_speed = max(target_speed, self._cmd_speed - RAMP_DECEL)
        self._publish_twist(self._cmd_speed, steer)

    def _publish_twist(self, lin, ang):
        cmd = Twist()
        cmd.linear.x  = lin
        cmd.angular.z = ang
        self.pub.publish(cmd)

    def _distance_to_wp(self):
        with self._wp_lock:
            if self.wp_index >= len(self.waypoints):
                return float('inf')
            wx, wy = self.waypoints[self.wp_index]
        return math.sqrt((self.current_x - wx)**2 + (self.current_y - wy)**2)

    def _distance_to_home(self):
        return math.sqrt(self.current_x**2 + self.current_y**2)

    def _heading_error(self):
        """Heading error to the current waypoint."""
        with self._wp_lock:
            if self.wp_index >= len(self.waypoints):
                return 0.0
            wx, wy = self.waypoints[self.wp_index]
        dx = wx - self.current_x
        dy = wy - self.current_y
        target = math.atan2(dy, dx)
        err    = target - self.current_yaw
        return math.atan2(math.sin(err), math.cos(err))

    def _heading_error_to_xy(self, tx, ty):
        """Heading error to an arbitrary world (x, y) coordinate."""
        dx = tx - self.current_x
        dy = ty - self.current_y
        target = math.atan2(dy, dx)
        err    = target - self.current_yaw
        return math.atan2(math.sin(err), math.cos(err))

    def _estimate_cube_pos(self):
        """
        Estimate the cube's world (x, y) position at the moment of detection.
        The cube is assumed to be 0.24 m directly in front of the robot,
        derived from the robot's current odometry position and heading.
            cube_x = robot_x + 0.24 * cos(yaw)
            cube_y = robot_y + 0.24 * sin(yaw)
        """
        cx = self.current_x + 0.24 * math.cos(self.current_yaw)
        cy = self.current_y + 0.24 * math.sin(self.current_yaw)
        return cx, cy

    def _path_blocked(self):
        with self._wp_lock:
            lookahead = self.waypoints[self.wp_index : self.wp_index + REPLAN_LOOKAHEAD]
        for wx, wy in lookahead:
            col = int(round((wx - self.map_origin[0]) / self.map_resolution))
            row = self.map_h - int(round((wy - self.map_origin[1]) / self.map_resolution))
            if 0 <= row < self.map_h and 0 <= col < self.map_w:
                if self.live_grid[row, col] == 0:
                    return True
        return False

    def _trigger_replan(self):
        self.replan_needed = False
        if time.time() - self._last_replan_time < REPLAN_COOLDOWN_S:
            return False
        with self._replan_lock:
            if self.replanning:
                return False
            self.replanning = True
        self.get_logger().info('New obstacle(s) detected — triggering background replan ...')
        threading.Thread(target=self._replan_thread, daemon=True).start()
        return True

    def _plan_return_path(self):
        """
        Plan a path from the robot's current position back to (0,0) using
        the live map so all detected Phase 2 obstacles are avoided.
        Runs synchronously — A* on this small map completes in <50 ms.
        """
        start_x = self.current_x
        start_y = self.current_y

        grid_snapshot = self.live_grid.copy()
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                       (ROBOT_RADIUS_PX * 2 + 1, ROBOT_RADIUS_PX * 2 + 1))
        inflated = cv2.erode(grid_snapshot, kernel)
        dist     = cv2.distanceTransform(inflated, cv2.DIST_L2, 5)

        def snap_to_free(px, grid):
            if grid[px[0], px[1]]:
                return px
            free_coords = np.argwhere(grid == 1)
            if len(free_coords) == 0:
                return px
            dists = np.sum((free_coords - np.array(px))**2, axis=1)
            return tuple(free_coords[np.argmin(dists)])

        start_px = world_to_pixel(start_x, start_y,
                                   self.map_origin, self.map_resolution, self.map_h)
        goal_px  = world_to_pixel(0.0, 0.0,
                                   self.map_origin, self.map_resolution, self.map_h)
        start_px = snap_to_free(start_px, inflated)
        goal_px  = snap_to_free(goal_px,  inflated)

        path = astar(inflated, dist, start_px, goal_px)

        if path is None:
            self.get_logger().warn('Return: full inflation failed — retrying with reduced inflation')
            kernel_small   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                 (ROBOT_RADIUS_PX + 1, ROBOT_RADIUS_PX + 1))
            inflated_small = cv2.erode(grid_snapshot, kernel_small)
            dist_small     = cv2.distanceTransform(inflated_small, cv2.DIST_L2, 5)
            start_px = snap_to_free(start_px, inflated_small)
            goal_px  = snap_to_free(goal_px,  inflated_small)
            path = astar(inflated_small, dist_small, start_px, goal_px)
            dist = dist_small

        if path is None:
            self.get_logger().error('Return: no path found — will head directly to (0,0)')
            return False

        start_world = (start_x, start_y)
        wp_pixels   = thin(path, WAYPOINT_STEP, start_world,
                           self.map_origin, self.map_resolution, self.map_h)
        world_wps   = [pixel_to_world(r, c, self.map_origin,
                                       self.map_resolution, self.map_h)
                       for r, c in wp_pixels]

        with self._wp_lock:
            self.waypoints = world_wps
            self.wp_index  = 0
        self.search_phase = 'TURNING'

        self.get_logger().info(
            f'Return path planned: {len(world_wps)} waypoints → (0,0)')

        # Save a numbered archive snapshot showing the return path overlaid
        # on the live map, so the return plan is preserved alongside the
        # outbound replan sequence (live_map_001.png, 002.png, …)
        self._replan_count += 1
        self._save_live_map(replan_index=self._replan_count)

        return True

    def _save_live_map(self, replan_index=None):
        """
        Save current live map state.

        Always writes ~/Downloads/map/live_map.png (the 'latest' view).
        When replan_index is provided (1, 2, 3 …) also writes a non-
        overwriting archive copy:  live_map_001.png, live_map_002.png, …
        """
        try:
            pgm_path  = os.path.join(MAP_DIR, 'map.pgm')
            yaml_path = os.path.join(MAP_DIR, 'map.yaml')
            img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
            if img is None or self.live_grid is None:
                return

            import yaml as _yaml
            with open(yaml_path) as f:
                meta = _yaml.safe_load(f)
            resolution = meta['resolution']
            origin     = meta['origin']

            h, w  = img.shape
            scale = 8
            vis   = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            vis   = cv2.resize(vis, (w * scale, h * scale),
                               interpolation=cv2.INTER_NEAREST)

            # Red squares for Phase 2 obstacles (cells free in Phase 1 map
            # but now marked occupied in the live grid)
            if self.phase1_navigable is not None:
                new_obs = (self.phase1_navigable == 1) & (self.live_grid == 0)
                for r, c in zip(*np.where(new_obs)):
                    vis[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = (0, 0, 255)

            # Cyan dots = remaining waypoints, grey = already passed
            with self._wp_lock:
                wps = list(self.waypoints)
                idx = self.wp_index
            for i, (wx, wy) in enumerate(wps):
                col = int(round((wx - origin[0]) / resolution))
                row = h - int(round((wy - origin[1]) / resolution))
                if 0 <= row < h and 0 <= col < w:
                    colour = (0, 220, 220) if i >= idx else (100, 100, 100)
                    cv2.circle(vis, (col*scale, row*scale), 4, colour, -1)

            # Green dot = current robot position
            rx_col = int(round((self.current_x - origin[0]) / resolution))
            rx_row = h - int(round((self.current_y - origin[1]) / resolution))
            if 0 <= rx_row < h and 0 <= rx_col < w:
                cv2.circle(vis, (rx_col*scale, rx_row*scale), 7, (60, 220, 60), -1)
                cv2.putText(vis, 'robot', (rx_col*scale + 8, rx_row*scale - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 220, 60), 1)

            # Always overwrite the 'latest' file
            cv2.imwrite(os.path.join(MAP_DIR, 'live_map.png'), vis)

            # Also write a numbered archive copy so no replan is ever lost
            if replan_index is not None:
                archive = os.path.join(MAP_DIR, f'live_map_{replan_index:03d}.png')
                cv2.imwrite(archive, vis)

        except Exception as e:
            self.get_logger().warn(f'live map save failed: {e}')

    def _replan_thread(self):
        try:
            start_x = self.current_x
            start_y = self.current_y
            grid_snapshot = self.live_grid.copy()

            kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                           (ROBOT_RADIUS_PX * 2 + 1, ROBOT_RADIUS_PX * 2 + 1))
            inflated = cv2.erode(grid_snapshot, kernel)
            dist     = cv2.distanceTransform(inflated, cv2.DIST_L2, 5)

            start_px = world_to_pixel(start_x, start_y,
                                      self.map_origin, self.map_resolution, self.map_h)
            if not inflated[start_px[0], start_px[1]]:
                free_coords = np.argwhere(inflated == 1)
                if len(free_coords) == 0:
                    self.get_logger().error(
                        'Replan: no free space after inflation — keeping existing plan.')
                    return
                dists    = np.sum((free_coords - np.array(start_px))**2, axis=1)
                start_px = tuple(free_coords[np.argmin(dists)])

            self.get_logger().info(
                f'Replan A*: from pixel {start_px} → goal {self.map_goal} ...')
            path = astar(inflated, dist, start_px, self.map_goal)

            if path is None:
                self.get_logger().warn(
                    'Replan: no path with full inflation — retrying with reduced inflation ...')
                kernel_small   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (ROBOT_RADIUS_PX + 1, ROBOT_RADIUS_PX + 1))
                inflated_small = cv2.erode(grid_snapshot, kernel_small)
                dist_small     = cv2.distanceTransform(inflated_small, cv2.DIST_L2, 5)
                if not inflated_small[start_px[0], start_px[1]]:
                    free_coords = np.argwhere(inflated_small == 1)
                    if len(free_coords) > 0:
                        dists    = np.sum((free_coords - np.array(start_px))**2, axis=1)
                        start_px = tuple(free_coords[np.argmin(dists)])
                path = astar(inflated_small, dist_small, start_px, self.map_goal)
                dist = dist_small

            if path is None:
                self.get_logger().error(
                    'Replan: no path found even with reduced inflation. '
                    'Robot will hold position until manually restarted.')
                return

            start_world = (start_x, start_y)
            wp_pixels   = thin(path, WAYPOINT_STEP, start_world,
                               self.map_origin, self.map_resolution, self.map_h)
            world_wps   = [pixel_to_world(r, c, self.map_origin,
                                          self.map_resolution, self.map_h)
                           for r, c in wp_pixels]

            skip_radius = 0.4
            start_idx   = 0
            for idx, (wx, wy) in enumerate(world_wps):
                d = math.sqrt((wx - start_x)**2 + (wy - start_y)**2)
                if d < skip_radius:
                    start_idx = idx + 1
                else:
                    break
            start_idx = min(start_idx, len(world_wps) - 1)

            with self._wp_lock:
                self.waypoints = world_wps
                self.wp_index  = start_idx

            self.search_phase = 'TURNING'
            # Reset stuck detector — fresh path means distance tracking restarts
            self._wp_min_dist = float('inf')
            self._stall_count  = 0

            self.get_logger().info(
                f'Replan complete: {len(world_wps)} waypoints | '
                f'start ({start_x:.2f}, {start_y:.2f}) → '
                f'goal ({world_wps[-1][0]:.2f}, {world_wps[-1][1]:.2f}) | '
                f'resuming from wp {start_idx+1}/{len(world_wps)}')

            # Increment counter and save both live_map.png and live_map_NNN.png
            self._replan_count += 1
            self._save_live_map(replan_index=self._replan_count)

        except Exception as e:
            self.get_logger().error(f'Replan thread error: {e}')

        finally:
            with self._replan_lock:
                self.replanning = False
            self._last_replan_time = time.time()

    def _init_cube_finding(self):
        """Reset cube-finding sweep state before entering CUBE_FINDING."""
        # Save a numbered snapshot the moment all waypoints are reached so
        # the final obstacle state going into the cube sweep is archived.
        self._replan_count += 1
        self._save_live_map(replan_index=self._replan_count)

        self.cf_phase            = 'ALIGN_NEG_X'
        self.sweep_readings      = []
        self.cube_target_rel_yaw = None
        self.sweep_start_yaw     = None
        self.sweep_last_yaw      = self.current_yaw
        self.swept_total         = 0.0

    # ── CONTROL LOOP (10 Hz) ──────────────────────────────────────────────────

    def control_loop(self):
        if self.state == 'DONE':
            return

        # Emergency stop (SEARCHING and WAITING_REPLAN only)
        if (self.state in ('SEARCHING', 'WAITING_REPLAN')
                and self.nearest_front < EMERGENCY_DIST):
            self.stop()
            if not self.replanning:
                if self._trigger_replan():
                    self.state = 'WAITING_REPLAN'
            return

        # ─────────────────────────────────────────────────────────────────────
        # WAITING_REPLAN
        # ─────────────────────────────────────────────────────────────────────
        if self.state == 'WAITING_REPLAN':
            if not self.replanning:
                self.get_logger().info('Replan done — resuming SEARCHING')
                self.search_phase = 'TURNING'
                self.state = 'SEARCHING'
            else:
                self.stop()
            return

        # ─────────────────────────────────────────────────────────────────────
        # SEARCHING — follow waypoints to the dead end
        # ─────────────────────────────────────────────────────────────────────
        if self.state == 'SEARCHING':

            if self.cube_detected:
                self.stop()
                self.cube_odom      = (self.current_x, self.current_y)
                self.cube_world_pos = self._estimate_cube_pos()
                self.get_logger().info(
                    f'RED CUBE DETECTED — robot odometry: '
                    f'x={self.current_x:.3f} m  y={self.current_y:.3f} m  '
                    f'yaw={math.degrees(self.current_yaw):.1f}°')
                self.get_logger().info(
                    f'Estimated cube world position (0.24 m ahead): '
                    f'x={self.cube_world_pos[0]:.3f} m  y={self.cube_world_pos[1]:.3f} m')
                if self.latest_img is not None:
                    snap_path = os.path.join(MAP_DIR, 'detection_snapshot.jpg')
                    cv2.imwrite(snap_path, self.latest_img)
                    self.get_logger().info(f'Detection snapshot saved to {snap_path}')
                self.state = 'DETECTED'
                return

            if self.replan_needed and not self.replanning:
                # Fire the background replan but keep driving on the existing
                # waypoints — the thread atomically swaps them when done (~50ms).
                # Only the emergency stop (nearest_front < EMERGENCY_DIST) needs
                # a hard stop; routine obstacle detection is 1.5m+ away.
                self._trigger_replan()

            with self._wp_lock:
                all_done = self.wp_index >= len(self.waypoints)
                n_wps    = len(self.waypoints)

            if all_done:
                self.get_logger().info('All waypoints reached — entering CUBE_FINDING')
                self._init_cube_finding()
                self.state = 'CUBE_FINDING'
                return

            if self.search_phase == 'TURNING':
                err = self._heading_error()
                self.get_logger().info(
                    f'[TURNING] wp {self.wp_index+1}/{n_wps}  err={math.degrees(err):.1f}°')
                if abs(err) < HEADING_TOL:
                    self.stop()
                    self.search_phase = 'DRIVING'
                    self.get_logger().info(f'Aligned — driving to waypoint {self.wp_index+1}')
                else:
                    self._publish_twist(0.0, TURN_SPEED if err > 0 else -TURN_SPEED)

            elif self.search_phase == 'DRIVING':
                dist = self._distance_to_wp()
                self.get_logger().info(
                    f'[DRIVING] wp {self.wp_index+1}/{n_wps}  dist={dist:.3f} m')
                if dist < WAYPOINT_RADIUS:
                    with self._wp_lock:
                        self.wp_index += 1
                        all_done = self.wp_index >= len(self.waypoints)
                    # Reset stuck detector for the new target waypoint
                    self._wp_min_dist = float('inf')
                    self._stall_count  = 0
                    if all_done:
                        self.stop()
                        self.get_logger().info('All waypoints reached — entering CUBE_FINDING')
                        self._init_cube_finding()
                        self.state = 'CUBE_FINDING'
                    else:
                        next_err = abs(self._heading_error())
                        if next_err > WAYPOINT_TURN_THRESHOLD:
                            self.stop()
                            self.search_phase = 'TURNING'
                            self.get_logger().info(
                                f'Waypoint reached — large turn ({math.degrees(next_err):.0f}°), '
                                f'stopping to align wp {self.wp_index+1}')
                        else:
                            self.get_logger().info(
                                f'Waypoint reached — curving to wp {self.wp_index+1} '
                                f'({math.degrees(next_err):.0f}°)')
                else:
                    # Stuck detector: if the robot hasn't closed the distance to
                    # this waypoint by STALL_IMPROVE_M within STALL_TICKS ticks
                    # (~2 s), it is likely circling an obstacle — force a replan
                    # from the current position so A* finds a fresh route.
                    if dist < self._wp_min_dist - STALL_IMPROVE_M:
                        self._wp_min_dist = dist
                        self._stall_count  = 0
                    else:
                        self._stall_count += 1
                        if self._stall_count >= STALL_TICKS:
                            self._stall_count  = 0
                            self._wp_min_dist  = float('inf')
                            self.get_logger().warn(
                                f'Stuck at wp {self.wp_index+1} '
                                f'(no progress for {STALL_TICKS} ticks) — forcing replan')
                            self._trigger_replan()
                    steer = max(-TURN_SPEED, min(TURN_SPEED,
                                self._heading_error() * WAYPOINT_TRACKING_GAIN))
                    self._drive_ramped(FORWARD_SPEED, steer)

        # ─────────────────────────────────────────────────────────────────────
        # CUBE_FINDING — structured sweep at the dead end
        #
        # ALIGN_NEG_X: face -120° (60° CCW from -X axis)
        # SWEEP:       rotate 120° CW through -X, recording red pixel counts
        #              as relative yaw offsets from sweep_start_yaw
        # ALIGN_CUBE:  rotate to face cube using relative yaw from sweep start
        # APPROACH:    creep forward until cube fills the frame
        # ─────────────────────────────────────────────────────────────────────
        elif self.state == 'CUBE_FINDING':

            if self.cf_phase == 'ALIGN_NEG_X':
                target_yaw = -2.0 * math.pi / 3.0
                err = math.atan2(
                    math.sin(target_yaw - self.current_yaw),
                    math.cos(target_yaw - self.current_yaw))
                if abs(err) < HEADING_TOL:
                    self.stop()
                    self.sweep_start_yaw = self.current_yaw
                    self.sweep_last_yaw  = self.current_yaw
                    self.swept_total     = 0.0
                    self.sweep_readings  = []
                    self.cf_phase = 'SWEEP'
                    self.get_logger().info(
                        f'Aligned — starting 120° CW sweep around -X '
                        f'(sweep_start_yaw={math.degrees(self.sweep_start_yaw):.1f}°)')
                else:
                    self._publish_twist(0.0, CUBE_TURN_SPEED if err > 0 else -CUBE_TURN_SPEED)

            elif self.cf_phase == 'SWEEP':
                delta = self.sweep_last_yaw - self.current_yaw
                if delta >  math.pi:  delta -= 2 * math.pi
                if delta < -math.pi:  delta += 2 * math.pi
                self.swept_total   += abs(delta)
                self.sweep_last_yaw = self.current_yaw

                rel_yaw = math.atan2(
                    math.sin(self.current_yaw - self.sweep_start_yaw),
                    math.cos(self.current_yaw - self.sweep_start_yaw))
                self.sweep_readings.append((rel_yaw, self.latest_red_pixels))

                self.get_logger().info(
                    f'[SWEEP] swept={math.degrees(self.swept_total):.1f}°  '
                    f'rel_yaw={math.degrees(rel_yaw):.1f}°  '
                    f'red_px={self.latest_red_pixels}')

                if self.swept_total >= math.radians(SWEEP_DEG):
                    self.stop()
                    above = [(rel_yaw, px) for rel_yaw, px in self.sweep_readings
                             if px >= CUBE_PIXEL_THRESHOLD]
                    if above:
                        self.cube_target_rel_yaw = above[len(above) // 2][0]
                        self.get_logger().info(
                            f'Sweep done — cube at relative yaw '
                            f'{math.degrees(self.cube_target_rel_yaw):.1f}° from sweep start')
                        self.cf_phase = 'ALIGN_CUBE'
                    else:
                        self.get_logger().warn('No cube detected during sweep — stopping')
                        self.state = 'DONE'
                else:
                    self._publish_twist(0.0, -CUBE_TURN_SPEED)

            elif self.cf_phase == 'ALIGN_CUBE':
                current_rel = math.atan2(
                    math.sin(self.current_yaw - self.sweep_start_yaw),
                    math.cos(self.current_yaw - self.sweep_start_yaw))
                err = math.atan2(
                    math.sin(self.cube_target_rel_yaw - current_rel),
                    math.cos(self.cube_target_rel_yaw - current_rel))
                if abs(err) < HEADING_TOL:
                    self.stop()
                    self.cf_phase = 'APPROACH'
                    self.get_logger().info('Facing cube — beginning approach')
                else:
                    self._publish_twist(0.0, CUBE_TURN_SPEED if err > 0 else -CUBE_TURN_SPEED)

            elif self.cf_phase == 'APPROACH':
                self.get_logger().info(
                    f'[APPROACH] red_px={self.latest_red_pixels} (stop at {CUBE_STOP_PIXELS})')
                if self.latest_red_pixels >= CUBE_STOP_PIXELS:
                    self.stop()
                    self.cube_odom      = (self.current_x, self.current_y)
                    self.cube_world_pos = self._estimate_cube_pos()
                    self.get_logger().info(
                        f'Cube reached — robot odometry: '
                        f'x={self.current_x:.3f} m  y={self.current_y:.3f} m  '
                        f'yaw={math.degrees(self.current_yaw):.1f}°')
                    self.get_logger().info(
                        f'Estimated cube world position (0.24 m ahead): '
                        f'x={self.cube_world_pos[0]:.3f} m  y={self.cube_world_pos[1]:.3f} m')
                    if self.latest_img is not None:
                        snap_path = os.path.join(MAP_DIR, 'cube_approach_snapshot.jpg')
                        cv2.imwrite(snap_path, self.latest_img)
                        self.get_logger().info(f'Approach snapshot saved to {snap_path}')
                    self.is_returning = True
                    if self._plan_return_path():
                        self.get_logger().info('Return path ready — beginning RETURNING')
                    else:
                        self.get_logger().warn(
                            'No A* return path — will head directly to (0,0)')
                    self.state = 'RETURNING'
                else:
                    self._drive_ramped(CUBE_FWD_SPEED, 0.0)

        # ─────────────────────────────────────────────────────────────────────
        # DETECTED — cube seen during SEARCHING (immediate pixel-count trigger)
        # ─────────────────────────────────────────────────────────────────────
        elif self.state == 'DETECTED':
            self.stop()
            self.is_returning = True
            if self._plan_return_path():
                self.get_logger().info('Return path ready — beginning RETURNING')
                self.state = 'RETURNING'
            else:
                self.get_logger().warn('No A* return path — heading directly to (0,0)')
                self.state = 'RETURNING'

        # ─────────────────────────────────────────────────────────────────────
        # RETURNING — follow A*-planned path back to (0,0)
        # ─────────────────────────────────────────────────────────────────────
        elif self.state == 'RETURNING':

            if self._distance_to_home() < HOME_RADIUS:
                self.stop()
                self.home_odom = (self.current_x, self.current_y, self.current_z)
                self.get_logger().info(
                    'Home reached — aligning to original heading (yaw=0°)')
                self.state = 'ALIGNING_HOME'
                return

            with self._wp_lock:
                all_done = self.wp_index >= len(self.waypoints)
                n_wps    = len(self.waypoints)

            if all_done:
                err = self._heading_error_to_xy(0.0, 0.0)
                self.get_logger().info(
                    f'[RETURNING/DIRECT] dist={self._distance_to_home():.3f} m  '
                    f'err={math.degrees(err):.1f}°')
                if abs(err) > HEADING_TOL:
                    self._publish_twist(0.0, TURN_SPEED if err > 0 else -TURN_SPEED)
                else:
                    self._drive_ramped(FORWARD_SPEED, 0.0)
                return

            if self.search_phase == 'TURNING':
                err = self._heading_error()
                self.get_logger().info(
                    f'[RETURNING/TURNING] wp {self.wp_index+1}/{n_wps}  '
                    f'err={math.degrees(err):.1f}°')
                if abs(err) < HEADING_TOL:
                    self.stop()
                    self.search_phase = 'DRIVING'
                else:
                    self._publish_twist(0.0, TURN_SPEED if err > 0 else -TURN_SPEED)

            elif self.search_phase == 'DRIVING':
                dist = self._distance_to_wp()
                self.get_logger().info(
                    f'[RETURNING/DRIVING] wp {self.wp_index+1}/{n_wps}  dist={dist:.3f} m')
                if dist < WAYPOINT_RADIUS:
                    with self._wp_lock:
                        self.wp_index += 1
                        all_done = self.wp_index >= len(self.waypoints)
                    if not all_done:
                        next_err = abs(self._heading_error())
                        if next_err > WAYPOINT_TURN_THRESHOLD:
                            self.stop()
                            self.search_phase = 'TURNING'
                else:
                    steer = max(-TURN_SPEED, min(TURN_SPEED, self._heading_error() * 2.0))
                    self._drive_ramped(FORWARD_SPEED, steer)

        # ─────────────────────────────────────────────────────────────────────
        # ALIGNING_HOME — rotate to face yaw=0 (original heading at run start)
        # ─────────────────────────────────────────────────────────────────────
        elif self.state == 'ALIGNING_HOME':
            err = math.atan2(
                math.sin(0.0 - self.current_yaw),
                math.cos(0.0 - self.current_yaw))
            self.get_logger().info(
                f'[ALIGNING_HOME] err={math.degrees(err):.1f}°')
            if abs(err) < HEADING_TOL:
                self.stop()
                self.get_logger().info('Aligned to original heading — DONE')
                self.state = 'DONE'
            else:
                self._publish_twist(0.0, TURN_SPEED if err > 0 else -TURN_SPEED)

        # ── Final stop and summary when DONE ──────────────────────────────────
        if self.state == 'DONE':
            self.stop()
            elapsed = time.time() - self.start_time
            mins    = int(elapsed // 60)
            secs    = int(elapsed % 60)

            lines = [
                '=== RUN COMPLETE ===',
                f'Total run time : {mins}m {secs}s',
                '',
            ]

            if self.cube_odom:
                lines.append(
                    f'Robot position at detection  : '
                    f'x={self.cube_odom[0]:.3f} m  y={self.cube_odom[1]:.3f} m')
            if self.cube_world_pos:
                lines.append(
                    f'Estimated cube world position: '
                    f'x={self.cube_world_pos[0]:.3f} m  y={self.cube_world_pos[1]:.3f} m  '
                    f'(0.24 m ahead of robot at detection)')

            lines.append('')

            if self.home_odom:
                lines.append(
                    f'Home reached (odometry)      : '
                    f'x={self.home_odom[0]:.3f} m  '
                    f'y={self.home_odom[1]:.3f} m  '
                    f'z={self.home_odom[2]:.3f} m  '
                    f'(error {self._distance_to_home():.3f} m from origin)')
            else:
                lines.append(
                    f'Final position               : '
                    f'x={self.current_x:.3f} m  y={self.current_y:.3f} m')

            for line in lines:
                self.get_logger().info(line)

            # Save identical summary to ~/Downloads/map/run_summary.txt
            try:
                summary_path = os.path.join(MAP_DIR, 'run_summary.txt')
                with open(summary_path, 'w') as f:
                    f.write('\n'.join(lines) + '\n')
                self.get_logger().info(f'Run summary saved → {summary_path}')
            except Exception as e:
                self.get_logger().warn(f'Could not save run summary: {e}')


# ── POST-RUN OVERLAY ──────────────────────────────────────────────────────────

def save_odom_map(trail, map_dir, outbound_trail=None, return_trail=None,
                  live_grid=None, phase1_navigable=None,
                  map_origin=None, map_resolution=None, map_h=None):
    """
    Save odom_overlay.png with outbound (orange) and return (green) trails.
    If live_grid and phase1_navigable are provided, also paints red squares
    for every cell that was free in the Phase 1 map but is now occupied —
    i.e. the Phase 2 obstacles the robot detected during the run.
    """
    pgm_path  = os.path.join(map_dir, 'map.pgm')
    yaml_path = os.path.join(map_dir, 'map.yaml')
    if not os.path.exists(pgm_path) or not os.path.exists(yaml_path):
        print(f'Map files not found in {map_dir} — skipping odom overlay')
        return

    import yaml as _yaml
    with open(yaml_path) as f:
        meta = _yaml.safe_load(f)
    resolution = meta['resolution']
    origin     = meta['origin']

    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    scale = 8
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = cv2.resize(vis, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # ── Obstacle overlay (drawn first so trails render on top) ────────────────
    if live_grid is not None and phase1_navigable is not None:
        new_obs = (phase1_navigable == 1) & (live_grid == 0)
        obs_rows, obs_cols = np.where(new_obs)
        for r, c in zip(obs_rows, obs_cols):
            vis[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = (0, 0, 255)

    # ── Trail lines ───────────────────────────────────────────────────────────
    def world_to_px(wx, wy):
        col = int(round((wx - origin[0]) / resolution))
        row = h - int(round((wy - origin[1]) / resolution))
        return col * scale, row * scale

    OUTBOUND_COLOUR = (0, 200, 255)   # orange
    RETURN_COLOUR   = (120, 255, 120) # green

    if outbound_trail and len(outbound_trail) > 1:
        pts = [world_to_px(x, y) for x, y in outbound_trail]
        for i in range(len(pts) - 1):
            cv2.line(vis, pts[i], pts[i+1], OUTBOUND_COLOUR, 2)

    if return_trail and len(return_trail) > 1:
        pts = [world_to_px(x, y) for x, y in return_trail]
        for i in range(len(pts) - 1):
            cv2.line(vis, pts[i], pts[i+1], RETURN_COLOUR, 2)

    if not outbound_trail and not return_trail and len(trail) > 1:
        pts = [world_to_px(x, y) for x, y in trail]
        for i in range(len(pts) - 1):
            cv2.line(vis, pts[i], pts[i+1], OUTBOUND_COLOUR, 2)

    # ── Start / end markers ───────────────────────────────────────────────────
    if trail:
        sx, sy = world_to_px(*trail[0])
        ex, ey = world_to_px(*trail[-1])
        cv2.circle(vis, (sx, sy), 7, (60,  60, 255), -1)
        cv2.putText(vis, 'start', (sx+6, sy-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1)
        cv2.circle(vis, (ex, ey), 7, (255, 255, 255), -1)
        cv2.putText(vis, 'end',   (ex+6, ey-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── Legend — placed dynamically in the grey (unknown) area outside the arena ──
    # Grey pixels in the ROS pgm (value 51–229) are unknown space outside the
    # navigable corridor. We score five candidate positions by how many grey
    # pixels they cover and pick the one with the highest score, so the legend
    # lands in a grey margin rather than on top of the map itself.
    legend_w   = 168
    legend_h   = 70 if (live_grid is not None) else 50

    # Build a scaled binary mask: 1 where the original pgm pixel is grey
    grey_orig   = (img > 50) & (img < 230)
    grey_scaled = np.kron(grey_orig.astype(np.uint8),
                          np.ones((scale, scale), dtype=np.uint8))
    img_h_sc, img_w_sc = grey_scaled.shape

    candidates = [
        (8,                    8),                               # top-left
        (8,                    img_h_sc // 2 - legend_h // 2),  # left-centre
        (8,                    img_h_sc - legend_h - 8),        # bottom-left
        (img_w_sc - legend_w - 8, 8),                           # top-right
        (img_w_sc - legend_w - 8, img_h_sc // 2 - legend_h // 2),  # right-centre
        (img_w_sc - legend_w - 8, img_h_sc - legend_h - 8),    # bottom-right
    ]

    lx, ly     = 8, 8   # fallback
    best_score = -1
    for cx, cy in candidates:
        if cx < 0 or cy < 0 or cx + legend_w > img_w_sc or cy + legend_h > img_h_sc:
            continue
        score = int(np.sum(grey_scaled[cy:cy + legend_h, cx:cx + legend_w]))
        if score > best_score:
            best_score = score
            lx, ly = cx, cy

    cv2.rectangle(vis, (lx, ly), (lx + legend_w, ly + legend_h), (40, 40, 40), -1)
    cv2.line(vis, (lx+6, ly+15), (lx+26, ly+15), OUTBOUND_COLOUR, 2)
    cv2.putText(vis, 'Outbound', (lx+32, ly+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, OUTBOUND_COLOUR, 1)
    cv2.line(vis, (lx+6, ly+35), (lx+26, ly+35), RETURN_COLOUR, 2)
    cv2.putText(vis, 'Returning', (lx+32, ly+40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, RETURN_COLOUR, 1)
    if live_grid is not None:
        cv2.rectangle(vis, (lx+6, ly+50), (lx+26, ly+62), (0, 0, 255), -1)
        cv2.putText(vis, 'Obstacle', (lx+32, ly+61),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    out_path = os.path.join(map_dir, 'odom_overlay.png')
    cv2.imwrite(out_path, vis)
    print(f'Odom overlay saved → {out_path}  ({len(trail)} points)')


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = WaypointNav()
    try:
        rclpy.spin(node)
    finally:
        save_odom_map(
            node.odom_trail, MAP_DIR,
            outbound_trail=node.outbound_trail,
            return_trail=node.return_trail,
            live_grid=node.live_grid,
            phase1_navigable=node.phase1_navigable,
            map_origin=node.map_origin,
            map_resolution=node.map_resolution,
            map_h=node.map_h,
        )
        node._save_live_map()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()