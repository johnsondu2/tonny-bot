"""
map_planner.py

Reads map.pgm + map.yaml from ~/Downloads/map  (saved there by the Phase 1 bash script).

Automatically finds:
  - Robot origin  : world (0,0) derived from the YAML
  - Dead end      : the farthest reachable point through corridor centres

Plans an A* path from origin to dead end, weighted to stay
as far from walls as possible (follows corridor centres using
a distance-transform cost map).

Can be used two ways:
  1. Standalone:   python3 map_planner.py
     → writes path_waypoints.csv + path_output.png to MAP_DIR

  2. Imported module (used by autonomous_search.py at runtime):
     from map_planner import load_map, largest_free_region, astar, ...
     All public functions work independently with no global state.
"""

import cv2
import numpy as np
import yaml
import heapq
import csv
import os

# ── Standalone config (only used when run as __main__) ────────────────────────
MAP_DIR  = os.path.expanduser("~/Downloads/map")
PGM_FILE = os.path.join(MAP_DIR, "map.pgm")
YAML_FILE = os.path.join(MAP_DIR, "map.yaml")
OUT_CSV  = os.path.join(MAP_DIR, "path_waypoints.csv")
OUT_IMG  = os.path.join(MAP_DIR, "path_output.png")

# How strongly to prefer corridor centres vs raw distance.
# Higher = tighter centre-hugging. 25.0 works well for most arenas.
CENTRE_WEIGHT = 5000.0

# Waypoint spacing: keep every Nth cell (N * resolution = metres between waypoints).
# 5 * 0.05 m = 0.25 m between waypoints.
WAYPOINT_STEP = 3

# Dead-end detection: bonus score per missing free neighbour (rewards enclosure).
ENCLOSURE_BONUS = 5.0


# ── Map loading ───────────────────────────────────────────────────────────────

def load_map(pgm_path, yaml_path):
    """
    Load a ROS 2 nav2 map from its .pgm image and .yaml metadata files.

    Returns:
        free_grid   : (H, W) uint8 array — 1 = free cell, 0 = obstacle/unknown
        resolution  : metres per pixel (e.g. 0.05)
        origin      : [x, y, theta] real-world coordinates of bottom-left pixel
        raw_img     : (H, W) uint8 original grayscale image (for visualisation)
    """
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    resolution  = meta["resolution"]
    origin      = meta["origin"]           # [x, y, theta] of bottom-left corner
    free_thresh = meta.get("free_thresh", 0.25)
    negate      = meta.get("negate", 0)

    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read map image: {pgm_path}")

    # Convert pixel intensity to occupancy probability.
    # In a ROS .pgm map: white (255) = free, black (0) = occupied.
    prob = img.astype(float) / 255.0
    if negate:
        prob = 1.0 - prob

    # Mark cells as free only if their probability exceeds the free threshold.
    # Cells below occupied_thresh are obstacles; between thresholds = unknown.
    free_grid = (prob >= (1.0 - free_thresh)).astype(np.uint8)
    return free_grid, resolution, origin, img


# ── Navigable region ──────────────────────────────────────────────────────────

def largest_free_region(free_grid):
    """
    Find the largest connected component of free cells using 8-connectivity.
    This eliminates isolated free patches outside the navigable arena and
    ensures A* only plans within the main reachable area.

    Returns a (H, W) uint8 binary mask — 1 inside the region, 0 outside.
    """
    num_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(
        free_grid.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        raise RuntimeError("No free space found in map — check pgm/yaml files.")
    # Label 0 is background; find the largest non-background label by area.
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labeled == largest).astype(np.uint8)


# ── Coordinate conversion ─────────────────────────────────────────────────────

def world_to_pixel(wx, wy, origin, resolution, img_height):
    """
    Convert real-world (x, y) metres to map pixel (row, col).

    The map image has row 0 at the top, but ROS world Y increases upward,
    so the row is flipped: row = img_height - row_from_bottom.
    """
    col = int(round((wx - origin[0]) / resolution))
    row = img_height - int(round((wy - origin[1]) / resolution))
    return row, col


def pixel_to_world(row, col, origin, resolution, img_height):
    """
    Convert map pixel (row, col) to real-world (x, y) metres.
    Inverse of world_to_pixel.
    """
    wx = col * resolution + origin[0]
    wy = (img_height - row) * resolution + origin[1]
    return wx, wy


# ── Dead-end detection ────────────────────────────────────────────────────────

def find_dead_end(navigable, dist_map, start):
    """
    Find the natural corridor terminus — the last high-clearance cell before
    the corridor closes off.

    Strategy:
      1. Run Dijkstra from start using the distance-transform weighted cost.
         This gives a 'depth into corridor' measure for every reachable cell.
      2. Find all local maxima of the distance transform (corridor ridge cells).
      3. Return the local maximum with the highest Dijkstra cost = deepest into
         the corridor = the dead end the robot should search toward.
    """
    h, w  = navigable.shape
    dirs8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    # Cost map: cells near walls are expensive, corridor centres are cheap
    cost_map = 1.0 + CENTRE_WEIGHT / (dist_map + 0.5)
    cost_map[navigable == 0] = np.inf

    # Dijkstra from start
    dijkstra = np.full((h, w), np.inf)
    dijkstra[start] = 0.0
    pq = [(0.0, start)]

    while pq:
        g, (r, c) = heapq.heappop(pq)
        if g > dijkstra[r, c]:
            continue
        for dr, dc in dirs8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if navigable[nr, nc] == 0:
                continue
            step = 1.414 if (dr and dc) else 1.0
            ng = g + step * cost_map[nr, nc]
            if ng < dijkstra[nr, nc]:
                dijkstra[nr, nc] = ng
                heapq.heappush(pq, (ng, (nr, nc)))

    # Find local maxima of the distance transform within the navigable region.
    # A local maximum is a cell whose clearance value is >= all 8 neighbours.
    local_maxima = []
    for r in range(h):
        for c in range(w):
            if navigable[r, c] == 0 or np.isinf(dijkstra[r, c]):
                continue
            d = dist_map[r, c]
            is_peak = all(
                not (0 <= r+dr < h and 0 <= c+dc < w and navigable[r+dr, c+dc])
                or dist_map[r+dr, c+dc] <= d
                for dr, dc in dirs8
            )
            if is_peak:
                local_maxima.append((r, c))

    if not local_maxima:
        # Fallback: the cell that is hardest to reach from start
        reachable = np.where(navigable == 1, dijkstra, -np.inf)
        return tuple(np.unravel_index(np.argmax(reachable), (h, w)))

    # The deepest local maximum = most expensive to reach = dead end
    return max(local_maxima, key=lambda rc: dijkstra[rc[0], rc[1]] * (dist_map[rc[0], rc[1]] ** 0.5))


# ── A* path planner ───────────────────────────────────────────────────────────

def astar(navigable, dist_map, start, goal):
    """
    A* from start to goal on the navigable grid.

    Move cost = step_length × (1 + CENTRE_WEIGHT / (dist_to_wall + 0.5))

    This makes the planner naturally prefer corridor centres (high dist_map values)
    over paths that graze walls. The resulting path is safer and smoother.

    Returns the path as a list of (row, col) tuples from start to goal,
    or None if no path exists.
    """
    h, w  = navigable.shape
    dirs8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    cost_map = 1.0 + CENTRE_WEIGHT / (dist_map + 0.5)
    cost_map[navigable == 0] = np.inf

    def heur(a, b):
        # Euclidean distance heuristic (admissible for diagonal movement)
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    counter = 0  # Tie-breaker to avoid comparing tuples
    pq = [(heur(start, goal), counter, 0.0, start)]
    came_from = {}
    g_score   = {start: 0.0}

    while pq:
        _, _, gc, cur = heapq.heappop(pq)

        if cur == goal:
            # Reconstruct path by walking back through came_from
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            path.reverse()
            return path

        if gc > g_score.get(cur, 1e18):
            continue  # Stale entry in the priority queue

        for dr, dc in dirs8:
            nb = (cur[0]+dr, cur[1]+dc)
            if not (0 <= nb[0] < h and 0 <= nb[1] < w):
                continue
            if navigable[nb[0], nb[1]] == 0:
                continue
            step = 1.414 if (dr and dc) else 1.0
            ng   = g_score[cur] + step * cost_map[nb[0], nb[1]]
            if ng < g_score.get(nb, 1e18):
                came_from[nb] = cur
                g_score[nb]   = ng
                counter += 1
                heapq.heappush(pq, (ng + heur(nb, goal), counter, ng, nb))

    return None  # No path found


# ── Path thinning ─────────────────────────────────────────────────────────────

def thin(path, step, start_world, origin, resolution, img_height,
         min_start_dist=0.10):
    """
    Reduce a dense pixel-level path to evenly spaced waypoints.

    Steps:
      1. Keep every `step`-th cell from the path.
      2. Always include the final goal cell.
      3. Strip waypoints within min_start_dist of the robot's start position
         (they are already behind the robot or too close to matter).
      4. Merge the last two waypoints if they are within min_start_dist
         of each other (prevents a micro-step at the very end).

    Returns a list of (row, col) pixel coordinates.
    """
    pts = path[::step]
    if pts[-1] != path[-1]:
        pts.append(path[-1])  # Always include the goal

    # Strip waypoints too close to start position
    sx, sy = start_world
    filtered = []
    for r, c in pts:
        wx, wy = pixel_to_world(r, c, origin, resolution, img_height)
        if ((wx - sx)**2 + (wy - sy)**2) ** 0.5 >= min_start_dist:
            filtered.append((r, c))

    if not filtered:
        filtered = pts  # Fallback: nothing was close enough to strip

    # Merge last two waypoints if they are very close together
    if len(filtered) >= 2:
        r1, c1 = filtered[-2]
        r2, c2 = filtered[-1]
        wx1, wy1 = pixel_to_world(r1, c1, origin, resolution, img_height)
        wx2, wy2 = pixel_to_world(r2, c2, origin, resolution, img_height)
        if ((wx2 - wx1)**2 + (wy2 - wy1)**2) ** 0.5 < min_start_dist:
            filtered = filtered[:-2] + [filtered[-1]]

    return filtered


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualise(navigable, dist_map, path, waypoints, start, goal, out_path):
    """
    Save a colour visualisation of the planned path overlaid on the distance
    transform. Blue = robot origin, green = dead-end goal, cyan dots = waypoints,
    white pixels = planned path.
    """
    h, w  = navigable.shape
    scale = 8  # Upscale for visibility

    dist_norm = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.applyColorMap(dist_norm, cv2.COLORMAP_WINTER)
    vis[navigable == 0] = [20, 20, 20]  # Dark grey for obstacles/unknown
    vis = cv2.resize(vis, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    for r, c in path:
        vis[r * scale, c * scale] = (255, 255, 255)  # White = path

    for r, c in waypoints:
        cv2.circle(vis, (c * scale, r * scale), 4, (0, 220, 220), -1)  # Cyan = waypoints

    r0, c0 = start
    cv2.circle(vis, (c0 * scale, r0 * scale), 8, (60, 60, 255), -1)
    cv2.putText(vis, "origin", (c0 * scale + 8, r0 * scale - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1)

    rg, cg = goal
    cv2.circle(vis, (cg * scale, rg * scale), 8, (60, 220, 60), -1)
    cv2.putText(vis, "dead end", (cg * scale + 8, rg * scale + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 1)

    cv2.imwrite(out_path, vis)
    print(f"Saved visualisation → {out_path}")


# ── Standalone entry point ────────────────────────────────────────────────────

def main():
    """
    Run the path planner as a standalone script.
    Reads the Phase 1 map from ~/Downloads/map, plans a path, and writes
    path_waypoints.csv + path_output.png to the same directory.

    This is run after Phase 1 mapping is complete to preview the planned path
    before Phase 2 begins. The autonomous_search.py node re-runs the same
    planning at startup, so the CSV is not required by Phase 2.
    """
    print("=" * 55)
    print("  Path Planner — automatic dead-end detection")
    print("=" * 55)

    print("\n[1/5] Loading map...")
    free_grid, resolution, origin, raw_img = load_map(PGM_FILE, YAML_FILE)
    h, w = free_grid.shape
    print(f"      Size       : {w} x {h} px  "
          f"({w*resolution:.2f} x {h*resolution:.2f} m)")
    print(f"      Resolution : {resolution} m/px")
    print(f"      Origin     : {origin[:2]} m  (bottom-left of image)")

    print("\n[2/5] Building navigable region + distance map...")
    navigable = largest_free_region(free_grid)
    dist_map  = cv2.distanceTransform(navigable, cv2.DIST_L2, 5)
    print(f"      Max wall clearance : {dist_map.max()*resolution:.2f} m")

    print("\n[3/5] Locating start and dead end...")
    start = world_to_pixel(0.0, 0.0, origin, resolution, h)
    if navigable[start[0], start[1]] == 0:
        # Snap to nearest free cell if origin lands on a wall pixel
        free_coords = np.argwhere(navigable == 1)
        dists = np.sum((free_coords - np.array(start))**2, axis=1)
        start = tuple(free_coords[np.argmin(dists)])

    goal = find_dead_end(navigable, dist_map, start)

    sx, sy = pixel_to_world(*start, origin, resolution, h)
    gx, gy = pixel_to_world(*goal,  origin, resolution, h)
    print(f"      Start    : pixel {start}  →  world ({sx:.2f}, {sy:.2f}) m")
    print(f"      Dead end : pixel {goal}   →  world ({gx:.2f}, {gy:.2f}) m")

    print("\n[4/5] Running A*...")
    path = astar(navigable, dist_map, start, goal)
    if path is None:
        print("  !! No path found. Check your map.")
        return

    start_world = pixel_to_world(*start, origin, resolution, h)
    wp_pixels   = thin(path, WAYPOINT_STEP, start_world, origin, resolution, h)
    world_wps   = [pixel_to_world(r, c, origin, resolution, h) for r, c in wp_pixels]

    total_dist = sum(
        ((world_wps[i+1][0]-world_wps[i][0])**2 +
         (world_wps[i+1][1]-world_wps[i][1])**2)**0.5
        for i in range(len(world_wps)-1)
    )
    print(f"      Path length : {total_dist:.2f} m")
    print(f"      Waypoints   : {len(world_wps)}  (~{WAYPOINT_STEP*resolution:.2f} m apart)")

    print("\n[5/5] Saving outputs...")
    os.makedirs(MAP_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_m", "y_m"])
        writer.writerows([(f"{x:.4f}", f"{y:.4f}") for x, y in world_wps])
    print(f"      Waypoints CSV →  {OUT_CSV}")

    visualise(navigable, dist_map, path, wp_pixels, start, goal, OUT_IMG)

    print("\nFirst 10 waypoints (x, y) metres:")
    for x, y in world_wps[:10]:
        print(f"  ({x:.3f}, {y:.3f})")
    print("\nDone.")


if __name__ == "__main__":
    main()