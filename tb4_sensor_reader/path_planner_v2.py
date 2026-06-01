"""
path_planner_v2.py

Reads map.pgm + map.yaml from ~/Downloads/map.
Automatically finds:
  - Robot origin  : world (0,0) derived from the YAML
  - Dead end      : the farthest reachable point through corridor centres

Plans an A* path from origin to dead end, weighted to stay
as far from walls as possible (follows the brightest/greenest
cells on the distance transform).

Outputs:
  - path_waypoints.csv   : (x_m, y_m) world coordinates
  - path_output.png      : visualisation

Usage:
    pip install opencv-python pyyaml
    python3 path_planner_v2.py
"""

import cv2
import numpy as np
import yaml
import heapq
import csv
import os
# ── Config ────────────────────────────────────────────────────────────────────
MAP_DIR   = os.path.expanduser("~/Downloads/map")
PGM_FILE  = os.path.join(MAP_DIR, "map.pgm")
YAML_FILE = os.path.join(MAP_DIR, "map.yaml")
OUT_CSV   = os.path.join(MAP_DIR, "path_waypoints.csv")
OUT_IMG   = os.path.join(MAP_DIR, "path_output.png")

# How strongly to prefer corridor centres vs raw distance.
# Higher = tighter centre-hugging. 5.0 works well for most arenas.
CENTRE_WEIGHT = 25.0

# Waypoint spacing: keep every Nth cell (N * resolution = metres between waypoints).
# 5 * 0.05 m = 0.25 m between waypoints.
WAYPOINT_STEP = 5

# Dead-end detection: bonus score per missing free neighbour (rewards enclosure).
ENCLOSURE_BONUS = 5.0


# ── Load map ──────────────────────────────────────────────────────────────────
def load_map(pgm_path, yaml_path):
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    resolution = meta["resolution"]
    origin     = meta["origin"]          # [x, y, theta] — bottom-left of image
    free_thresh = meta.get("free_thresh", 0.25)
    negate      = meta.get("negate", 0)

    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {pgm_path}")

    prob = img.astype(float) / 255.0
    if negate:
        prob = 1.0 - prob

    free_grid = (prob >= (1.0 - free_thresh)).astype(np.uint8)
    return free_grid, resolution, origin, img


# ── Main navigable region ─────────────────────────────────────────────────────
def largest_free_region(free_grid):
    num_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(
        free_grid.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        raise RuntimeError("No free space found in map.")
    # skip label 0 (background); find largest non-background region
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labeled == largest).astype(np.uint8)


# ── Coordinate helpers ────────────────────────────────────────────────────────
def world_to_pixel(wx, wy, origin, resolution, img_height):
    """World metres → (row, col) pixel. Y axis is flipped (image top = world max-Y)."""
    col = int(round((wx - origin[0]) / resolution))
    row = img_height - int(round((wy - origin[1]) / resolution))
    return row, col


def pixel_to_world(row, col, origin, resolution, img_height):
    """(row, col) pixel → (wx, wy) world metres."""
    wx = col * resolution + origin[0]
    wy = (img_height - row) * resolution + origin[1]
    return wx, wy


# ── Find dead end ─────────────────────────────────────────────────────────────
def find_dead_end(navigable, dist_map, start):
    """
    Finds the natural corridor terminus — the last high-clearance cell before
    the corridor closes off.

    Strategy:
      1. Compute the distance-weighted A* cost from start to every reachable cell
         (Dijkstra). This gives a 'how far along the corridor centre' measure.
      2. Find all local maxima of the distance transform (cells whose clearance
         is >= all 8 neighbours). These are the ridge/spine peaks of the corridor.
      3. Among those peaks, pick the one with the highest Dijkstra cost from start
         — that is the peak that is deepest into the corridor, which is the dead end.
    """
    h, w  = navigable.shape
    dirs8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    # Dijkstra from start along corridor centres (same cost map as A*)
    cost_map = 1.0 + CENTRE_WEIGHT / (dist_map + 0.5)
    cost_map[navigable == 0] = np.inf

    dijkstra = np.full((h, w), np.inf)
    dijkstra[start] = 0.0
    pq = [(0.0, start)]

    while pq:
        g, (r, c) = heapq.heappop(pq)
        if g > dijkstra[r, c]: continue
        for dr, dc in dirs8:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < h and 0 <= nc < w): continue
            if navigable[nr, nc] == 0: continue
            step = 1.414 if dr and dc else 1.0
            ng = g + step * cost_map[nr, nc]
            if ng < dijkstra[nr, nc]:
                dijkstra[nr, nc] = ng
                heapq.heappush(pq, (ng, (nr, nc)))

    # Find local maxima of the distance transform within navigable space
    # A cell is a local max if its dist value >= all 8 neighbours
    local_maxima = []
    for r in range(h):
        for c in range(w):
            if navigable[r, c] == 0 or np.isinf(dijkstra[r, c]):
                continue
            d = dist_map[r, c]
            is_peak = all(
                not (0 <= r+dr < h and 0 <= c+dc < w and navigable[r+dr,c+dc])
                or dist_map[r+dr, c+dc] <= d
                for dr, dc in dirs8
            )
            if is_peak:
                local_maxima.append((r, c))

    if not local_maxima:
        # Fallback: just return farthest reachable cell by Dijkstra cost
        reachable = np.where(navigable == 1, dijkstra, -np.inf)
        return tuple(np.unravel_index(np.argmax(reachable), (h, w)))

    # Pick the local maximum with the highest Dijkstra cost from start
    # = deepest into the corridor centre
    # also ensures it doesn't pick dead end position as a narrow gap near the cylinders
    dead_end = max(local_maxima, key=lambda rc: dijkstra[rc[0], rc[1]] * (dist_map[rc[0], rc[1]] ** 0.5))
    return dead_end


# ── A* weighted by wall distance ──────────────────────────────────────────────
def astar(navigable, dist_map, start, goal):
    """
    A* from start to goal.
    Move cost = step_length * (1 + CENTRE_WEIGHT / (dist_to_wall + 0.5))
    so the planner naturally stays in the green/bright corridor centre.
    """
    h, w  = navigable.shape
    dirs8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    cost_map = 1.0 + CENTRE_WEIGHT / (dist_map + 0.5)
    cost_map[navigable == 0] = np.inf

    def heur(a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    counter = 0
    pq = [(heur(start, goal), counter, 0.0, start)]
    came_from = {}
    g_score   = {start: 0.0}

    while pq:
        _, _, gc, cur = heapq.heappop(pq)

        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            path.reverse()
            return path

        if gc > g_score.get(cur, 1e18):
            continue

        for dr, dc in dirs8:
            nb = (cur[0]+dr, cur[1]+dc)
            if not (0 <= nb[0] < h and 0 <= nb[1] < w): continue
            if navigable[nb[0], nb[1]] == 0: continue
            step = 1.414 if dr and dc else 1.0
            ng   = g_score[cur] + step * cost_map[nb[0], nb[1]]
            if ng < g_score.get(nb, 1e18):
                came_from[nb] = cur
                g_score[nb]   = ng
                counter += 1
                heapq.heappush(pq, (ng + heur(nb, goal), counter, ng, nb))

    return None  # no path found


# ── Thin path to waypoints ────────────────────────────────────────────────────
def thin(path, step, start_world, origin, resolution, img_height,
         min_start_dist=0.10):
    """
    Thin the path to every Nth cell, then:
      1. Strip any waypoints within min_start_dist of the robot start.
      2. If the last two waypoints are within min_start_dist of each other,
         drop the second to last — keeping the final destination intact.
    """
    pts = path[::step]
    if pts[-1] != path[-1]:
        pts.append(path[-1])

    # Strip waypoints too close to start
    sx, sy = start_world
    filtered = []
    for r, c in pts:
        wx, wy = pixel_to_world(r, c, origin, resolution, img_height)
        dist = ((wx - sx)**2 + (wy - sy)**2) ** 0.5
        if dist >= min_start_dist:
            filtered.append((r, c))

    if not filtered:
        filtered = pts  # fallback

    # Drop second to last if too close to final waypoint
    if len(filtered) >= 2:
        r1, c1 = filtered[-2]
        r2, c2 = filtered[-1]
        wx1, wy1 = pixel_to_world(r1, c1, origin, resolution, img_height)
        wx2, wy2 = pixel_to_world(r2, c2, origin, resolution, img_height)
        if ((wx2 - wx1)**2 + (wy2 - wy1)**2) ** 0.5 < min_start_dist:
            filtered = filtered[:-2] + [filtered[-1]]

    return filtered


# ── Visualise ─────────────────────────────────────────────────────────────────
def visualise(navigable, dist_map, path, waypoints, start, goal, out_path):
    h, w  = navigable.shape
    scale = 8

    dist_norm = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.applyColorMap(dist_norm, cv2.COLORMAP_WINTER)
    vis[navigable == 0] = [20, 20, 20]
    vis = cv2.resize(vis, (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)

    for r, c in path:
        vis[r*scale, c*scale] = (255, 255, 255)

    for r, c in waypoints:
        cv2.circle(vis, (c*scale, r*scale), 4, (0, 220, 220), -1)

    r0, c0 = start
    cv2.circle(vis, (c0*scale, r0*scale), 8, (60, 60, 255), -1)
    cv2.putText(vis, "origin", (c0*scale+8, r0*scale-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1)

    rg, cg = goal
    cv2.circle(vis, (cg*scale, rg*scale), 8, (60, 220, 60), -1)
    cv2.putText(vis, "dead end", (cg*scale+8, rg*scale+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 1)

    cv2.imwrite(out_path, vis)
    print(f"Saved visualisation  →  {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Path Planner v2 — automatic dead-end detection")
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
    # Snap to nearest free cell in case origin lands on a wall pixel
    if navigable[start[0], start[1]] == 0:
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
    waypoints = thin(path, WAYPOINT_STEP, start_world, origin, resolution, h)
    world_wps = [pixel_to_world(r, c, origin, resolution, h) for r, c in waypoints]

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

    visualise(navigable, dist_map, path, waypoints, start, goal, OUT_IMG)

    print("\nFirst 10 waypoints (x, y) metres:")
    for x, y in world_wps[:10]:
        print(f"  ({x:.3f}, {y:.3f})")
    print("\nDone.")


if __name__ == "__main__":
    main()