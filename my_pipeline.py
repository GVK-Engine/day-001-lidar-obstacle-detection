"""
my_pipeline.py
==============
My first LiDAR obstacle detection pipeline.
I built this on Day 1 of my 90-day robotics portfolio.

What it does:
  1. Load LiDAR point cloud
  2. Crop to region we care about
  3. Downsample to reduce points
  4. Remove the ground
  5. Cluster remaining points into objects
  6. Draw boxes around each object
  7. Visualize everything
"""

# ── IMPORTS ──────────────────────────────────────────────────────────
import numpy as np        # fast math on arrays
import open3d as o3d      # 3D point cloud processing
import time               # for measuring how long things take

print("All libraries imported successfully!")


# ── STEP 1: CREATE SYNTHETIC DATA (no KITTI file needed yet) ─────────

def create_demo_scene():
    """
    Create a fake LiDAR scene with:
    - A flat ground plane
    - 3 cars (box-shaped clusters)
    - 2 pedestrians (smaller clusters)

    In real life, this data comes from the LiDAR sensor.
    """
    all_points = []

    # Ground plane
    # Lots of points near z = -1.7 (ground is 1.7m below the LiDAR on roof)
    n_ground = 20000
    gx = np.random.uniform(-30, 60, n_ground)   # forward/back
    gy = np.random.uniform(-15, 15, n_ground)   # left/right
    gz = np.random.normal(-1.7, 0.05, n_ground) # height (ground)
    ground = np.column_stack([gx, gy, gz])      # combine into (N, 3)
    all_points.append(ground)
    print(f"Ground: {len(ground)} points")

    # Car 1 — about 15 meters ahead, slightly left
    car1_center = [15.0, -2.0, -1.0]
    n_car = 200
    car1x = np.random.uniform(car1_center[0]-2.2, car1_center[0]+2.2, n_car)
    car1y = np.random.uniform(car1_center[1]-0.9, car1_center[1]+0.9, n_car)
    car1z = np.random.uniform(car1_center[2]-0.7, car1_center[2]+0.7, n_car)
    car1 = np.column_stack([car1x, car1y, car1z])
    all_points.append(car1)
    print(f"Car 1: {len(car1)} points at {car1_center}")

    # Car 2 — about 30 meters ahead, slightly right
    car2_center = [30.0, 3.0, -1.0]
    car2x = np.random.uniform(car2_center[0]-2.2, car2_center[0]+2.2, n_car)
    car2y = np.random.uniform(car2_center[1]-0.9, car2_center[1]+0.9, n_car)
    car2z = np.random.uniform(car2_center[2]-0.7, car2_center[2]+0.7, n_car)
    car2 = np.column_stack([car2x, car2y, car2z])
    all_points.append(car2)
    print(f"Car 2: {len(car2)} points at {car2_center}")

    # Pedestrian — about 8 meters ahead, left
    ped_center = [8.0, -4.0, -0.8]
    n_ped = 40
    pedx = np.random.uniform(ped_center[0]-0.3, ped_center[0]+0.3, n_ped)
    pedy = np.random.uniform(ped_center[1]-0.25, ped_center[1]+0.25, n_ped)
    pedz = np.random.uniform(ped_center[2]-0.85, ped_center[2]+0.85, n_ped)
    ped = np.column_stack([pedx, pedy, pedz])
    all_points.append(ped)
    print(f"Pedestrian: {len(ped)} points at {ped_center}")

    # Stack everything into one big array
    all_points_combined = np.vstack(all_points)
    # vstack = stack arrays vertically (add rows)
    # Like stacking tables on top of each other

    print(f"\nTotal scene: {len(all_points_combined)} points")
    return all_points_combined


# ── STEP 2: LOAD INTO OPEN3D ─────────────────────────────────────────

def numpy_to_pointcloud(points):
    """Convert numpy array to Open3D PointCloud object."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# ── STEP 3: CROP REGION OF INTEREST ──────────────────────────────────

def crop_roi(pcd):
    """
    Keep only points in the driving corridor.
    Remove points that are:
    - Behind the car (x < 0)
    - Too far ahead (x > 60m)
    - Too far left/right (|y| > 20m)
    - Above the car (z > 2m) or too far below (z < -3m)
    """
    pts = np.asarray(pcd.points)   # get the numpy array from pcd

    # Create a boolean mask
    # True = keep this point
    # False = remove this point
    mask = (
        (pts[:, 0] >= 0)    &    # x >= 0     (not behind us)
        (pts[:, 0] <= 60)   &    # x <= 60m   (not too far)
        (pts[:, 1] >= -20)  &    # y >= -20m  (not too far left)
        (pts[:, 1] <= 20)   &    # y <= 20m   (not too far right)
        (pts[:, 2] >= -3)   &    # z >= -3m   (not underground)
        (pts[:, 2] <= 2)         # z <= 2m    (not above car)
    )

    # Get indices where mask is True
    keep_indices = np.where(mask)[0]

    # Select only those points
    cropped = pcd.select_by_index(keep_indices)

    before = len(pts)
    after  = len(cropped.points)
    print(f"ROI crop: {before} → {after} points")
    return cropped


# ── STEP 4: VOXEL DOWNSAMPLE ─────────────────────────────────────────

def downsample(pcd, voxel_size=0.2):
    """
    Reduce number of points using voxel grid.
    Each 0.2m × 0.2m × 0.2m cube → replaced by 1 point.
    """
    before = len(pcd.points)
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    after = len(downsampled.points)
    reduction = (1 - after/before) * 100
    print(f"Downsample (voxel={voxel_size}m): {before} → {after} points ({reduction:.0f}% reduction)")
    return downsampled


# ── STEP 5: REMOVE GROUND ────────────────────────────────────────────

def remove_ground(pcd):
    """
    Use RANSAC to find and remove the ground plane.
    Returns (objects, ground) as two separate PointClouds.
    """
    # segment_plane finds the best-fitting plane
    plane_model, inlier_indices = pcd.segment_plane(
        distance_threshold = 0.3,    # 0.3m from plane = ground
        ransac_n           = 3,      # need 3 points to define a plane
        num_iterations     = 1000,   # try 1000 random plane fits
    )

    # plane_model = [a, b, c, d] where ax + by + cz + d = 0
    a, b, c, d = plane_model
    print(f"Ground plane: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # Split into ground and non-ground
    objects = pcd.select_by_index(inlier_indices, invert=True)
    ground  = pcd.select_by_index(inlier_indices)

    print(f"Ground removal: {len(ground.points)} ground pts | {len(objects.points)} object pts")
    return objects, ground


# ── STEP 6: CLUSTER OBJECTS ──────────────────────────────────────────

def find_clusters(pcd):
    """
    Use DBSCAN to group nearby points into clusters.
    Each cluster = one detected object.
    """
    # cluster_dbscan returns a label for each point
    labels = np.array(
        pcd.cluster_dbscan(
            eps        = 0.6,    # points within 0.6m = same cluster
            min_points = 10,     # need 10+ points to form a cluster
        )
    )

    # -1 means "noise" (not in any cluster)
    n_clusters = labels.max() + 1
    n_noise    = np.sum(labels == -1)
    print(f"Clustering: {n_clusters} objects found, {n_noise} noise points")

    # Extract each cluster as its own PointCloud
    clusters = []
    for cluster_id in range(n_clusters):
        # Get indices of all points belonging to this cluster
        indices = np.where(labels == cluster_id)[0]

        # Only keep clusters with reasonable size
        # Too small = noise, Too big = multiple objects merged
        if 15 <= len(indices) <= 3000:
            cluster_pcd = pcd.select_by_index(indices)
            clusters.append(cluster_pcd)

    print(f"Valid clusters (after size filter): {len(clusters)}")
    return clusters


# ── STEP 7: DRAW BOUNDING BOXES ──────────────────────────────────────

def get_bounding_box(cluster_pcd, cluster_id):
    """
    Find the smallest box that contains all points in this cluster.
    Returns the box corners, center, and size.
    """
    pts  = np.asarray(cluster_pcd.points)

    # min and max for each axis
    mins = pts.min(axis=0)   # [min_x, min_y, min_z]
    maxs = pts.max(axis=0)   # [max_x, max_y, max_z]

    center     = (mins + maxs) / 2.0
    dimensions = maxs - mins

    print(f"  Cluster {cluster_id}: "
          f"center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})  "
          f"size={dimensions[0]:.1f}×{dimensions[1]:.1f}×{dimensions[2]:.1f}m  "
          f"({len(pts)} pts)")

    return center, dimensions


def make_box_lineset(center, dimensions, color):
    """
    Create a wireframe box (12 edges) from center + dimensions.
    This is what we draw around each detected object.
    """
    cx, cy, cz = center
    lx = dimensions[0] / 2.0   # half-length in x
    ly = dimensions[1] / 2.0   # half-length in y
    lz = dimensions[2] / 2.0   # half-length in z

    # 8 corners of the box
    corners = np.array([
        [cx-lx, cy-ly, cz-lz],   # corner 0: front-left-bottom
        [cx+lx, cy-ly, cz-lz],   # corner 1: front-right-bottom
        [cx+lx, cy+ly, cz-lz],   # corner 2: back-right-bottom
        [cx-lx, cy+ly, cz-lz],   # corner 3: back-left-bottom
        [cx-lx, cy-ly, cz+lz],   # corner 4: front-left-top
        [cx+lx, cy-ly, cz+lz],   # corner 5: front-right-top
        [cx+lx, cy+ly, cz+lz],   # corner 6: back-right-top
        [cx-lx, cy+ly, cz+lz],   # corner 7: back-left-top
    ])

    # 12 edges connecting the 8 corners
    edges = [
        # Bottom face (4 edges)
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face (4 edges)
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Verticals connecting top and bottom (4 edges)
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(corners)
    lineset.lines  = o3d.utility.Vector2iVector(edges)
    lineset.paint_uniform_color(color)
    return lineset


# ── STEP 8: VISUALIZE ────────────────────────────────────────────────

def visualize_scene(clusters, ground):
    """
    Show everything in an interactive 3D window.
    Mouse: drag to rotate, scroll to zoom, Q to quit.
    """
    # Colors for each cluster (different color per object)
    cluster_colors = [
        [0.9, 0.3, 0.3],   # red
        [0.3, 0.6, 0.9],   # blue
        [0.3, 0.85, 0.4],  # green
        [0.99, 0.75, 0.2], # yellow
        [0.7, 0.3, 0.9],   # purple
    ]

    geometries = []

    # Add ground (grey)
    ground.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(ground)

    # Add each cluster with its color + bounding box
    for i, cluster in enumerate(clusters):
        color = cluster_colors[i % len(cluster_colors)]

        # Color the cluster points
        cluster.paint_uniform_color(color)
        geometries.append(cluster)

        # Add bounding box wireframe
        pts   = np.asarray(cluster.points)
        mins  = pts.min(axis=0)
        maxs  = pts.max(axis=0)
        center     = (mins + maxs) / 2.0
        dimensions = maxs - mins
        box = make_box_lineset(center, dimensions, color=[1.0, 0.0, 0.0])
        geometries.append(box)

    # Add coordinate frame (X=red, Y=green, Z=blue arrows at origin)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    geometries.append(frame)

    # Open the viewer
    o3d.visualization.draw_geometries(
        geometries,
        window_name = f"LiDAR Obstacle Detection — {len(clusters)} objects detected",
        width       = 1280,
        height      = 720,
    )


# ── MAIN: RUN EVERYTHING ─────────────────────────────────────────────

def run_pipeline():
    print("\n" + "="*60)
    print("  LiDAR Obstacle Detection Pipeline")
    print("  Day 1 — Vamshikrishna Gadde")
    print("="*60 + "\n")

    start_time = time.perf_counter()   # start the clock

    # Step 1: Create demo data
    print("STEP 1: Creating demo scene...")
    raw_points = create_demo_scene()

    # Step 2: Load into Open3D
    print("\nSTEP 2: Loading into Open3D...")
    pcd = numpy_to_pointcloud(raw_points)

    # Step 3: Crop ROI
    print("\nSTEP 3: Cropping region of interest...")
    pcd = crop_roi(pcd)

    # Step 4: Downsample
    print("\nSTEP 4: Voxel downsampling (voxel=0.2m)...")
    pcd = downsample(pcd, voxel_size=0.2)

    # Step 5: Ground removal
    print("\nSTEP 5: RANSAC ground removal...")
    objects, ground = remove_ground(pcd)

    # Step 6: Cluster
    print("\nSTEP 6: DBSCAN clustering...")
    clusters = find_clusters(objects)

    # Step 7: Bounding boxes
    print("\nSTEP 7: Computing bounding boxes...")
    for i, cluster in enumerate(clusters):
        get_bounding_box(cluster, i)

    # Stop clock
    end_time   = time.perf_counter()
    runtime_ms = (end_time - start_time) * 1000

    print("\n" + "="*60)
    print(f"  PIPELINE COMPLETE")
    print(f"  Objects detected: {len(clusters)}")
    print(f"  Runtime:          {runtime_ms:.1f} ms")
    print("="*60)

    # Step 8: Visualize
    print("\nSTEP 8: Opening 3D viewer...")
    print("  Controls: drag=rotate  scroll=zoom  Q=quit\n")
    visualize_scene(clusters, ground)


# Run it!
if __name__ == "__main__":
    run_pipeline()