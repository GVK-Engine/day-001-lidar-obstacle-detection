"""
run_kitti.py
============
Run our LiDAR pipeline on REAL KITTI data.
This is the moment our project becomes real.
"""

import numpy as np
import open3d as o3d
import time

# ── YOUR KITTI PATH ────────────────────────────────────
KITTI_BIN = r'C:\Users\vamsh\Downloads\kitti\2011_09_26_drive_0001_sync\2011_09_26\2011_09_26_drive_0001_sync\velodyne_points\data\0000000001.bin'

# ── LOAD REAL DATA ─────────────────────────────────────
print("\n" + "="*60)
print("  Running on REAL KITTI Data!")
print("="*60)

raw = np.fromfile(KITTI_BIN, dtype=np.float32).reshape(-1, 4)
print(f"\n  Loaded:          {len(raw):,} real LiDAR points")
print(f"  Forward range:   {raw[:,0].min():.1f}m to {raw[:,0].max():.1f}m")
print(f"  Side range:      {raw[:,1].min():.1f}m to {raw[:,1].max():.1f}m")
print(f"  Height range:    {raw[:,2].min():.1f}m to {raw[:,2].max():.1f}m")

# ── PIPELINE ───────────────────────────────────────────
t0 = time.perf_counter()

# Load into Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(raw[:, :3])

# ROI crop
pts  = np.asarray(pcd.points)
mask = (
    (pts[:, 0] >= 0)   & (pts[:, 0] <= 60) &
    (pts[:, 1] >= -20) & (pts[:, 1] <= 20) &
    (pts[:, 2] >= -3)  & (pts[:, 2] <= 2)
)
pcd = pcd.select_by_index(np.where(mask)[0])
print(f"\n  After ROI crop:    {len(pcd.points):,} points")

# Voxel downsample
pcd = pcd.voxel_down_sample(voxel_size=0.2)
print(f"  After downsample:  {len(pcd.points):,} points")

# Ground removal
_, inliers = pcd.segment_plane(
    distance_threshold=0.3,
    ransac_n=3,
    num_iterations=1000
)
objects = pcd.select_by_index(inliers, invert=True)
ground  = pcd.select_by_index(inliers)
print(f"  Ground points:     {len(ground.points):,}")
print(f"  Object points:     {len(objects.points):,}")

# Cluster
labels = np.array(
    objects.cluster_dbscan(eps=0.6, min_points=10)
)
n_clusters = max(0, labels.max() + 1)

# Filter valid clusters
clusters = []
for i in range(n_clusters):
    idx = np.where(labels == i)[0]
    if 15 <= len(idx) <= 3000:
        clusters.append(objects.select_by_index(idx))

runtime = (time.perf_counter() - t0) * 1000

# ── RESULTS ────────────────────────────────────────────
print(f"\n" + "="*60)
print(f"  DETECTED OBJECTS ON REAL ROAD")
print(f"="*60)
print(f"  {'ID':>3}  {'Distance':>10}  {'Size L×W×H':>18}  {'Type':>10}")
print(f"  {'─'*3}  {'─'*10}  {'─'*18}  {'─'*10}")

for i, cluster in enumerate(clusters):
    pts_c  = np.asarray(cluster.points)
    mins   = pts_c.min(axis=0)
    maxs   = pts_c.max(axis=0)
    center = (mins + maxs) / 2
    dims   = maxs - mins
    dist   = np.sqrt(center[0]**2 + center[1]**2)

    # Guess object type from size
    if dims[0] > 3.0 and dims[1] > 1.4:
        obj_type = "CAR"
    elif dims[2] > 1.2 and dims[0] < 1.2:
        obj_type = "PERSON"
    elif dims[0] > 1.5 and dims[2] > 1.0:
        obj_type = "CYCLIST"
    else:
        obj_type = "OBJECT"

    print(f"  {i:>3}  {dist:>8.1f}m  "
          f"  {dims[0]:.1f}×{dims[1]:.1f}×{dims[2]:.1f}m  "
          f"  {obj_type:>10}")

print(f"\n  Total objects:  {len(clusters)}")
print(f"  Runtime:        {runtime:.1f}ms")
print("="*60)

# ── VISUALIZE ──────────────────────────────────────────
print("\n  Opening 3D viewer...")
print("  REAL objects on a REAL road!")
print("  (drag=rotate  scroll=zoom  Q=quit)\n")

colors = [
    [0.9, 0.3, 0.3],
    [0.3, 0.6, 0.9],
    [0.3, 0.85, 0.4],
    [0.99, 0.75, 0.2],
    [0.7, 0.3, 0.9],
    [0.9, 0.5, 0.2],
    [0.2, 0.8, 0.8],
    [0.8, 0.2, 0.5],
]

geometries = []

# Ground (grey)
ground.paint_uniform_color([0.5, 0.5, 0.5])
geometries.append(ground)

# Each cluster with color and box
for i, cluster in enumerate(clusters):
    color = colors[i % len(colors)]
    cluster.paint_uniform_color(color)
    geometries.append(cluster)

    # Bounding box wireframe
    pts_c  = np.asarray(cluster.points)
    mins   = pts_c.min(axis=0)
    maxs   = pts_c.max(axis=0)
    cx, cy, cz = (mins + maxs) / 2
    lx = (maxs[0] - mins[0]) / 2
    ly = (maxs[1] - mins[1]) / 2
    lz = (maxs[2] - mins[2]) / 2

    corners = np.array([
        [cx-lx, cy-ly, cz-lz],
        [cx+lx, cy-ly, cz-lz],
        [cx+lx, cy+ly, cz-lz],
        [cx-lx, cy+ly, cz-lz],
        [cx-lx, cy-ly, cz+lz],
        [cx+lx, cy-ly, cz+lz],
        [cx+lx, cy+ly, cz+lz],
        [cx-lx, cy+ly, cz+lz],
    ])
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines  = o3d.utility.Vector2iVector(edges)
    ls.paint_uniform_color([1.0, 0.0, 0.0])
    geometries.append(ls)

# Origin frame
geometries.append(
    o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
)

o3d.visualization.draw_geometries(
    geometries,
    window_name=f"REAL KITTI Data — {len(clusters)} objects detected",
    width=1280,
    height=720,
)