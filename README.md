# Day 1 - LiDAR Obstacle Detection Pipeline

**Portfolio Series 1: Perception**

MS Robotics and Autonomous Systems Engineering, Arizona State University, Dec 2026.

---

## The Problem

A LiDAR sensor fires 100,000 laser beams per second.
What comes back is 100,000 floating dots in 3D space - no labels, no object names, just coordinates.

A self-driving car looking at raw LiDAR data cannot detect anything.
It cannot brake for a pedestrian it does not know exists.
It cannot steer around a car it cannot identify.

This pipeline solves that. It takes raw, unstructured point cloud data and outputs a list of detected 3D obstacles with their positions, sizes, and distances - in real time.

---

## What the Industry Does Today

Companies use PCL (Point Cloud Library) as a starting foundation.
PCL provides raw geometric tools, not complete detection solutions.
Every AV company - Waymo, Cruise, Aurora - builds their own pipeline on top of these primitives.
That is exactly what this project implements, from raw binary file to labeled 3D bounding boxes.

---

## Pipeline

```
Raw KITTI .bin file
        |
        v
ROI Crop                remove points outside driving corridor
        |               (x: 0-60m, y: +/-20m, z: -3m to 2m)
        v
Voxel Downsample        reduce 120k points to ~15k
        |               each 0.2m cube becomes one point
        v
RANSAC Ground Removal   fit dominant ground plane, remove it
        |               1000 iterations, 0.3m threshold
        v
DBSCAN Clustering       group nearby non-ground points
        |               eps=0.6m, min 10 points per cluster
        v
Bounding Boxes          axis-aligned 3D box per valid cluster
        |               filter: 15 to 3000 points per cluster
        v
Open3D Visualization    interactive 3D viewer
```

---

## My Contribution - Voxel Size Trade-off Analysis

Most implementations pick a voxel size and never question it.
I tested three configurations across 10 frames and made a data-driven decision.

| Voxel Size | Avg Runtime | Avg Clusters | Verdict |
|------------|-------------|--------------|---------|
| 0.1m       | 9.2ms       | 5.0          | Minimal speed gain over 0.2m, not worth the extra density |
| **0.2m**   | **8.3ms**   | **4.9**      | **Optimal - best balance of speed and accuracy** |
| 0.4m       | 6.7ms       | 8.5          | Creates false detections - dangerous for AV systems |

The 0.4m setting finds 8.5 clusters when only 3-4 real objects exist.
Those extra detections are ghost objects that do not exist on the road.
In a self-driving car, ghost detections cause unnecessary emergency braking.
The 1.6ms speed gain is not worth the safety risk.

voxel=0.2m is the only setting that is both fast enough for real-time and accurate enough for safe operation.

---

## Results

### Demo Mode (synthetic scene)

Three known objects placed at exact positions: 2 cars and 1 pedestrian.

```
Pipeline output:
  Cluster 0: center=(15.1, -2.0, -0.8)  size=4.3 x 1.7 x 1.1m  (122 pts)  -> Car
  Cluster 1: center=(30.2,  3.0, -0.8)  size=4.1 x 1.8 x 1.1m  (122 pts)  -> Car
  Cluster 2: center=( 8.0, -4.0, -0.6)  size=0.5 x 0.5 x 1.4m  ( 24 pts)  -> Pedestrian

  Objects detected: 3
  Runtime:          16.9ms
```

Demo result (3D viewer):
https://drive.google.com/file/d/1c_QR9fI8jygRXkXXYF_KOKMvno3ocpen/view?usp=drive_link

---

### Real KITTI Data - 2011_09_26_drive_0001, Frame 1

Data recorded from a Velodyne HDL-64E LiDAR mounted on a car
driving in Karlsruhe, Germany, 2011.

```
Loaded:              121,151 real LiDAR points
After ROI crop:       59,923 points
After downsample:     15,759 points
Ground points:         9,507
Object points:         6,252

Detected objects:         28
  Including: Cars, Cyclists, Pedestrians
Runtime:              86.3ms
```

Sample detections:
```
 ID    Distance    Size (L x W x H)    Type
  0      29.3m    2.8 x 1.5 x 1.5m    Cyclist
  7      20.0m    0.5 x 0.2 x 1.6m    Person
  8      21.6m    7.7 x 4.5 x 2.7m    Car
  9      25.2m    4.2 x 1.9 x 1.7m    Car
 18      21.6m    0.4 x 0.5 x 1.8m    Person
```

Real KITTI result (3D viewer):
https://drive.google.com/file/d/1R6OG_-UJj--JHNBOirxky3MNCNLwCxnO/view?usp=sharing

---

## Notes on Runtime

Demo mode runs at 16.9ms, well inside the 50ms real-time constraint for a 20Hz LiDAR.
Real KITTI data runs at 86.3ms on CPU due to the larger point cloud (121k vs 20k points).
Runtime optimization is planned as part of the Series 1 capstone on Day 12.

---

## What I Learned

RANSAC ground removal is sensitive to the distance threshold.
Set it too small and bumpy road surfaces leave ground points in the object cloud.
Set it too large and low vehicles get removed as ground.
Tuning requires examining actual failure cases, not just the nominal result.

DBSCAN eps must be tuned alongside voxel size - they are not independent parameters.
At voxel=0.4m, the point spacing changes enough that eps=0.6m starts merging adjacent cars.
Both parameters must be chosen together.

The voxel size trade-off is a real engineering decision with safety consequences.
This is not a tuning preference - the wrong choice creates ghost detections.

---

## Who Uses This

| Company | Team | Role |
|---------|------|------|
| Waymo   | LiDAR Perception | Perception Engineer |
| Tesla   | Autopilot Sensing | Perception Engineer |
| Cruise  | Sensing | LiDAR Systems Engineer |
| Aurora  | Perception | Perception Engineer |

This pipeline is the first layer of every production AV stack.

---

## Run It

```bash
# Clone and install
git clone https://github.com/GVK-Engine/day-001-lidar-obstacle-detection
cd day-001-lidar-obstacle-detection
pip install -r requirements.txt

# Run demo (no dataset needed)
py -3.11 my_pipeline.py

# Run voxel benchmark
py -3.11 benchmark.py

# Run on real KITTI data (update path in file first)
py -3.11 run_kitti.py
```

---

## KITTI Dataset

Free download after registration:
https://www.cvlibs.net/datasets/kitti/raw_data.php

Sequence used: 2011_09_26_drive_0001 (City, 114 frames, 0.4 GB)

Download: synced+rectified data + calibration files

---

## Project Structure

```
day-001-lidar-obstacle-detection/
├── my_pipeline.py        full pipeline with demo mode
├── benchmark.py          voxel size trade-off analysis
├── run_kitti.py          pipeline on real KITTI .bin files
├── requirements.txt      dependencies
└── README.md
```

---

## Stack

Python 3.11 | Open3D 0.19 | NumPy | SciPy | KITTI Dataset

---

## Series Progress

| Project | Title | Status |
|---------|-------|--------|
| P1.1 | LiDAR Obstacle Detection | Complete |
