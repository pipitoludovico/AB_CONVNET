import open3d as o3d
import numpy as np

# Define the coordinates of the atoms
coordinates = np.array([
    [12.909, 11.491, -30.313],
    [12.805, 11.889, -28.871],
    [14.107, 12.270, -28.244],
    [15.363, 11.628, -28.758],
    [15.296, 11.341, -30.233],
    [16.530, 10.587, -30.622]
])

# Define the voxel size
voxel_size = 1.0

# Create a PointCloud from the atomic coordinates
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(coordinates)

# Voxelize the PointCloud
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=voxel_size)

# Visualize the VoxelGrid
o3d.visualization.draw_geometries([voxel_grid])
