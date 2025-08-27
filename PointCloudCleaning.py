"""
Point cloud cleaning (Open3D)
"""
import open3d as o3d

def getInlierOutlier(pcd, ind):
    inlier  = pcd.select_by_index(ind)
    outlier = pcd.select_by_index(ind, invert=True)
    return inlier, outlier

def voxelDown(pcd, voxel_size=0.02):
    return pcd.voxel_down_sample(voxel_size=voxel_size)

def selectEveryKPoints(pcd, K=5):
    return pcd.uniform_down_sample(every_k_points=K)

def removeStatisticalOutlier(pcd, voxel_size=0.02, nb_neighbors=20, std_ratio=2.0):
    v = voxelDown(pcd, voxel_size)
    cl, ind = v.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return getInlierOutlier(v, ind)

def removeRadiusOutlier(pcd, voxel_size=0.02, nb_points=16, radius=0.05):
    v = voxelDown(pcd, voxel_size)
    cl, ind = v.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return getInlierOutlier(v, ind)
