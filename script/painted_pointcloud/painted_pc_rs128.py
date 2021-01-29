# coding=utf-8
from __future__ import print_function, division, absolute_import
import numpy.linalg as LA
import numpy as np
SCALE_FACTOR = 1


def del_nan(a):
    idx = np.argwhere(np.isnan(a[:,0]))
    return np.delete(a,idx,axis=0)

def undistort_projection(points, intrinsic_matrix, extrinsic_matrix):
    points = np.column_stack([points, np.ones_like(points[:, 0])])

    # 外参矩阵
    points = np.matmul(extrinsic_matrix, points.T, )

    # 内参矩阵
    points = np.matmul(intrinsic_matrix, points[:3, :], ).T

    # 深度归一化
    points[:, :2] /= points[:, 2].reshape(-1, 1)

    return points

def back_projection(points, intrinsic_matrix, extrinsic_matrix):
    # 还原深度
    points[:, :2] *= points[:, 2].reshape(-1, 1)

    # 还原相平面相机坐标
    points[:, :3] = np.matmul(LA.inv(intrinsic_matrix), points[:, :3].T).T

    # 还原世界坐标
    # 旋转平移矩阵
    R, T = extrinsic_matrix[:3, :3], extrinsic_matrix[:3, 3]
    points[:, :3] = np.matmul(LA.inv(R), points[:, :3].T - T.reshape(-1, 1)).T

    return points

def painted_pc(pc, img, extrinsic_matrix, intrinsic_matrix):
    # 投影验证
    projection_points = undistort_projection(pc[:, :3], intrinsic_matrix, extrinsic_matrix)

    # 裁切到图像平面
    projection_points = np.column_stack([np.squeeze(projection_points), pc[:, 3:]])
    img_shape = img.shape
    #after lidar to img,filtering points in img's range ;
    val_flag_1 = np.logical_and(projection_points[:, 0] >= 0, projection_points[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(projection_points[:, 1] >= 0, projection_points[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, projection_points[:, 2] >= 0)
    projection_points = projection_points[pts_valid_flag]

    pts_img  = projection_points
    pts_img[:, [0, 1]] = pts_img[:, [1, 0]]  # height,width to width height[1242,375] to [375,1242]
    pts_img = pts_img.astype(int)
    row = pts_img[:, 0]
    col = pts_img[:, 1]
    img = img.astype(int)
    pc_color = img[row, col, :]  # [N,3] b g r
    #pc_color[:, [0, 1, 2]] = pc_color[:, [2, 1, 0]]
    pc_color = np.hstack(
        (pc[pts_valid_flag], pc_color))  # [N,6] x,y,z,b,g,r
    return pc_color


