
import numpy as np
import pickle as pickle

# show pc with aimed color(label/aimed rgb)

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    #points, is_numpy = check_numpy_to_torch(points)
    #angle, _ = check_numpy_to_torch(angle)

    cosa = np.cos(angle).reshape(-1, 1)
    sina = np.sin(angle).reshape(-1, 1)
    zeros = np.zeros((points.shape[0], 1))
    ones = np.ones((points.shape[0], 1))
    rot_matrix = np.concatenate((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), axis=1).reshape(-1, 3, 3).astype(float)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """

    template =np.array([[1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]])/ 2

    #corners3d = boxes3d[:,3:6].reshape((-1,1,3))
    corners3d = boxes3d[:, None, 3:6].repeat(8,axis=1) * template[None, :, :]
    #corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.reshape(-1, 8, 3), boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d +=  boxes3d[:, None, 0:3]
    return corners3d

def rots_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    """
    #boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    rot_corner = np.full((boxes3d.shape[0],3),0.0)
    rot_corner[:,0] = boxes3d[:,3] * 0.6
    rot_corner[:,1] = boxes3d[:,3] * 0.6

    rot_corner = rot_corner.reshape((-1, 1, 3)).repeat(2, axis=1)

    angle = boxes3d[:,6]
    #angle, _ = check_numpy_to_torch(angle)
    angle = angle.reshape((-1))
    angle = angle.reshape((-1, 1)).repeat(2, axis=1)
    cosa = np.cos(angle) # N,1
    sina = np.sin(angle) # N,1
    rot_corner[:, :, 0] = rot_corner[:,:,0]*cosa
    rot_corner[:, :,1] = rot_corner[:, :,1]*sina

    rot_corner[:,0,:] = boxes3d[:,0:3]
    rot_corner[:,1, :] = boxes3d[:, 0:3] + rot_corner[:,1, :]
    return rot_corner

def rots_to_corners_3d_json(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    """
    #boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    rot_corner = np.full((boxes3d.shape[0],3),0.0)
    rot_corner[:,0] = boxes3d[:,3] * 0.6
    rot_corner[:,1] = boxes3d[:,3] * 0.6

    rot_corner = rot_corner[:,None,:].repeat(1, 2, 1) # N,2,3

    angle = boxes3d[:,6]
    #angle, _ = check_numpy_to_torch(angle)
    angle = angle.reshape((-1))
    angle = angle[:,None].repeat(1,2)
    cosa = np.cos(angle) # N,1
    sina = np.sin(angle) # N,1

    rot_corner[:,:,0] = rot_corner[:,:,0].mul(cosa)
    rot_corner[:, :,1] = rot_corner[:,:,1].mul(sina)

    rot_corner[:,0,:] = boxes3d[:,0:3]
    rot_corner[:,1, :] = boxes3d[:, 0:3] + rot_corner[:,1, :]
    return rot_corner

def get_aimed_bbox(path,id):

    result = pickle.load(open(path,'rb'))
    res = result[id]['boxes_lidar']
    frame_id = result[id]['frame_id']
    score = result[id]['score']
    index = np.where(score>0.3)
    res = res[index] #[N,7]
    corner3d = boxes_to_corners_3d(res)
    corner3d[:,:,2] += 0.5
    return corner3d, frame_id

def get_bbox_label(path):

    result = pickle.load(open(path, 'rb'))
    count = len(result)
    res_dict = {}
    rot_dict = {}
    for i in range(count):
        frame_id = result[i]['point_cloud']['lidar_idx']
        res = result[i]['annos']
        res_bbox = res['gt_boxes_lidar']
        corner3d = boxes_to_corners_3d(res_bbox)
        rot_corner3d = rots_to_corners_3d(res_bbox)
        res_dict.update({frame_id:corner3d})
        rot_dict.update({frame_id:rot_corner3d})

    return res_dict,rot_dict


