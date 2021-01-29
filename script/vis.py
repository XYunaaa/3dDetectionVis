import open3d as o3d
import numpy as np
import colorsys, random
from get_aimed_bbox import get_bbox_label,boxes_to_corners_3d,rots_to_corners_3d
#from get_aimed_bbox_torch import boxes_to_corners_3d_,rots_to_corners_3d_
# show pc with aimed color(label/aimed rgb)
from LineMesh import LineMesh
import json
import cv2,os
import script.painted_pointcloud.painted_pc_rs128 as Painted

class Plot:

    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb,savefig=False,name=None):

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        #pc_normals = np.zeros((pc_xyzrgb.shape[0],3)) + 10
        #pc.normals = open3d.utility.Vector3dVector(pc_normals)
        if pc_xyzrgb.shape[1] == 3:
            o3d.draw_geometries([pc])
            return 0
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])
        if not savefig:
            o3d.visualization.draw_geometries([pc])
            return 0
        else:

            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis_control = vis.get_view_control()
            param = o3d.io.read_pinhole_camera_parameters('para.json')
            vis.add_geometry(pc)
            vis_control.convert_from_pinhole_camera_parameters(param)
            ''' for open3d >=0.10.0
            front = [ -0.9917820008427326, -0.06773734865811093, 0.10853623542925953 ]
            lookat = [ 39.381500244140625, 1.1175003051757812, 0.69850003719329834 ]
            up = [ 0.10261310693082715, 0.085530815590998441, 0.99103735039116536 ]
            zoom = 0.379
            vis_control.set_front(front)
            vis_control.set_lookat(lookat)
            vis_control.set_up(up)
            vis_control.set_zoom(zoom)
            '''
            #vis_control.rotate(x=1,y=1)
            #vis_control.set_zoom(0.75)
            #vis_control.translate(x=-1,y=-50)
            #param = vis_control.convert_to_pinhole_camera_parameters()
            #o3d.io.write_pinhole_camera_parameters('para.json',param)
            #
            #vis.update_geometry()
            #vis_control.scale(-8)
            vis.poll_events()
            vis.update_renderer()
            vis.run()
            vis.capture_screen_image(name)
            #
            vis.destroy_window()
            vis.close()
            #image = cv2.imread('test.png')
            #image = image[200:-200,200:-500,:]
            #cv2.imwrite(name,image)
            return 0

    @staticmethod
    def save_view_point(pc,filename):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pc)
        vis_control = vis.get_view_control()
        vis.run()
        param = vis_control.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(filename, param)
        vis_control.scale(-8)
        vis.destroy_window()

    @staticmethod
    def add_bbox(bboxes,class_name,color_list,radius=0.04):
        lines_list = []
        corners_3d_list = []
        idx = 0
        for bbox in bboxes:
            c = class_name[idx]
            color = color_list[c]
            ## draw each box's corner and line
            bbox = np.array(bbox)
            lines = [[0, 1], [1, 2], [2,3],[3,0],[0, 4], [1, 5], [2,6],[3,7],[4,5], [5, 6], [6,7],[7,4]]
            colors = [color for i in range(len(lines))]
            colors_corner = [color for i in range(len(bbox))]

            ## define 8 corners of bboxes3d ##
            corners_3d_bbox = o3d.geometry.PointCloud()
            corners_3d_bbox.points = o3d.utility.Vector3dVector(bbox)
            corners_3d_bbox.colors = o3d.utility.Vector3dVector(colors_corner)
            ## define bbox3d's lines ##
            lines_bbox = LineMesh(bbox,lines,colors,radius=radius)
            '''
            Open3d has bug; can't change line width ;use LineMesh Class change line width
            lines_bbox = o3d.geometry.LineSet()
            lines_bbox.lines = o3d.utility.Vector2iVector(lines)
            lines_bbox.colors = o3d.utility.Vector3dVector(colors)
            lines_bbox.points = o3d.utility.Vector3dVector(bbox)
            '''
            corners_3d_list.append(corners_3d_bbox)
            lines_list.append(lines_bbox)
            idx+=1

        return lines_list, corners_3d_list

    @staticmethod
    def add_rotation(rot,class_name,color_list,radius=0.06):

        lines_list = []
        corners_3d_list = []
        idx = 0
        for r in rot:
            ## draw each rotation's corner and line
            c = class_name[idx]
            color = color_list[c]
            r = np.array(r)
            lines = [[0, 1],[1,0]]
            colors = [color for i in range(len(lines))]

            ##定义朝向的起始点（bbox的中心） ##
            corners_3d_bbox = o3d.geometry.PointCloud()
            r_tmp = r[0,:].reshape((1,-1))
            colors_corner = [color for i in range(len(r_tmp))]

            corners_3d_bbox.points = o3d.utility.Vector3dVector(r_tmp)
            corners_3d_bbox.colors = o3d.utility.Vector3dVector(colors_corner)
            ## define bbox3d's lines ##
            lines_bbox = LineMesh(r,lines,colors,radius=radius)
            corners_3d_list.append(corners_3d_bbox)
            lines_list.append(lines_bbox)
            idx += 1

        return lines_list, corners_3d_list
    @staticmethod
    def draw_pc_bbox(pc_xyzrgb,bboxes,class_name,color_list,savefig=False,name=None):

        pc = o3d.geometry.PointCloud()
        ## pointcloud
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        # 车辆 蓝色 骑行者 红色 行人 绿色
        #color_list = [[0,0,1],[1,0,0],[0,1,0]]
        class_name[np.where(class_name=="vehicle")[0]] = 0
        class_name[np.where(class_name == "cyclist")[0]] = 1
        class_name[np.where(class_name == "ped")[0]] = 2
        class_name = class_name.astype(np.int32)
        ## bbox
        if len(bboxes)!=1:

            #gt = bboxes[0]
            pre = bboxes[0]
            #lines_list_gt, corners_3d_list_gt = Plot.add_bbox(gt,color=[0/255,0/255,255/255])
            lines_list_pre, corners_3d_list_pre = Plot.add_bbox(pre,class_name,color_list)

            #gt_rot = bboxes[2]
            pre_rot = bboxes[1]
            #lines_list_gt_r, corners_3d_list_gt_r = Plot.add_rotation(gt_rot,color=[0/255,0/255,255/255])
            lines_list_pre_r, corners_3d_list_pre_r = Plot.add_rotation(pre_rot,class_name,color_list)
        else:
            lines_list, corners_3d_list = Plot.add_bbox(bboxes,class_name,color_list)

        #Plot.save_view_point(pc, 'para1.json')

        if pc_xyzrgb.shape[1] == 3:
            o3d.draw_geometries([pc])
            return 0
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])


        if not savefig:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis_control = vis.get_view_control()
            vis.add_geometry(pc)
            if len(bboxes)!=1:

                for i in range(len(corners_3d_list_pre)):
                    vis.add_geometry(corners_3d_list_pre[i])
                    lines_list_pre[i].add_line(vis)

                for i in range(len(corners_3d_list_pre_r)):
                    vis.add_geometry(corners_3d_list_pre_r[i])
                    lines_list_pre_r[i].add_line(vis)
            else:
                for i in range(len(corners_3d_list)):
                    vis.add_geometry(corners_3d_list[i])
                    lines_list[i].add_line(vis)

            param = o3d.io.read_pinhole_camera_parameters('para_128.json')
            vis_control.convert_from_pinhole_camera_parameters(param)
            vis.poll_events()
            vis.update_renderer()
            vis.run()
            return 0
        else:

            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis_control = vis.get_view_control()
            vis.add_geometry(pc)

            if len(bboxes) != 1:
                for i in range(len(corners_3d_list_gt)):
                    vis.add_geometry(corners_3d_list_gt[i])
                    lines_list_gt[i].add_line(vis)
                for i in range(len(corners_3d_list_pre)):
                    vis.add_geometry(corners_3d_list_pre[i])
                    lines_list_pre[i].add_line(vis)
            else:
                for i in range(len(corners_3d_list)):
                    vis.add_geometry(corners_3d_list[i])
                    lines_list[i].add_line(vis)

            param = o3d.io.read_pinhole_camera_parameters('para_128.json')
            vis_control.convert_from_pinhole_camera_parameters(param)
            vis.poll_events()
            vis.update_renderer()
            vis.run()
            vis.capture_screen_image(name)
            vis.destroy_window()
            vis.close()
            return 0


def get_predication_json(result_path,id):
    with open(result_path, 'r') as f:
        result = json.load(f)
        for res in result:
            boxes_lidar = []
            rotation_y = []
            class_name = []
            if res[0]['frameIdx'] ==id:
                for r in res:
                    res_tmp = [r['x'],r['y'],r['z'],r['width'],r['height'],r['length'],r['rotationYaw']]
                    rot_tmp = r['rotationYaw']
                    boxes_lidar.append(res_tmp)
                    rotation_y.append(rot_tmp)
                    class_name.append(r['class'])

                res = np.array(boxes_lidar).reshape((-1,7))
                corner3d = boxes_to_corners_3d(res)
                rot3d = rots_to_corners_3d(res)
                class_name = np.array(class_name)
                return corner3d,rot3d,class_name

lidar_path = '../data/ng_velodyne/'
img_path = '../data/image/'
calib_path = '../data/calib/'
res_path = '../data/RES.json'
AIMED_ID = '000000' # 备选帧000000 000080 000101 000110 000170 000418 000433 000556
pre, rot_pre,class_name = get_predication_json(res_path,int(AIMED_ID))
lidar_id_path = lidar_path + AIMED_ID + '.npy'
pc = np.load(lidar_id_path)  # 非地面原始点云
pc = pc[:,:3]
img_id_path = img_path + AIMED_ID + '.png'
img = cv2.imread(img_id_path)
intrinsic_matrix = np.loadtxt(os.path.join(calib_path, 'intrinsic'))
distortion = np.loadtxt(os.path.join(calib_path, 'distortion'))
extrinsic_matrix = np.loadtxt(os.path.join(calib_path, 'extrinsic'))
# 消除图像distortion
img = cv2.undistort(img, intrinsic_matrix, distortion)
pc_color = Painted.painted_pc(pc, img, extrinsic_matrix, intrinsic_matrix)
pc_color[:,[0,1,2,3,4,5]] = pc_color[:,[0,1,2,5,4,3]]
pre[:,:,[0,1,2]] = pre[:,:,[1,0,2]]
rot_pre[:,:,[0,1]] = rot_pre[:,:,[1,0]]
color_list = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]  #车辆 蓝色 骑行者 红色 行人 绿色
Plot.draw_pc_bbox(pc_color, [pre,rot_pre],class_name,color_list, savefig=False)
