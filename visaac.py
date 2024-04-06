import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rt
import os
from manopth.manolayer import ManoLayer
import torch
import random
from smplx import SMPLX, SMPL, SMPLH, MANO
import utils
from config import *

# this is to synthesis & visualize a new scene from dexgraspnet keyword and gmd scene keyword.
def vis_from_isaac(save_path: str = os.path.join(SAVE_PATH, "scenes", "scene_{:08d}".format(2)), # path to gmd rendering
        meshes = []
):

    
    # load pose dictionary
    pose_dict = np.load(os.path.join(save_path, "pose.npy"), allow_pickle = True).item()
    body_pose = pose_dict['body_pose']
    body_params = pose_dict['body_params'] # rotation vector formula
    hand_pose = pose_dict['hand_pose']
    hand_params = pose_dict['hand_params'] # rotation vector formula
    obj_pose = pose_dict['obj_pose'] # poses in form of 3D translation + quaternion rotation     
    scene_meta = pose_dict['scene_meta']
    gmd_length = scene_meta['gmd_length']
    handover_length = scene_meta['handover_length']

    # load GraspNet information
    extra_params = {}
    extra_params['use_pca'] = False
    extra_params['use_face_contour'] = True
    extra_params['flat_hand_mean'] = True
    target_obj_mesh = o3d.io.read_triangle_mesh(os.path.join(save_path, "object.obj"))
    grasp_side = scene_meta['hand_side']
    hand_mesh = o3d.io.read_triangle_mesh(os.path.join(save_path, "hand.obj"))
    hand_mesh_color = (1, 0, 0)
    hand_mesh.paint_uniform_color(hand_mesh_color) 

    # construct body model (optically smpl-h yet fails) 
    gender = 'neutral'
    model_type = 'smplx'
    if model_type == 'smpl': model = SMPL(SMPL_PATH[gender], **extra_params, ext = SMPL_PATH[gender].split('.')[-1])
    elif model_type == 'smplh':
        if gender == 'neutral': gender = 'female' # SMPLH doesn't provide available neutral model
        model = SMPLH(SMPLH_PATH[gender], **extra_params, ext = SMPLH_PATH[gender].split('.')[-1])
    elif model_type == 'smplx': model = SMPLX(SMPLX_PATH[gender], **extra_params, ext = SMPLX_PATH[gender].split('.')[-1])

    show_pos = np.array((0., 0., 0.))

    for i in range(int((hand_pose.shape[0] + 9)/10)): 
        frame_id = i * 10
        if frame_id <= gmd_length: 
            color = (0, 0.4 + 0.6 * frame_id/gmd_length, 0.6 - 0.6 * frame_id/gmd_length)
        else:
            color = ((frame_id-gmd_length)/handover_length, 1, 0)
            show_pos += np.array((0., 0., 0.4))
          # globel position

        # generate body mesh
        input_args = {
                'body_pose': torch.tensor(body_params[frame_id]).reshape(1, -1).float(),
                f'{grasp_side}_hand_pose': torch.tensor(hand_params[frame_id]).reshape(1, -1).float(),
            }
        output = model(global_orient=torch.zeros(1, 3), betas=torch.zeros(1, 10), **input_args)
        body_joints = output.joints[0].detach().cpu().numpy().squeeze()
        vertices = output.vertices[0].detach().cpu().numpy().squeeze()
        faces = model.faces.astype(np.int32)
        body_mesh = o3d.geometry.TriangleMesh()
        body_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        body_mesh.triangles = o3d.utility.Vector3iVector(faces)

        body_ground_displace = np.eye(4)
        body_ground_displace[:3, :3] = Rt.from_quat(body_pose[frame_id, 3:]).as_matrix()
        body_ground_displace[:3, 3] = body_pose[frame_id, :3]
        hand_ground_displace = np.eye(4)
        hand_ground_displace[:3, :3] = Rt.from_quat(hand_pose[frame_id, 3:]).as_matrix()
        hand_ground_displace[:3, 3] = hand_pose[frame_id, :3]
        obj_ground_displace = np.eye(4)
        obj_ground_displace[:3, :3] = Rt.from_quat(obj_pose[frame_id, 3:]).as_matrix()
        obj_ground_displace[:3, 3] = obj_pose[frame_id, :3]

        # transform the handmesh to match the full body
        body_mesh = body_mesh.paint_uniform_color(color).transform(body_ground_displace).translate(show_pos)
        hand_mesh_here = o3d.geometry.TriangleMesh(hand_mesh).transform(hand_ground_displace).translate(show_pos)
        obj_mesh_here = o3d.geometry.TriangleMesh(target_obj_mesh).transform(obj_ground_displace).translate(show_pos)
        meshes.append(body_mesh)
        meshes.append(hand_mesh_here)
        meshes.append(obj_mesh_here)
        
if __name__ == '__main__':
    meshes = []
    vis_from_isaac(meshes = meshes)
    # visualize all the recorded meshes 
    o3d.visualization.draw_geometries(meshes, mesh_show_wireframe=True, mesh_show_back_face=True)