import numpy as np
import json
import code
import open3d as o3d
from scipy.spatial.transform import Rotation as Rt
import os
import pickle
from manopth.manolayer import ManoLayer
import torch
import torch.nn as nn
import argparse
from time import time
from datetime import datetime
import logging
import sys
import re
import random
from nosmpl.vis.vis_o3d import vis_mesh_o3d
from smplx import SMPLX, SMPL, SMPLH, MANO
import smplx
import utils
from config import *

scene_id = 1

def synthesis_from_element(dex_key = "Refrigerator_a728186f2bb912572d8564c06b061019_0.0013412556701002402index000", # refer to /share/haoran/HRI/DexGraspNet/scenes.txt
        gmd_batch = "try_03200001", gmd_scene = "sample05_rep00", # path to gmd rendering
        dex_save = True, # save obj and hand meshes from dex grasp net 
        ):

    # load GraspNet information
    dex_grasp_net_path = os.path.join("/share/haoran/HRI/handover-sim/handover/data","grasp_net/meta", "{}.npz".format(dex_key))
    dex_grasp_net_meta = dict(np.load(dex_grasp_net_path, allow_pickle = True))
    grasp_side = str(dex_grasp_net_meta['hand_side'])
    hand_beta_tensor = torch.tensor(dex_grasp_net_meta['hand_beta'], dtype=torch.float32).unsqueeze(0) # Non-zero meta may hurt the alignment
    hand_object_displace = np.eye(4)
    hand_object_displace[:3, 3] = dex_grasp_net_meta['hand_trans']
    hand_object_displace[:3, :3] = Rt.from_rotvec(dex_grasp_net_meta['hand_rot']).as_matrix()
    hand_theta_tensor = torch.tensor(dex_grasp_net_meta['hand_theta'], dtype=torch.float32).unsqueeze(0)

    # construct hand mesh
    extra_params = {}
    extra_params['use_pca'] = False
    extra_params['use_face_contour'] = True
    extra_params['flat_hand_mean'] = True
    hand_model = MANO(MANO_PATH[grasp_side], **extra_params)
    hand_output = hand_model(global_orient=torch.zeros(1, 3), betas=hand_beta_tensor, hand_pose = hand_theta_tensor, axis = 1)
    hand_transformed_joints = hand_output.joints[0].detach().cpu().numpy().squeeze()
    hand_transformed_vertices = hand_output.vertices[0].detach().cpu().numpy().squeeze()
    hand_faces = hand_model.faces.astype(np.int32)
    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.vertices = o3d.utility.Vector3dVector(hand_transformed_vertices)
    hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)

    # construct body model (optically smpl-h yet fails)
    gender = 'neutral'
    model_type = 'smplx'
    if model_type == 'smpl': model = SMPL(SMPL_PATH[gender], **extra_params, ext = SMPL_PATH[gender].split('.')[-1])
    elif model_type == 'smplh':
        if gender == 'neutral': gender = 'female' # SMPLH doesn't provide available neutral model
        model = SMPLH(SMPLH_PATH[gender], **extra_params, ext = SMPLH_PATH[gender].split('.')[-1])
    elif model_type == 'smplx': model = SMPLX(SMPLX_PATH[gender], **extra_params, ext = SMPLX_PATH[gender].split('.')[-1])

    # load GMD scene information
    data_path = os.path.join("/share1/haoran/mobile/guided-motion-diffusion/save/unet_adazero_xl_x0_abs_proj10_fp16_clipwd_224", gmd_batch)
    smpl_params_path = os.path.join(data_path, f"{gmd_scene}_smpl_params.npy")
    smpl_params = np.load(smpl_params_path, allow_pickle = True)
    motion = smpl_params.item()['motion']
    motion = np.concatenate((Rt.from_matrix(utils.rotation_6d_to_matrix(torch.from_numpy(motion[:-1, :, :].transpose(2, 0, 1))).numpy()
                                            .reshape(-1, 3, 3)).as_rotvec().reshape(-1, 24, 3), motion[[-1], :3, :].transpose(2, 0, 1)), axis = 1)
    gmd_length = motion.shape[0]

    # load handover scene body parameter
    handover_length = 21
    init_motion = motion[-1].copy()
    end_motion = init_motion.copy()
    var_params = [ # 6DoFs of handover-ik
        random.gauss(0, 0.1), # wrist displacement of x axis (left) - 0.1
        random.gauss(0, 0.05), # wrist displacement of y axis (up) - 0.05
        random.gauss(0, 0.05), # wrist displacement of z axis (forward) - 0.05
        random.gauss(0, 0.5), 
        random.gauss(0, 0.5), 
        random.gauss(0, 0.5), # wrist rotation dimensions - 0.5
        ]
    end_motion[JOINT_DICT['right_shoulder']] = Rt.from_euler('xyz', (+ 3.247 * var_params[1] - 1.81 * var_params[2] -np.pi / 12, np.pi / 3 + var_params[0] * 1.507 + 3.79 * var_params[2] , -np.pi / 6)).as_rotvec()
    end_motion[JOINT_DICT['right_elbow']] = Rt.from_euler('xyz', (-np.pi / 4, np.pi / 3 + var_params[0] * 1.507 - 6.8 * var_params[2], 3.247 * var_params[1] - 1.81 * var_params[2])).as_rotvec()
    end_motion[JOINT_DICT['right_wrist']] = Rt.from_euler('xyz', (-np.pi / 3,  2 * var_params[2], -np.pi / 6 - var_params[0] * 1.507  +  3.29 * var_params[2])).as_rotvec()
    motion = np.concatenate((motion, np.linspace(init_motion, end_motion, handover_length + 1)[1:]))  # 21 extra frames for handover
    thetas = motion[:, :-1, :]

    # pose information for Isaac Gym 
    with open(os.path.join(SAVE_PATH, "log.txt"), "a") as f:
        f.write("{}&{}&{}&{}\n".format(scene_id, gmd_batch, gmd_scene, dex_key))
    save_path = os.path.join(SAVE_PATH, "scenes", "scene_{:08d}".format(scene_id))
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    pose_dict = {}
    pose_dict['body_pose'] = np.zeros((motion.shape[0], 7))
    pose_dict['body_params'] = np.zeros((motion.shape[0], 21, 3)) # rotation vector formula
    pose_dict['hand_pose'] = np.zeros((motion.shape[0], 7))
    pose_dict['hand_params'] = np.zeros((motion.shape[0], 15, 3)) # rotation vector formula
    pose_dict['obj_pose'] = np.zeros((motion.shape[0], 7)) # poses in form of 3D translation + quaternion rotation     
    pose_dict['scene_meta'] = {
        'dex_key': dex_key,
        'gmd_batch': gmd_batch, 
        'gmd_scene': gmd_scene,
        'hand_side': "right",
        'gmd_length': gmd_length,
        'handover_length': handover_length} # meta information of the scene

    # loop through all the frames
    for i in range(motion.shape[0]): 
        frame_id = i
        theta = thetas[frame_id]

        # generate body mesh
        input_args = {
                'body_pose': torch.from_numpy(theta[1:-2].reshape(1, -1)).float(),
                f'{grasp_side}_hand_pose': hand_theta_tensor,
            }
        output = model(global_orient=torch.zeros(1, 3), betas=torch.zeros(1, 10), **input_args)
        body_joints = output.joints[0].detach().cpu().numpy().squeeze()

        # the transformation for models
        body_ground_displace = np.eye(4)
        body_ground_displace[:3, :3] = Rt.from_rotvec(theta[0]).as_matrix() # the first translation root at the origin
        body_ground_displace[:3, 3] = motion[frame_id, -1, :] - motion[0, -1, :]

        # extract 4 joints from mano and smpl
        selected_hand_joints = hand_transformed_joints[HAND_KEY_JOINT[grasp_side]] # (4, 3)  palm, thumb, middle, pinky
        augmented_hand_joints =  np.concatenate((selected_hand_joints, np.ones((4, 1))), axis = 1).T 
        selected_body_joints = body_joints[BODY_KEY_JOINT[grasp_side]] # (4, 3)  palm, thumb, middle, pinky
        augmented_body_joints =  np.concatenate((selected_body_joints, np.ones((4, 1))), axis = 1).T
        hand_body_displace = augmented_body_joints @ np.linalg.inv(augmented_hand_joints)
        
        # calculate the overall displacement
        hand_ground_displace = body_ground_displace @ hand_body_displace
        hand_ground_displace[:3, :3] = Rt.from_matrix(hand_ground_displace[:3, :3]).as_matrix()      
        obj_ground_displace = hand_ground_displace @ np.linalg.inv(hand_object_displace)

        # get the displacement of bofy and obj
        pose_dict['body_params'][frame_id] = theta[1:-2]
        pose_dict['hand_params'][frame_id] = hand_theta_tensor.numpy().reshape(-1, 3)
        pose_dict['body_pose'][frame_id, :3] = body_ground_displace[:3, 3]
        pose_dict['body_pose'][frame_id, 3:] = Rt.from_matrix(body_ground_displace[:3, :3]).as_quat()
        pose_dict['hand_pose'][frame_id, :3] = hand_ground_displace[:3, 3]
        pose_dict['hand_pose'][frame_id, 3:] = Rt.from_matrix(hand_ground_displace[:3, :3]).as_quat()
        pose_dict['obj_pose'][frame_id, :3] = obj_ground_displace[:3, 3]
        pose_dict['obj_pose'][frame_id, 3:] = Rt.from_matrix(obj_ground_displace[:3, :3]).as_quat()
      
    # save the body and obj parameters
    if dex_save:
        target_obj_mesh = o3d.io.read_triangle_mesh(os.path.join("/share/haoran/HRI/handover-sim/handover/data", "assets", dex_key.split("index")[0], "model_normalized_convex.obj"))
        o3d.io.write_triangle_mesh(os.path.join(save_path, "object.obj"),target_obj_mesh)
        hand_transformed_vertices = hand_output.vertices[0].detach().cpu().numpy().squeeze()
        hand_faces = hand_model.faces.astype(np.int32)
        hand_mesh = o3d.geometry.TriangleMesh()
        hand_mesh.vertices = o3d.utility.Vector3dVector(hand_transformed_vertices)
        hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
        o3d.io.write_triangle_mesh(os.path.join(save_path, "hand.obj"),hand_mesh)
    np.save(os.path.join(save_path, "pose"), pose_dict)

if __name__ == '__main__':
    for i in range(10):
        scene_id = i
        synthesis_from_element(gmd_scene = f"sample0{scene_id}_rep00")