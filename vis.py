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
def vis_from_element(dex_key = "Refrigerator_a728186f2bb912572d8564c06b061019_0.0013412556701002402index000", # refer to /share/haoran/HRI/DexGraspNet/scenes.txt
        gmd_batch = "try_03210001", gmd_scene = "sample04_rep00", # path to gmd rendering
        meshes = []
        ):

    meshes = meshes

    # load GraspNet information
    dex_grasp_net_path = os.path.join("/share/haoran/HRI/handover-sim/handover/data","grasp_net/meta", "{}.npz".format(dex_key))
    dex_grasp_net_meta = dict(np.load(dex_grasp_net_path, allow_pickle = True))
    target_obj_mesh = o3d.io.read_triangle_mesh(os.path.join("/share/haoran/HRI/handover-sim/handover/data", "assets", dex_key.split("index")[0], "model_normalized_convex.obj"))
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

    show_pos = np.array((0., 0., 0.))

    for i in range(int((motion.shape[0] + 9)/10)): 
        frame_id = i * 10
        if frame_id <= gmd_length: 
            color = (0, 0.4 + 0.6 * frame_id/gmd_length, 0.6 - 0.6 * frame_id/gmd_length)
        else:
            color = ((frame_id-gmd_length)/handover_length, 1, 0)
            show_pos += np.array((0., 0., 0.4))
          # globel position
        theta = thetas[frame_id]

        # generate body mesh
        input_args = {
                'body_pose': torch.from_numpy(theta[1:-2].reshape(1, -1)).float(),
                f'{grasp_side}_hand_pose': hand_theta_tensor,
            }
        output = model(global_orient=torch.zeros(1, 3), betas=torch.zeros(1, 10), **input_args)
        body_joints = output.joints[0].detach().cpu().numpy().squeeze()
        vertices = output.vertices[0].detach().cpu().numpy().squeeze()
        faces = model.faces.astype(np.int32)
        body_mesh = o3d.geometry.TriangleMesh()
        body_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        body_mesh.triangles = o3d.utility.Vector3iVector(faces)

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

        # transform the handmesh to match the full body
        body_mesh = body_mesh.paint_uniform_color(color).transform(body_ground_displace).translate(show_pos)
        hand_mesh_here = o3d.geometry.TriangleMesh(hand_mesh).transform(hand_ground_displace).translate(show_pos)
        obj_mesh_here = o3d.geometry.TriangleMesh(target_obj_mesh).transform(obj_ground_displace).translate(show_pos)
        meshes.append(body_mesh)
        meshes.append(hand_mesh_here)
        meshes.append(obj_mesh_here)
        
if __name__ == '__main__':
    MODE = "test" # test: specific obj, grasp, etc/ display: random obj, grasp, etc
    with open("/share/haoran/HRI/DexGraspNet/scenes.txt", "r") as f:
        if MODE in ['test']: obj_title = "Refrigerator_a728186f2bb912572d8564c06b061019_0.0013412556701002402"
        elif MODE in ['display']: obj_title = random.choice(f.readlines()).split(".np")[0]
    if MODE in ['test']: obj_title = 0   
    elif MODE in ['display']: grasp_index = random.randrange(128)
    print(obj_title, grasp_index)
    grasp_index = 0
    meshes = []
    vis_from_element("{}index{:03d}".format(obj_title, grasp_index), "try_03200001", "sample05_rep00", meshes = meshes)
    # visualize all the recorded meshes 
    o3d.visualization.draw_geometries(meshes, mesh_show_wireframe=True, mesh_show_back_face=True)