import os


MODEL_PATH = '/share1/haoran/czq/SMPL/body-model-visualizer/data/body_models'
MANO_PATH = {'right': os.path.join(MODEL_PATH, "mano/MANO_RIGHT.pkl"),
             'left': os.path.join(MODEL_PATH, "mano/MANO_LEFT.pkl")}
SMPL_PATH = {'female': os.path.join(MODEL_PATH, "smpl/SMPL_FEMALE.pkl"),
             'male': os.path.join(MODEL_PATH, "smpl/SMPL_MALE.pkl"),
             'neutral': os.path.join(MODEL_PATH, "smpl/SMPL_NEUTRAL.pkl")}
SMPLH_PATH = {'female': os.path.join(MODEL_PATH, "smplh/SMPLH_FEMALE.pkl"),
             'male': os.path.join(MODEL_PATH, "smplh/SMPLH_MALE.pkl"),
             'neutral': None}
SMPLX_PATH = {'female': os.path.join(MODEL_PATH, "smplx/SMPLX_FEMALE.pkl"),
             'male': os.path.join(MODEL_PATH, "smplx/SMPLX_MALE.pkl"),
             'neutral': os.path.join(MODEL_PATH, "smplx/SMPLX_NEUTRAL.pkl")}
HAND_KEY_JOINT = {'right': [0, 15, 6, 9], 'left': [0, 15, 6, 9]}
BODY_KEY_JOINT = {'right': [21, 54, 45, 48], 'left': [20, 39, 30, 33]}
JOINT_DICT = {'right_shoulder': 17, 'right_elbow': 19, 'right_wrist': 21}
SAVE_PATH = "/share1/haoran/mobile/Mobile_Handover/data/gmd"