import numpy as np
import json
from pyquaternion import Quaternion
from pytorch3d.transforms import quaternion_to_matrix
import torch
from scipy.spatial.transform import Rotation
from camera import project_3d, correct_angle, preprocess_monoloco, to_spherical
import math
import os

def quaternion_yaw(q: Quaternion, in_image_frame: bool = True) -> float:
    if in_image_frame:
        v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
        yaw = -np.arctan2(v[2], v[0])
    else:
        v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
    return float(yaw)

def local2global(traj_estimated, ego_pose):

    # Ego translation
    ego_translation = ego_pose[:, :, 4:]
    # Ego rotation
    ego_quaternion = ego_pose[:, :, :4]
    
    # Quaternion to rotation matrix
    ego_rotation_matrix = quaternion_to_matrix(ego_quaternion)
    # Reshape traj_estimated to [batch, seq_len, 3, 1] for matrix multiplication
    traj_estimated = traj_estimated.unsqueeze(-1)
    # Camera to global
    traj_estimated = torch.matmul(ego_rotation_matrix, traj_estimated)
    traj_estimated = traj_estimated.squeeze(-1) + ego_translation  # Remove the extra dimension then add

    return traj_estimated

def normalize_hwl(lab):

    AV_H = 1.72
    AV_W = 0.75
    AV_L = 0.68
    HLW_STD = 0.1

    hwl = lab[4:7]
    hwl_new = list((np.array(hwl) - np.array([AV_H, AV_W, AV_L])) / HLW_STD)
    lab_new = lab[0:4] + hwl_new + lab[7:]
    return lab_new

def extract_ground_truth(box_obj, kk, spherical=True):
    # boxes_obj: x, y, z, w, l, h, yaw

    boxes_gt = [] # 2D
    boxes_3d = []
    ys = []

    # Obtain 2D & 3D box
    boxes_gt.append(project_3d(box_obj, kk))
    boxes_3d.append(box_obj[:6])

    # Angle
    yaw = box_obj[6]
    assert - math.pi <= yaw <= math.pi
    sin, cos, _ = correct_angle(yaw, box_obj[:3])
    hwl = [box_obj[5], box_obj[3], box_obj[4]]

    # Spherical coordinates
    xyz = list(box_obj[:3])
    dd = np.linalg.norm(box_obj[:3])
    if spherical:
        rtp = to_spherical(xyz)
        loc = rtp[1:3] + xyz[2:3] + rtp[0:1]  # [theta, psi, z, r]
    else:
        loc = xyz + [dd]

    output = loc + hwl + [sin, cos, yaw]
    ys = (output)

    return boxes_gt, boxes_3d, ys

base_name = 'hard'
# path to pedestrian and ego json files
json_file_path_ped = './data/' + base_name + '/' + base_name + '_local_coordinates.json'
json_file_path_ego = './data/' + base_name + '/' + base_name + '_ego_coordinates.json'

# output dir
output_dir = './output/' + base_name + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


f = open(json_file_path_ped, 'r')
# Prepare the output dictionary
converted_data = []

for i, line in enumerate(f): # frame_data in frames:
    frame_data = json.loads(line.strip()) 
    converted_data.append(frame_data)

f.close()

f = open(json_file_path_ego, 'r')
# Prepare the output dictionary
converted_data_ego = []

for i, line in enumerate(f): # frame_data in frames:
    frame_data = json.loads(line.strip()) 
    converted_data_ego.append(frame_data)

f.close()


# make 2d array to record frame and appering id
frame_appearing_id = []
for _, frame_data in enumerate(converted_data):
    frame_num = frame_data['frame']
    coord_ls = frame_data['coordinates']
    # check if the coordinates are empty
    if len(coord_ls) == 0:
        # print("No coordinates found")
        frame_appearing_id.append([frame_num, []])
        continue

    # get the appearing id
    appearing_id = []
    for coord in coord_ls:
        appearing_id.append(coord['id'])
    
    frame_appearing_id.append([frame_num, appearing_id])

# resolution: 1280 x 720
# data format: 
# {'X':[], 'Y':[], 'kps':[], 'boxes_3d':[], 'boxes_2d':[],'ego_pose':[], 'traj_3d_ego':[], 'name':[]}
seq_len = 10
interval = 15
kk = [[700.34, 0, 629.3],
              [0, 700.345, 336.499],
              [0, 0, 1]]


X_ls = []
Y_ls = []
name_ls = []
kps_ls = []
boxes_3d_ls = []
boxes_2d_seq_ls = []
ego_pose_ls = []
traj_3d_ego_ls = []
image_path_ls = []

for i in range(len(frame_appearing_id) - (seq_len * interval)):
    
    appearing_id = frame_appearing_id[i]

    # no pedestrian
    if len(appearing_id[1]) == 0:
        continue
    
    for id in appearing_id[1]:
        # get the coordinates of the id
        id_check = True
        frame_ls = [i]
        for future_frame, future_appearing_id in frame_appearing_id[i+interval-1: i+seq_len*interval-1:interval]:
            # exist in all future frames
            if id in future_appearing_id:
                frame_ls.append(future_frame)
                continue
            else:
                id_check = False
                break
        

        if id_check:
            # get sequential keypoints
            X_seq = []
            Y_seq = []
            name_seq = []
            kps_seq = []
            boxes_3d_seq = []
            boxes_2d_seq = []
            ego_pose_seq = []
            ego_pose_check_seq = []
            traj_3d_ego_seq = []

            for frame_no in frame_ls:
                frame_data = converted_data[frame_no]
                frame_num = frame_data['frame']
                frame_name = base_name + '_' + str(frame_num) 
                coord_ls = frame_data['coordinates']


                ego_frame_data = converted_data_ego[frame_no]
                ego_pose = ego_frame_data['coordinates']

                for coord in coord_ls:
                    if coord['id'] == id:
                        kps = coord['keypoints']
                        kps = [kps[k] for k in range(len(kps)) if k % 3 != 2]
                        # 34 => 17, 2
                        kps = [kps[k:k+2] for k in range(0, len(kps), 2)]
                        kps_seq.append(kps)
                        yaw = quaternion_yaw(Quaternion(ego_pose['q4'], ego_pose['q1'], ego_pose['q2'], ego_pose['q3']))
                        boxes_3d_seq.append([coord['x'], coord['y'], coord['z'], 0, 0, 0, yaw]) # x, y, z, w, l, h, yaw
                        boxes_2d_seq.append(coord['bbox']) # xyxy
                        ego_pose_new = [ego_pose['q4'], ego_pose['q1'], ego_pose['q2'], ego_pose['q3'], ego_pose['x'], ego_pose['y'],  ego_pose['z']]
                        ego_pose_seq.append(ego_pose_new) # xyzq1q2q3q4
                        name_seq.append(frame_name)
                        


            # local to global
            boxes_3d_seq_np = np.array(boxes_3d_seq)
            ego_pose_seq_np = np.array(ego_pose_seq)
            ego_pose_check_np = np.array(ego_pose_check_seq)

            traj_3d_ego_seq = local2global(torch.tensor(boxes_3d_seq_np[:,:3]).unsqueeze(0).float(), torch.tensor(ego_pose_seq_np).unsqueeze(0).float())
            traj_3d_ego_seq = traj_3d_ego_seq.squeeze(0).numpy().tolist()
            
            # X, Y
            for time_step in range(seq_len):
                boxes_3d = boxes_3d_seq_np[time_step]
                keypoint = kps_seq[time_step]
                keypoint = np.array(keypoint)
                keypoint = keypoint.transpose()
                keypoint = keypoint.reshape(1, 2, 17)
                boxes_gt, boxes_3d, ys = extract_ground_truth(boxes_3d, kk)
                inp = preprocess_monoloco(keypoint, kk).view(-1).tolist()
                lab = normalize_hwl(ys)
                X_seq.append(inp)
                Y_seq.append(lab)

            # append to the list
            X_ls.append(X_seq)
            Y_ls.append(Y_seq)
            kps_ls.append(kps_seq)
            boxes_3d_ls.append(boxes_3d_seq)
            boxes_2d_seq_ls.append(boxes_2d_seq)
            ego_pose_ls.append(ego_pose_seq)
            traj_3d_ego_ls.append(traj_3d_ego_seq)
            name_ls.append(name_seq)


# save the data
for i in range(len(X_ls)):
    data = {'X':X_ls[i], 'Y':Y_ls[i], 'kps':kps_ls[i], 'boxes_3d':boxes_3d_ls[i], 'boxes_2d':boxes_2d_seq_ls[i], 'ego_pose':ego_pose_ls[i], 'traj_3d_ego':traj_3d_ego_ls[i], 'name':name_ls[i]}
    with open(output_dir + str(i) + '.json', 'w') as f:
        json.dump(data, f)

print("Data saved")

            



                        

                




                





            









    
    







    