import datetime
import os
from code.utils import create_output_dir, create_dataset, load_config, joint2traj, recover_traj, loc2traj, batch_process_coords
from code.models import create_loc_model, create_traj_model
import sys
sys.path.append('..')
from code.train.losses import compute_ADE_FDE
from code.data import KeypointsDataset
import torch
import numpy as np
import random
import json
from torch.utils.data import DataLoader
from itertools import chain
import time
import copy
from collections import defaultdict

class Evaluator:
    def __init__(self): 

        self.eval_mode = "traj_pred"
        self.r_seed = 42
        self.bs = 1
        self.obs = 10
        self.pred = 10
        self.loc_cfg_path = "code/configs/localization.yaml"
        self.traj_cfg_path = "code/configs/traj_pred.yaml"
        self.joints_folder = "code/real_time_data/"
        self.load_loc = "checkpoints/loc/best_loc_model.pth"
        self.load_traj = "checkpoints/traj_pred/best_traj_model.pth"

        # select device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print('Device: ', self.device)
        
        # random seed
        random.seed(self.r_seed)
        torch.manual_seed(self.r_seed)
        np.random.seed(self.r_seed)
        if use_cuda:
            torch.cuda.manual_seed(self.r_seed)

        # load config
        if self.loc_cfg_path:
            self.loc_config = load_config(self.loc_cfg_path)
            if torch.cuda.is_available():
                self.loc_config["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
            else:
                self.loc_config["DEVICE"] = "cpu"
        if self.traj_cfg_path:
            self.traj_config = load_config(self.traj_cfg_path)
            if torch.cuda.is_available():
                self.traj_config["DEVICE"] = f"cuda:{torch.cuda.current_device()}"
            else:
                self.traj_config["DEVICE"] = "cpu"
        # create models
        if self.eval_mode in ["loc", "traj_pred"]:
            self.loc_model = create_loc_model(self.loc_config)
            if self.load_loc != "":
                self.loc_model.load_state_dict(torch.load(self.load_loc))
            print(">>> Localization model params: {:.3f}M".format(sum(p.numel() for p in self.loc_model.parameters()) / 1000000.0))

        if self.eval_mode in ["traj_pred"]:
            self.traj_model = create_traj_model(self.traj_config)
            if self.load_traj != "":
                self.traj_model.load_state_dict(torch.load(self.load_traj))
            print(">>> Trajectory prediction model params: {:.3f}M".format(sum(p.numel() for p in self.traj_model.parameters()) / 1000000.0))
            self.total_parameters = sum(p.numel() for p in self.loc_model.parameters()) + sum(p.numel() for p in self.traj_model.parameters())
            print(">>> Total params: {:.3f}M".format(self.total_parameters / 1000000.0))


    def evaluate_traj_pred(self, debug, X, Y, names, kps, boxes_3d, boxes_2d, K, ego_pose, camera_pose, traj_3d_ego, image_path):
        # Dataloader
        print(">>> creating dataloaders")
        self.dic_jo = create_dataset(debug, X, Y, names, kps, boxes_3d, boxes_2d, K, ego_pose, camera_pose, traj_3d_ego, image_path)
        self.dataloaders = {phase: DataLoader(KeypointsDataset(self.dic_jo, phase=phase),
                                            batch_size=self.bs, shuffle=False) for phase in ['test']} #dict to store dataloaders

        self.loc_model.eval()
        self.traj_model.eval()

        # output lists
        batch_id = 0
        ade_batch = 0 
        fde_batch = 0
        obs_pred_ls = []
        gt_traj_ls = []
        total_samples = 0

        with torch.no_grad():

            for inputs, labels, _, _, ego_pose, camera_pose, traj_3d_ego, _, _ in self.dataloaders['test']:
                
                labels = labels.to(self.device)
                batch_size, seq_length, _ = inputs.size()
                labels = labels.view(batch_size * seq_length, -1) 

                scene_train_real_ped, scene_train_mask, padding_mask = joint2traj(inputs)

                scene_train_real_ped = scene_train_real_ped.to(self.traj_config["DEVICE"])
                scene_train_mask = scene_train_mask.to(self.traj_config["DEVICE"])
                padding_mask = padding_mask.to(self.traj_config["DEVICE"])
                scene_train_real_ped = scene_train_real_ped[:,0,:,:,:]
                scene_train_mask = scene_train_mask[:,0,:,:]

                # Testing: only input observation
                scene_train_real_ped_obs = scene_train_real_ped[:,:self.obs,:,:]
                outputs = self.loc_model(scene_train_real_ped, padding_mask) 

                scene_train_real_ped_obs = scene_train_real_ped[:,:self.obs,:,:]
                padding_mask[:,self.obs:] = True
                outputs = self.loc_model(scene_train_real_ped_obs, padding_mask)
                
                # traj
                traj_estimated_ls = recover_traj(outputs, ego_pose, camera_pose)
                
                scene_train_real_ped, scene_train_mask, padding_mask = loc2traj(traj_estimated_ls)
                scene_train_real_ped_gt, scene_train_mask_gt, padding_mask_gt = loc2traj(traj_3d_ego)

                in_joints, in_masks, out_joints, out_masks, padding_mask, _ = batch_process_coords(scene_train_real_ped, scene_train_mask, padding_mask, self.traj_config, training=False)
                in_joints_gt, in_masks_gt, out_joints_gt, out_masks_gt, padding_mask_gt, _ = batch_process_coords(scene_train_real_ped_gt, scene_train_mask_gt, padding_mask_gt, self.traj_config, training=False)
                
                padding_mask = padding_mask.to(self.traj_config["DEVICE"])
                pred_joints = self.traj_model(in_joints, padding_mask)

                pred_joints = pred_joints[:,-self.pred:]
                pred_joints = pred_joints.cpu()
                pred_joints = pred_joints + scene_train_real_ped[:,0:1,(self.obs-1):self.obs, 0, 0:2]
                out_joints = out_joints_gt.cpu() 

                out_joints = out_joints + scene_train_real_ped_gt[:,0:1,(self.obs-1):self.obs, 0, :]
                pred_joints = pred_joints.reshape(out_joints.size(0), self.pred, 1, 2)  

                # obs + pred
                # concatenate the observed and predicted trajectories
                pred_outputs = pred_joints.clone()
                # to numpy
                pred_outputs = pred_outputs.cpu().numpy()
                pred_outputs = pred_outputs[0,:,0,:]
                obs_pred = np.concatenate((traj_estimated_ls[0,:self.obs,:2], pred_outputs), axis=0)

                obs_pred_ls.append(obs_pred)
                gt_traj_ls.append(traj_3d_ego.cpu().numpy()[0,:,:2])
                
                if debug:
                    # Debug information
                    print(f"Debug - out_joints shape: {out_joints.shape}")
                    print(f"Debug - pred_joints shape: {pred_joints.shape}")
                    print(f"Debug - self.pred: {self.pred}")

                    for k in range(len(out_joints)):

                        person_out_joints = out_joints[k,:,0:1]
                        person_pred_joints = pred_joints[k,:,0:1]

                        gt_xy = person_out_joints[:,0,:2]
                        pred_xy = person_pred_joints[:,0,:2]
                        
                        # Ensure we don't exceed the available data length
                        min_len = min(len(gt_xy), len(pred_xy), self.pred)
                        sum_ade = 0

                        for t in range(min_len):
                            d1 = (gt_xy[t,0].detach().cpu().numpy() - pred_xy[t,0].detach().cpu().numpy())
                            d2 = (gt_xy[t,1].detach().cpu().numpy() - pred_xy[t,1].detach().cpu().numpy())
                        
                            dist_ade = [d1,d2]
                            sum_ade += np.linalg.norm(dist_ade)
                        
                        sum_ade /= min_len if min_len > 0 else 1
                        ade_batch += sum_ade
                        
                        # FDE calculation with bounds checking
                        if min_len > 0:
                            last_idx = min_len - 1
                            d3 = (gt_xy[last_idx,0].detach().cpu().numpy() - pred_xy[last_idx,0].detach().cpu().numpy())
                            d4 = (gt_xy[last_idx,1].detach().cpu().numpy() - pred_xy[last_idx,1].detach().cpu().numpy())
                            dist_fde = [d3,d4]
                            scene_fde = np.linalg.norm(dist_fde)
                            fde_batch += scene_fde
                        
                        total_samples += 1
                
                batch_id+=1
            if debug:
                ade_avg = ade_batch/total_samples
                fde_avg = fde_batch/total_samples
                print(f"ADE: {ade_avg}, FDE: {fde_avg}")

            return obs_pred_ls, gt_traj_ls, pred_outputs