import torch
from torch.utils.data import DataLoader
from rlhf import RLHF_Model
import os
import numpy as np

def prob_to_reward(prob):
        if prob == 1:
            return 10
        reward = torch.log(prob / (1 - prob))
        return reward.item()

def npz_to_traj(npz_path, skip=None):
    traj_data = np.load(npz_path)
    obs, action = traj_data['observation_list'], traj_data['action_list']
    traj = np.concatenate((obs, action), axis=1)
    if skip is not None:
        traj = traj[::skip, :]

    return traj