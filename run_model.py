import torch
from torch.utils.data import DataLoader
from rlhf import RLHF_Model
from loader import RLHF_Dataset
import os
import numpy as np
from rlhf_utils import *

def run_model(model, trajectory, device):
    model.eval()
    with torch.no_grad():
        x = trajectory.to(device)
        x = x.float()
        pred = model(x)
        return pred
    

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RLHF_Model(input_size=25*120, output_size=1).to(device)

    model.load_state_dict(torch.load('checkpoints/model_26_best.pt'))

    # turning ordered dict into model
    model = model.to(device)

    # model.eval()
    # for i in range(80, 100):
    #     traj_data = np.load(f'data/trajectory_random_policy/test/RobotMotionStretch-v1_test_{i}.npz')
    #     obs, action = traj_data['observation_list'], traj_data['action_list']
    #     traj = np.concatenate((obs, action), axis=1)
    #     traj = traj[::4, :]

    #     traj = torch.from_numpy(traj).unsqueeze(0)

    #     # print('traj:', traj.shape)

    #     pred = run_model(model, traj, device)
    #     reward = model.prob_to_reward(pred)

    #     print('pred:', pred, 'reward:', reward)

    val_dataset = RLHF_Dataset('data/trajectory_random_policy/test')
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    for i, (x, y) in enumerate(val_loader):
        x = x.to(device)
        x = x.float()
        pred = model(x)
        reward = prob_to_reward(pred)
        print('pred:', pred, 'gt:', y, 'reward:', reward)
