import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
from rlhf_utils import *

class RLHF_Dataset(Dataset):
    def __init__(self, root):
        if type(root) == str:
            self.root = [os.path.join(os.getcwd(), root)]
        
        elif type(root) == list:
            self.root = root

        self.dataset = self.get_data()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        traj, label = self.dataset[idx]

        return traj, label
    
    def gen_rand_labels(self, num_labels):
        labels = torch.randint(0, 2, (num_labels,))
        labels = labels.unsqueeze(0)
        return labels

    def get_data(self):
        self.dataset = []
        # labels = self.gen_rand_labels(100) # (100,)
        csv_path = 's3_output_6.csv' # cols are: video_name_1, video_url_1, video_name_2, video_url_2, label_1, label_2
        df = pd.read_csv(csv_path)

        # loading the folder of npz files
        for folder in self.root:
            for root, dirs, files in os.walk(folder):
                for i, file in enumerate(files):
                    if file.endswith(".npz"):
                        file_path = os.path.join(root, file)

                        # # load the npz file
                        # data = np.load(file_path)
                        # # extract the data
                        # obs = data['observation_list']
                        # action = data['action_list']
                        # print('obs:', obs.shape)
                        # print('action:', action.shape)
                        # trajectory = np.concatenate((obs, action), axis=1) # (120, 25)

                        trajectory = npz_to_traj(file_path, skip=None)

                        # taking every fourth row of the trajectory
                        # trajectory = trajectory[::4, :]
                        
                        # label = torch.randint(0, 2, (1,))
                        # label = torch.ones((1,))
                        # label = labels[:, i]

                        # get the label of the npz file
                        video_name = file[:-4] + '.mp4'

                        # the label of the npz file is the label of the video with the same name
                        if video_name in df['video_name_1'].values:
                            label = df.loc[df['video_name_1'] == video_name, 'label_1'].values
                        elif video_name in df['video_name_2'].values:
                            label = df.loc[df['video_name_2'] == video_name, 'label_2'].values

                        print('video_name:', video_name, 'label:', label)
                        label = torch.tensor(label)

                        # print('trajectory:', trajectory)
                        print('trajectory:', trajectory.shape)
                        print('label:', label)
                        print('label:', label.shape)
                        self.dataset.append((trajectory, label))
        
        print('dataset size:', len(self.dataset))
        return self.dataset
    

if __name__ == '__main__':
    rlhf = RLHF_Dataset(root=['data/trajectory_random_policy/test'])
    rlhf.get_data()