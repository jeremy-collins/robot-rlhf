import torch
import torch.nn as nn

class RLHF_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(RLHF_Model, self).__init__()
        # classifier  with 1 hidden layer
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 32),
            # nn.ReLU(),
            # nn.Linear(32, 32),
            # nn.ReLU(),
            # nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            # nn.Softmax(dim=0)
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # flattening all but the batch dimension
        x = x.view(x.size(0), -1)
        y_pred = self.classifier(x)

        return y_pred


if __name__ == '__main__':
    traj_1 = torch.zeros((10))
    y = torch.ones((1))
    # datapoint = (traj_1, y)
    # dataset is a set of trajectories with random y values
    dataset = []
    for i in range(10):
        traj = torch.randn((10))
        y = torch.randint(0, 2, (1,))
        dataset.append((traj, y))

    # turning x and y into tensors
    # traj_tensor = torch.stack([datapoint[0] for datapoint in dataset])
    # y_tensor = torch.stack([datapoint[1] for datapoint in dataset])

    # traj_2 = torch.ones((10))

    model = RLHF_Model(10, 1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rlhf = RLHF(model, torch.optim.Adam(model.parameters(), lr=0.01), nn.BCELoss(), device)    

    rlhf.train(dataset, 100)