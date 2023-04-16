import torch
from torch.utils.data import DataLoader
from rlhf import RLHF_Model
from loader import RLHF_Dataset
import os

def train_epoch(model, optimizer, loss_fn, data_loader, device):
    size = len(data_loader.dataset)
    loss_sum = 0
    correct = 0
    model.train()
    for batch, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)

        x = x.float()
        y = y.float()

        # Compute prediction error
        pred = model(x)
        # print('pred:', pred)
        # print('y:', y)
        # print('correct:', (pred > 0.5).type(torch.float).eq(y).sum().item())

        correct += (pred > 0.5).type(torch.float).eq(y).sum().item()

        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    return loss_sum / size, correct / size

def val_epoch(model, loss_fn, data_loader, device):
    size = len(data_loader.dataset)
    loss_sum = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            
            x = x.float()
            y = y.float()

            pred = model(x)
            loss = loss_fn(pred, y).item()

            loss_sum += loss

            correct += (pred > 0.5).type(torch.float).eq(y).sum().item()

    return loss_sum / size, correct / size

if __name__ == '__main__':
    # Hyperparameters
    # learning_rate = 1e-5
    learning_rate = 5e-4
    batch_size = 4
    epochs = 20

    # Other variables
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = RLHF_Dataset('data/trajectory_random_policy/train')
    val_dataset = RLHF_Dataset('data/trajectory_random_policy/test')

    # Load data
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Define model
    model = RLHF_Model(input_size=25*120, output_size=1).to(device)

    # printing number of parameters
    print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCELoss(reduction='sum')

    # index is the number of models saved in checkpoints folder
    # model_0 -> 0, model_1_best -> 1, model_2 -> 2, model_2_best -> 2
    index = max([int(f.split('_')[1].split('.')[0]) for f in os.listdir('checkpoints') if f.endswith('.pt')]) + 1
    min_val_loss = 1000
    max_val_acc = 0

    # Train model
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss, train_acc = train_epoch(model, optimizer, loss_fn, train_loader, device)
        val_loss, val_acc = val_epoch(model, loss_fn, val_loader, device)
        print(f"Train loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Validation loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")

        # Save model
        torch.save(model.state_dict(), f'checkpoints/model_{index}.pt')

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'checkpoints/model_{index}_best.pt')

        if val_acc > max_val_acc:
            max_val_acc = val_acc
    
    print("Done!", 'best val loss:', min_val_loss, 'best val acc:', max_val_acc)