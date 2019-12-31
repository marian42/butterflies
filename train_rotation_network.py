from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from rotation_network import RotationNetwork
from rotation_dataset import RotationDataset

dataset = RotationDataset()
data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NETWORK_FILENAME = 'trained_models/rotation.to'

network = RotationNetwork()
try:
    network.load_state_dict(torch.load(NETWORK_FILENAME))
except:
    print("Found no model, training a new one.")
network.cuda()

optimizer = optim.Adam(network.parameters(), lr=0.00001)
criterion = nn.MSELoss()

def train():
    for epoch in count():
        loss_history = []
        for batch in tqdm(data_loader):
            image, correct_result = batch
            image = image.to(device)
            correct_result = correct_result.to(device)

            network.zero_grad()
            output = network(image).squeeze(1)
            loss = criterion(output, correct_result)
            loss.backward()
            optimizer.step()
            error = loss.item()
            loss_history.append(error)
        print(epoch, np.mean(loss_history))
        torch.save(network.state_dict(), NETWORK_FILENAME)

train()