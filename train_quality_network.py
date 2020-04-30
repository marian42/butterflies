from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from quality_network import QualityNetwork
from quality_dataset import QualityDataset

dataset = QualityDataset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NETWORK_FILENAME = 'trained_models/quality.to'

network = QualityNetwork()
try:
    network.load_state_dict(torch.load(NETWORK_FILENAME))
except:
    print("Found no model, training a new one.")
network.cuda()

optimizer = optim.Adam(network.parameters(), lr=0.00001)
criterion = nn.BCELoss()

def train():
    for epoch in count():
        loss_history = []
        network.train()

        dataset.shuffle()
        data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
        for batch in tqdm(data_loader):
            images, labels, _ = batch
            images = images.to(device)
            labels = nn.functional.one_hot(labels, dataset.label_count).type(torch.float32).to(device)

            network.zero_grad()
            output = network(images).squeeze(1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            error = loss.item()
            loss_history.append(error)
        print(epoch, np.mean(loss_history))
        torch.save(network.state_dict(), NETWORK_FILENAME)

train()