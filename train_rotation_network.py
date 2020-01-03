from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from rotation_network import RotationNetwork
from rotation_dataset import RotationDataset
import random
from PIL import Image, ImageDraw

dataset = RotationDataset()
data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, drop_last=True)

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

@torch.no_grad()
def create_example():
    network.eval()
    index = random.randint(0, len(dataset) -1)
    image, _ = dataset[index]
    prediction = network.forward(image.to(device).unsqueeze(0)).squeeze()
    image = transforms.ToPILImage()(image).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.line((32 + prediction[1] * 64, 32 - prediction[0] * 64, 32 - prediction[1] * 64, 32 + prediction[0] * 64), width=3, fill=0)
    image.save('data/test/{:s}.jpg'.format(dataset.image_ids[index]))

def train():
    for epoch in count():
        loss_history = []
        network.train()
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
        create_example()

train()