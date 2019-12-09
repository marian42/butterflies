from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import sys
from torch.utils.data import DataLoader
from torchvision import utils

from classifier import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER_FILENAME = 'trained_models/classifier.to'

classifier = Classifier()
classifier.cuda()

from mask_loader import MaskDataset
dataset = MaskDataset()
data_loader = DataLoader(dataset, shuffle=True, num_workers=4)

if "continue" in sys.argv:
    classifier.load_state_dict(torch.load(CLASSIFIER_FILENAME))

optimizer = optim.Adam(classifier.parameters(), lr=0.001)
criterion = nn.BCELoss()

def save_example(epoch, hash, image, mask):
    mask_binary = torch.zeros(mask.shape)
    mask_binary[mask > 0.5] = 1
    w, h = mask.shape[1] // 2, mask.shape[2] // 2
    mask[:, :w, :h] = mask_binary[:, :w, :h]
    mask[:, w:, h:] = mask_binary[:, w:, h:]
    result = image.clone().squeeze(0)
    result *= mask

    utils.save_image(result, 'data/generated/{:04d}-{:s}.jpg'.format(epoch, hash))

def train():
    for epoch in count():
        loss_history = []
        for batch in data_loader:
            image, mask, hash = batch
            image = image.to(device)
            mask = mask.to(device)


            classifier.zero_grad()
            output = classifier(image).squeeze(1)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            error = loss.item()
            loss_history.append(error)

            if epoch % 10 == 0:
                save_example(epoch, hash[0], image, output)
        print(epoch, np.mean(loss_history))
        if epoch % 10 == 0:
            torch.save(classifier.state_dict(), CLASSIFIER_FILENAME)

train()