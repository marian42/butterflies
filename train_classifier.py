from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import sys
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm
from skimage import io

from classifier import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER_FILENAME = 'trained_models/classifier.to'

SAVE_EXAMPLES = True

classifier = Classifier()
try:
    classifier.load_state_dict(torch.load(CLASSIFIER_FILENAME))
except:
    print("Found no model, training a new one.")
classifier.cuda()

from mask_loader import MaskDataset
dataset = MaskDataset()
data_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8)

optimizer = optim.Adam(classifier.parameters(), lr=0.0002)
criterion = nn.BCELoss()

def save_example(epoch, hash, image, mask):
    mask_binary = torch.zeros(mask.shape)
    mask_binary[mask > 0.5] = 1
    w, h = mask.shape[0] // 2, mask.shape[1] // 2
    mask[:w, :h] = mask_binary[:w, :h]
    mask[w:, h:] = mask_binary[w:, h:]
    result = image.clone().squeeze(0)
    result *= mask
    result = result.cpu().numpy().transpose((1, 2, 0)) * 255
    result = result.astype(np.uint8)

    io.imsave('data/test/{:s}.jpg'.format(hash), result, quality=95)

def train():
    total_batches = 0
    for epoch in count():
        loss_history = []
        for batch in tqdm(data_loader):
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

            if total_batches % 10 == 0 and SAVE_EXAMPLES:
                i = 0
                save_example(epoch, hash[i], image[i, :, :], output[i, :, :].detach())
            total_batches += 1
        print(epoch, np.mean(loss_history))
        torch.save(classifier.state_dict(), CLASSIFIER_FILENAME)

train()