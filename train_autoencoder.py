from itertools import count
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from collections import deque

from autoencoder import Autoencoder
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from image_loader import ImageDataset
dataset = ImageDataset()
BATCH_SIZE = 32

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

autoencoder = Autoencoder(is_variational=USE_VARIATIONAL_AUTOENCODER)

if os.path.exists(AUTOENCODER_FILENAME):
    print("Found autoencoder model, resuming training on existing model.")
    autoencoder.load_state_dict(torch.load(AUTOENCODER_FILENAME), strict=False)
else:
    print("Found no autoencoder model, training a new one.")

autoencoder.train()

optimizer = optim.Adam(autoencoder.parameters(), lr=0.00005)
criterion = lambda a, b: torch.mean(torch.abs(a - b))

LOG_STEPS = 1000

error_history = deque(maxlen=LOG_STEPS)
kld_history = deque(maxlen=LOG_STEPS)

def kld_loss(mean, log_variance):
    return -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp()) / mean.nelement()

def train():
    for epoch in count():
        batch_index = 0
        for sample in tqdm(data_loader):
            sample = sample.to(device)

            autoencoder.zero_grad()

            if USE_VARIATIONAL_AUTOENCODER:
                output, mean, log_variance = autoencoder.forward(sample)
                kld = kld_loss(mean, log_variance)
            else:
                output = autoencoder.decode(autoencoder.encode(sample))
                kld = 0

            reconstruction_loss = criterion(output, sample)
            error_history.append(reconstruction_loss.item())
            kld_history.append(kld.item())

            loss = reconstruction_loss + kld * 0.01
            
            loss.backward()
            optimizer.step()
            batch_index += 1
            if batch_index % LOG_STEPS == 0:
                torch.save(autoencoder.state_dict(), AUTOENCODER_FILENAME)
                tqdm.write("Epoch {:d}, batch {:d}".format(epoch, batch_index) \
                + ': reconstruction loss: {0:.5f}'.format(np.mean(error_history)) \
                + ', KLD loss: {0:.4f} (Saved model.)'.format(np.mean(kld_history)))

        print("Epoch " + str(epoch) \
                + ': reconstruction loss: {0:.5f}'.format(np.mean(error_history)) \
                + ', KLD loss: {0:.4f}'.format(np.mean(kld_history)))
    
        torch.save(autoencoder.state_dict(), AUTOENCODER_FILENAME)
        torch.save(autoencoder.state_dict(), 'trained_models/checkpoints/autoencoder_{:04d}.to'.format(epoch))

train()