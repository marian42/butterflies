import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from autoencoder import Autoencoder, LATENT_CODE_SIZE

from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUTOENCODER_FILENAME = 'trained_models/autoencoder.to'

from image_loader import ImageDataset
dataset = ImageDataset()
BATCH_SIZE = 256

data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load(AUTOENCODER_FILENAME))

latent_codes = np.zeros((len(dataset), LATENT_CODE_SIZE), dtype=np.float32)
position = 0

with torch.no_grad():
    for batch in tqdm(data_loader):
        current = autoencoder.encode(batch.to(device))
        latent_codes[position:position+current.shape[0], :] = current.cpu().numpy()
        position += current.shape[0]

np.save('data/latent_codes.npy', latent_codes)