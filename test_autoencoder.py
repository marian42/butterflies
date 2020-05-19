import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage import io
import numpy as np

from autoencoder import Autoencoder, LATENT_CODE_SIZE
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from image_loader import ImageDataset
dataset = ImageDataset(return_hashes=True)

SAMPLE_SIZE = 400
indices = [int(i / SAMPLE_SIZE * len(dataset)) for i in range(SAMPLE_SIZE)]
dataset.hashes = [dataset.hashes[i] for i in indices]

data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

autoencoder = Autoencoder(is_variational=USE_VARIATIONAL_AUTOENCODER)
autoencoder.load_state_dict(torch.load(AUTOENCODER_FILENAME))
autoencoder.eval()

STEPS = 5

with torch.no_grad():
    for sample in tqdm(data_loader):
        image, hash = sample
        hash = hash[0]

        latent_code = autoencoder.encode(image.to(device)).unsqueeze(0)

        
        result = torch.zeros((3, 128, 128 * (STEPS + 1)))
        result[:, :, :128] = image.cpu()

        for i in range(STEPS):
            output = autoencoder.decode(latent_code * (1.0 - i / (STEPS - 1)))
            result[:, :, 128 * (i + 1):128 * (i + 2)] = output
        
        result = (torch.clamp(result, 0, 1).numpy() * 255).astype(np.uint8).transpose((1, 2, 0))
        io.imsave('data/test/{:s}.jpg'.format(hash), result, quality=99)
