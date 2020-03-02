import torch
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm

from autoencoder import Autoencoder
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from image_loader import ImageDataset
dataset = ImageDataset(return_hashes=True)

data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

autoencoder = Autoencoder(is_variational=USE_VARIATIONAL_AUTOENCODER)
autoencoder.load_state_dict(torch.load(AUTOENCODER_FILENAME))
autoencoder.eval()

with torch.no_grad():
    for sample in tqdm(data_loader):
        image, hash = sample
        hash = hash[0]

        output = autoencoder.decode(autoencoder.encode(image.to(device).unsqueeze(0)))
        
        result = torch.zeros((3, 128, 256))
        result[:, :, :128] = image.cpu()
        result[:, :, 128:] = output

        utils.save_image(result, 'data/test/{:s}.jpg'.format(hash))
