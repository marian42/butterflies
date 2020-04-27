import os
import numpy as np
from skimage import io, transform
from tqdm import tqdm
import random
from quality_network import QualityNetwork
from config import *
import torch
import shutil
import csv

from image_loader import ImageDataset
dataset = ImageDataset(return_hashes=True)

NETWORK_FILENAME = 'trained_models/quality.to'

indices = list(range(len(dataset)))
random.shuffle(indices)

network = QualityNetwork()
network.load_state_dict(torch.load(NETWORK_FILENAME))
network.cuda()
network.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quality_file = open(QUALITY_DATA_FILENAME, 'r')
reader = csv.reader(quality_file)
label_ids = set(row[0] for row in reader)
quality_file.close

for label in range(3):
    try:
        os.mkdir('data/test/{:d}'.format(label))
    except FileExistsError:
        pass

for index in tqdm(indices):
    image, hash = dataset[index]

    if hash in label_ids:
        continue

    with torch.no_grad():
        prediction = network(image.unsqueeze(0).to(device)).squeeze()

    label = torch.argmax(prediction).item()

    shutil.copyfile('data/images_rotated_128/{:s}.jpg'.format(hash), 'data/test/{:d}/{:s}.jpg'.format(label, hash))