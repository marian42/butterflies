from tqdm import tqdm
from quality_network import QualityNetwork
from config import QUALITY_CALCULATED_FILENAME
import torch
from torch.utils.data import DataLoader

from image_loader import ImageDataset
dataset = ImageDataset(return_hashes=True)
dataloader = DataLoader(dataset, batch_size=500, num_workers=16)

NETWORK_FILENAME = 'trained_models/quality.to'

network = QualityNetwork()
network.load_state_dict(torch.load(NETWORK_FILENAME))
network.cuda()
network.eval()

quality_file = open(QUALITY_CALCULATED_FILENAME, 'w')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

counts = [0, 0, 0]

for image, ids in tqdm(dataloader):
    with torch.no_grad():
        prediction = network(image.to(device)).squeeze()

    labels = torch.argmax(prediction, dim=1)

    for index, id in enumerate(ids):
        label = labels[index].item()
        quality_file.write('{:s},{:d}\n'.format(id, label))
        counts[label] += 1

quality_file.close()
print('Total label counts:', counts)
