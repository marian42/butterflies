import torch
from torch.utils.data import Dataset
import numpy as np
import csv
import random
from config import *
import itertools
import torch

class QualityDataset(Dataset):
    def __init__(self, return_hashes=False):
        self.label_count = 3
        file = open(QUALITY_DATA_FILENAME, 'r')
        reader = csv.reader(file)

        self.ids_by_label = [[] for i in range(self.label_count)]

        for row in reader:
            label = int(row[1])
            self.ids_by_label[label].append(row[0])
        
        self.shuffle()

    def shuffle(self):
        size = min(len(i) for i in self.ids_by_label)
        ids = list(itertools.chain(*[random.sample(population, size) for population in self.ids_by_label]))
        indices = list(range(size * self.label_count))
        random.shuffle(indices)
        self.ids = [ids[i] for i in indices]
        self.labels = [i // size for i in indices]
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image = io.imread('data/images_rotated_128/{:s}.jpg'.format(self.ids[index]))
        image = image.transpose((2, 0, 1)).astype(np.float32) / 255
        image = torch.from_numpy(image)

        return image, self.labels[index], self.ids[index]