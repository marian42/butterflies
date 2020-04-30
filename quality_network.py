from autoencoder import Lambda
import torch.nn as nn
import torch

def create_block(in_channels, out_channels=None):
    if out_channels is None:
        out_channels = in_channels
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def QualityNetwork(output_size = 3):
    return nn.Sequential(
        nn.AvgPool2d(2), # 128 -> 64

        create_block(3, 8),
        nn.MaxPool2d(2), # 64 -> 32

        create_block(8, 16),
        nn.MaxPool2d(2), # 32 -> 16

        create_block(16, 16),
        nn.MaxPool2d(2), # 16 -> 8


        create_block(16, 16),
        nn.MaxPool2d(2), # 8 -> 4

        create_block(16, 32),
        nn.MaxPool2d(2), # 4 -> 2

        Lambda(lambda x: x.reshape(x.shape[0], -1)),

        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),
        nn.Linear(64, output_size),
        nn.Softmax(dim=1),
    )