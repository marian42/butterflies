from autoencoder import EncoderBlock, Lambda
import torch.nn as nn

amcm = 8
intermediate_size = 64

def RotationNetwork(output_size = 2):
    return nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = 1 * amcm, kernel_size = 3, padding=1),
        nn.BatchNorm2d(1 * amcm),
        nn.ReLU(inplace=True),

        EncoderBlock(1 * amcm, 1 * amcm), # 64 -> 32
        EncoderBlock(1 * amcm), # 32 -> 16
        EncoderBlock(2 * amcm), # 16 -> 8
        EncoderBlock(4 * amcm), # 8 -> 4
        EncoderBlock(8 * amcm, intermediate_size, bottleneck=True), # 4 -> 1

        Lambda(lambda x: x.reshape(x.shape[0], -1)),

        nn.Linear(intermediate_size, intermediate_size),
        nn.BatchNorm1d(intermediate_size),
        nn.ReLU(inplace=True),
        nn.Linear(intermediate_size, output_size),
    )