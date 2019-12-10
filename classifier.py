import torch
import torch.nn as nn


BREADTH = 8

class BlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockDown, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, input):
        x, skip_values = input
        x = self.layers(x)
        skip_values = skip_values + [x]
        x = torch.nn.functional.max_pool2d(x, 2)
        return x, skip_values

class BlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, use_sigmoid=False):
        super(BlockUp, self).__init__()
        layers = [
            nn.Conv2d(in_channels = in_channels * 2, out_channels = out_channels, kernel_size = 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, padding=1),
            nn.BatchNorm2d(out_channels)
        ]

        if use_sigmoid:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, input):
        x, skip_values = input
        x = self.up_conv(x)
        x = nn.functional.relu(x)
        x = torch.cat([x, skip_values[-1]], dim=1) 
        x = self.layers(x)
        return x, skip_values[:-1]

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, input):
        x, skip_values = input
        x = self.layers(x)
        return x, skip_values


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layers = nn.Sequential(
            BlockDown(3, BREADTH * 1),
            BlockDown(BREADTH * 1, BREADTH * 2),
            BlockDown(BREADTH * 2, BREADTH * 4),
            BlockDown(BREADTH * 4, BREADTH * 8),
            BlockDown(BREADTH * 8, BREADTH * 8),
            BlockDown(BREADTH * 8, BREADTH * 8),
            Block(BREADTH * 8, BREADTH * 8),
            BlockUp(BREADTH * 8, BREADTH * 8),
            BlockUp(BREADTH * 8, BREADTH * 8),
            BlockUp(BREADTH * 8, BREADTH * 4),
            BlockUp(BREADTH * 4, BREADTH * 2),
            BlockUp(BREADTH * 2, BREADTH * 1),
            BlockUp(BREADTH * 1, 1, use_sigmoid=True)
        )

        self.cuda()

    def forward(self, x):
        return self.layers((x, []))[0]