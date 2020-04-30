import torch
import torch.nn as nn
import cv2
import numpy as np

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


def remove_smaller_components(array):
    mask = (array > 0.5)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.squeeze().cpu().numpy().astype(np.uint8), connectivity=4)
    max_label = np.argmax(stats[1:, 4]) + 1
    array[torch.from_numpy(labels != max_label).unsqueeze(0)] = 0

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

    @torch.no_grad()
    def apply(self, image, clipping_range=0.2, background=(1, 1, 1), create_alpha=False, crop=True, margin=0.05):
        if len(image.shape) > 3:
            image = image.squeeze(0)
        
        mask = self(image.unsqueeze(0)).squeeze(0)
        remove_smaller_components(mask)
        background = torch.tensor(background, device=image.device, dtype=torch.float32).reshape(3, 1, 1)
        
        if clipping_range is not None:
            mask = (mask - 0.5) * 2
            mask.clamp_(-clipping_range, clipping_range)
            mask /= clipping_range
            mask = mask / 2 + 0.5

        coords = (mask > 0.5).nonzero()
    
        if coords.nelement() == 0:
            return None

        top_left, _ = torch.min(coords, dim=0)
        bottom_right, _ = torch.max(coords, dim=0)
        
        mask = mask[:, top_left[1]:bottom_right[1], top_left[2]:bottom_right[2]]
        image = image[:, top_left[1]:bottom_right[1], top_left[2]:bottom_right[2]]

        image = image * mask + (1.0 - mask) * background

        new_size = int(max(image.shape[1], image.shape[2]) * (1 + margin))
        
        result = torch.zeros((4 if create_alpha else 3, new_size, new_size), device=image.device)
        y, x = (new_size - image.shape[1]) // 2, (new_size - image.shape[2]) // 2
        result[:3, :, :] = background
        result[:3, y:y+image.shape[1], x:x+image.shape[2]] = image

        if create_alpha:
            result[3, y:y+image.shape[1], x:x+image.shape[2]] = mask

        return result