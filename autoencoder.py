import torch
import torch.nn as nn

standard_normal_distribution = torch.distributions.normal.Normal(0, 1)

class Lambda(nn.Module):
    def __init__(self, function):
        super(Lambda, self).__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)

class PrintShape(nn.Module):
    def __init__(self):
        super(PrintShape, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

LATENT_CODE_SIZE = 128

amcm = 32

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        return nn.functional.relu(self.layers(x) + x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, bottleneck=False):
        super(EncoderBlock, self).__init__()

        if out_channels is None:
            out_channels = in_channels * 2
        
        self.r1 = ResidualBlock(in_channels)
        self.r2 = ResidualBlock(in_channels)

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 4, stride = 1 if bottleneck else 2, padding = 0 if bottleneck else 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.r1(x)
        x = self.r2(x)
        x = self.layers(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, bottleneck=False):
        super(DecoderBlock, self).__init__()

        if out_channels is None:
            out_channels = in_channels // 2

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 4, stride = 1 if bottleneck else 2, padding = 0 if bottleneck else 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.r1 = ResidualBlock(out_channels)
        self.r2 = ResidualBlock(out_channels)
    
    def forward(self, x):
        x = self.layers(x)
        x = self.r1(x)
        x = self.r2(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, is_variational = True):
        super(Autoencoder, self).__init__()

        self.is_variational = is_variational

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 1 * amcm, kernel_size = 3, padding=1),
            nn.BatchNorm2d(1 * amcm),
            nn.ReLU(inplace=True),

            EncoderBlock(1 * amcm, 1 * amcm), # 128 -> 64
            EncoderBlock(1 * amcm), # 64 -> 32
            EncoderBlock(2 * amcm, 2 * amcm), # 32 -> 16
            EncoderBlock(2 * amcm), # 16 -> 8
            EncoderBlock(4 * amcm), # 8 -> 4
            EncoderBlock(8 * amcm, LATENT_CODE_SIZE, bottleneck=True), # 4 -> 1

            Lambda(lambda x: x.reshape(x.shape[0], -1)),

            nn.Linear(LATENT_CODE_SIZE, LATENT_CODE_SIZE),
            nn.BatchNorm1d(LATENT_CODE_SIZE),
            nn.ReLU(inplace=True),
            nn.Linear(LATENT_CODE_SIZE, LATENT_CODE_SIZE),
        )

        if is_variational:
            self.encoder.add_module('vae-bn', nn.BatchNorm1d(LATENT_CODE_SIZE))
            self.encoder.add_module('vae-lr', nn.ReLU(inplace=True))

            self.encode_mean = nn.Linear(in_features=LATENT_CODE_SIZE, out_features=LATENT_CODE_SIZE)
            self.encode_log_variance = nn.Linear(in_features=LATENT_CODE_SIZE, out_features=LATENT_CODE_SIZE)
        
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_CODE_SIZE, LATENT_CODE_SIZE),
            nn.BatchNorm1d(LATENT_CODE_SIZE),
            nn.ReLU(inplace=True),

            Lambda(lambda x: x.reshape(-1, LATENT_CODE_SIZE, 1, 1)),

            DecoderBlock(LATENT_CODE_SIZE, 8 * amcm, bottleneck=True), # 1 -> 4
            DecoderBlock(8 * amcm), # 4 -> 8
            DecoderBlock(4 * amcm), # 8 -> 16
            DecoderBlock(2 * amcm, 2 * amcm), # 16 -> 32
            DecoderBlock(2 * amcm), # 32 -> 64
            DecoderBlock(1 * amcm, 1 * amcm), # 32 -> 128

            nn.Conv2d(in_channels = 1 * amcm, out_channels = 3, kernel_size = 3, padding=1)
        )

        self.cuda()

    def encode(self, x, return_mean_and_log_variance = False):
        x = x.reshape((-1, 3, 128, 128))
        x = self.encoder.forward(x)

        if not self.is_variational:
            return x
            
        mean = self.encode_mean(x).squeeze()
        
        if self.training or return_mean_and_log_variance:
            log_variance = self.encode_log_variance(x).squeeze()
            standard_deviation = torch.exp(log_variance * 0.5)
            eps = standard_normal_distribution.sample(mean.shape).to(x.device)
        
        if self.training:
            x = mean + standard_deviation * eps
        else:
            x = mean

        if return_mean_and_log_variance:
            return x, mean, log_variance
        else:
            return x

    def decode(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(dim = 0)  # add dimension for channels
        x = self.decoder.forward(x)
        return x.squeeze()

    def forward(self, x):
        z, mean, log_variance = self.encode(x, return_mean_and_log_variance = True)
        x = self.decode(z)
        return x, mean, log_variance
