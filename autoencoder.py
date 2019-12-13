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

amcm = 8

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 1 * amcm, kernel_size = 4, stride = 2, padding = 1), # 128 -> 64
            nn.BatchNorm2d(1 * amcm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels = 1 * amcm, out_channels = 2 * amcm, kernel_size = 4, stride = 2, padding = 1), # 64 -> 32
            nn.BatchNorm2d(2 * amcm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels = 2 * amcm, out_channels = 4 * amcm, kernel_size = 4, stride = 2, padding = 1), # 32 -> 16
            nn.BatchNorm2d(4 * amcm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels = 4 * amcm, out_channels = 8 * amcm, kernel_size = 4, stride = 2, padding = 1), # 16 -> 8
            nn.BatchNorm2d(8 * amcm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels = 8 * amcm, out_channels = 16 * amcm, kernel_size = 4, stride = 2, padding = 1), # 8 -> 4
            nn.BatchNorm2d(16 * amcm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels = 16 * amcm, out_channels = LATENT_CODE_SIZE, kernel_size = 4, stride = 1), # 4 -> 1
            nn.BatchNorm2d(LATENT_CODE_SIZE),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            Lambda(lambda x: x.reshape(x.shape[0], -1)),

            nn.BatchNorm1d(LATENT_CODE_SIZE),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        self.encode_mean = nn.Linear(in_features=LATENT_CODE_SIZE, out_features=LATENT_CODE_SIZE)
        self.encode_log_variance = nn.Linear(in_features=LATENT_CODE_SIZE, out_features=LATENT_CODE_SIZE)
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features = LATENT_CODE_SIZE, out_features=LATENT_CODE_SIZE * 2),
            nn.BatchNorm1d(LATENT_CODE_SIZE * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            Lambda(lambda x: x.reshape(-1, LATENT_CODE_SIZE * 2, 1, 1)),
            nn.ConvTranspose2d(in_channels = LATENT_CODE_SIZE * 2, out_channels = 16 * amcm, kernel_size = 4, stride = 1), # 1 -> 4
            nn.BatchNorm2d(16 * amcm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(in_channels = 16 * amcm, out_channels = 8 * amcm, kernel_size = 4, stride = 2, padding = 1), # 4 -> 8
            nn.BatchNorm2d(8 * amcm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(in_channels = 8 * amcm, out_channels = 4 * amcm, kernel_size = 4, stride = 2, padding = 1), # 8 -> 16
            nn.BatchNorm2d(4 * amcm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(in_channels = 4 * amcm, out_channels = 2 * amcm, kernel_size = 4, stride = 2, padding = 1), # 16 -> 32
            nn.BatchNorm2d(2 * amcm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.ConvTranspose2d(in_channels = 2 * amcm, out_channels = 1 * amcm, kernel_size = 4, stride = 2, padding = 1), # 32 -> 64
            nn.BatchNorm2d(1 * amcm),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(in_channels = 1 * amcm, out_channels = 3, kernel_size = 4, stride = 2, padding = 1), # 64 -> 128
        )

        self.cuda()

    def encode(self, x, return_mean_and_log_variance = False):
        x = x.reshape((-1, 3, 128, 128))
        x = self.encoder.forward(x)

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
