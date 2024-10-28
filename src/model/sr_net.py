
import torch.nn as nn

from src.model.layers.residual import ResBlock


class SRNet(nn.Module):
    def __init__(self, sampling, features=64, kernel_size=3, blocks=3, channels=3):
        super(SRNet, self).__init__()
        up_kernel = 2*sampling+1
        self.sampling = sampling
        self.upsample = nn.ConvTranspose2d(channels, channels, kernel_size=up_kernel, stride=sampling, padding=up_kernel//2, output_padding=sampling-1)
        self.add_features = nn.Conv2d(channels, features, kernel_size=kernel_size, padding=kernel_size//2)
        self.residual = nn.ModuleList([ResBlock(features, kernel_size=kernel_size) for _ in range(blocks)])
        self.obtain_channels = nn.Conv2d(features, channels, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, low):
        bic = nn.functional.interpolate(low, scale_factor=self.sampling, mode='bicubic', align_corners=False)
        x_up =self.upsample(low)

        features = self.add_features(x_up)
        for res_block in self.residual:
            features = res_block(features)
        res = self.obtain_channels(features)
        return bic + res
