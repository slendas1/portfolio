import torch
import torch.nn as nn
import torch.nn.functional as F
from auxiliary_func import conv, deconv, ResidualBlock


class Discriminator(nn.Module):

    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()

        self.conv1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=4)  # (128, 128, 64)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=4,
                          instance_norm=True)  # (64, 64, 128)
        self.conv3 = conv(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=4,
                          instance_norm=True)  # (32, 32, 256)
        self.conv4 = conv(in_channels=conv_dim * 4, out_channels=conv_dim * 8, kernel_size=4,
                          instance_norm=True)  # (16, 16, 512)

        self.conv5 = conv(conv_dim * 8, out_channels=1, kernel_size=4, stride=1)  # (8, 8, 1)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2)
        out = F.leaky_relu(self.conv3(out), negative_slope=0.2)
        out = F.leaky_relu(self.conv4(out), negative_slope=0.2)

        out = self.conv5(out)
        return out


class Generator(nn.Module):

    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(Generator, self).__init__()
                                                                                          # (256, 256, 3)
        self.conv1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=7, stride=1, padding=3)  # (256, 256, 64)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=3,
                          instance_norm=True)  # (128, 128, 128)
        self.conv3 = conv(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=3,
                          instance_norm=True)  # (64, 64, 256)

        res_layers = []
        for layer in range(n_res_blocks):
            res_layers.append(ResidualBlock(conv_dim * 4))
        self.res_blocks = nn.Sequential(*res_layers)

        self.deconv4 = deconv(in_channels=conv_dim * 4, out_channels=conv_dim * 2, kernel_size=4, 
                              instance_norm=True)  # (128, 128, 128)
        self.deconv5 = deconv(in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=4,
                              instance_norm=True)  # (256, 256, 64)
        self.deconv6 = deconv(in_channels=conv_dim, out_channels=3, kernel_size=7, stride=1, padding=3,  instance_norm=True)  # (256, 256, 3)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2)
        out = F.leaky_relu(self.conv3(out), negative_slope=0.2)
        
        out = self.res_blocks(out)
        
        out = F.leaky_relu(self.deconv4(out), negative_slope=0.2)
        out = F.leaky_relu(self.deconv5(out), negative_slope=0.2)
        out = torch.tanh(self.deconv6(out))

        return out
