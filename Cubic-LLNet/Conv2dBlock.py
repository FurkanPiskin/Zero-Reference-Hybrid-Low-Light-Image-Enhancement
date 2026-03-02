import torch
import torch.nn as nn
from SpatialAttention import SpatialAttention

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,1], stride=1, padding=[1,0]):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[0],
                               stride=stride, padding=padding[0], bias=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[1],
                               stride=stride, padding=padding[1], bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

        self.attention = SpatialAttention(kernel_size=7)
    
    def forward(self, x):
        x = self.conv1(x) + self.conv2(x)
        x = self.bn(x)
        x = self.gelu(x)
        x=self.attention(x)
        return x
