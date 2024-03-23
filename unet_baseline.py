import torch
import torchvision
from torch import nn, optim
import numpy as np
import torch.nn.functional as F

""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    
class Conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, inputs):
        p = self.pool(inputs)
        x = self.conv(p)

        return x

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        #self.up2 = nn.ConvTranspose2d(in_c, in_c, kernel_size=2, stride=2, padding=0)
        self.conv_block = ConvBlock(in_c, out_c)
        self.conv = Conv(in_c, out_c)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        x = self.conv_block(x)

        return x

class UnetModel(nn.Module):
    def __init__(self, n_channels=3, n_classes=19):
        super().__init__()

        """ Encoder """
        self.inc = ConvBlock(n_channels, 64)
        self.e1 = EncoderBlock(64, 128)
        self.e2 = EncoderBlock(128, 256)
        self.e3 = EncoderBlock(256, 512)
        self.e4 = EncoderBlock(512, 1024)

        """ Decoder """
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        """ Classifier """
        #self.outputs = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)
        self.outputs = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, inputs):
        """ Encoder """
        inc = self.inc(inputs)
        e1 = self.e1(inc)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        """ Decoder """
        d1 = self.d1(e4, e3)
        d2 = self.d2(d1, e2)
        d3 = self.d3(d2, e1)
        d4 = self.d4(d3, inc)

        """ Classifier """
        outputs = self.outputs(d4)
        
        return outputs