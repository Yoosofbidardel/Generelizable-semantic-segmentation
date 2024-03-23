import torch
import torchvision
from torch import nn, optim
import numpy as np
import torch.nn.functional as F
import math
import random as rnd

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
        #x = self.bn1(inputs)
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

        self.in_c = in_c
        self.out_c = out_c
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

class SiameseSGR(nn.Module):
    def __init__(self, n_channels=3, n_classes=19):
        super().__init__()

        """ Encoder """
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.inc = ConvBlock(64, 64)
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
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        """ Decoder """
        d1 = self.d1(e4, e3)
        d2 = self.d2(d1, e2)
        d3 = self.d3(d2, e1)
        d4 = self.d4(d3, x)

        """ Classifier """
        outputs = self.outputs(d4)
        
        return outputs
    
    def l1(self, inputs):
        """ Encoder """
        conv1 = self.conv1(inputs)

        return conv1

    def l2(self, inputs):
        """ Encoder """
        bn1 = self.bn1(inputs)
        conv2 = self.conv2(bn1)
        x = self.bn2(conv2)
        x = self.relu(x)        
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        """ Decoder """
        d1 = self.d1(e4, e3)
        d2 = self.d2(d1, e2)
        d3 = self.d3(d2, e1)
        d4 = self.d4(d3, x)

        """ Classifier """
        #outputs = self.outputs(d4)

        return d4
    
    def l3(self, inputs):
        return self.outputs(inputs)

# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2):
        # Calculate the euclidean distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

        loss_contrastive = torch.mean(torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
    
class SensitivtyGuidedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f1_1, f1_2, fx_1, fx_2):
        # Calculate the euclidean distance and calculate the contrastive loss
        C = f1_1.size(dim=1)
        S = torch.mean(torch.abs(f1_1 - f1_2), dim=1)

        return torch.mean(S*torch.sum(torch.abs(fx_1-fx_2), dim=1))/C
        

class SelfGuidedRandomization(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f1, f2):
        B = f1.size(dim=0)
        C = f1.size(dim=1)
        W = f1.size(dim=2)
        H = f1.size(dim=3)

        f1_mu, f1_sigma = self.stats(f1, B, C, W, H)
        f2_mu, f2_sigma = self.stats(self.rndCrop(f2, W, H), B, C, W, H)
        F_mu = f2_mu - f1_mu
        F_sigma = f2_sigma - f1_sigma

        lmbd = rnd.uniform(0, 1)
        a1 = (f1_sigma + lmbd*F_sigma)
        a1 = a1 * ((f1 - f1_mu)/f1_sigma)
        a1 = a1 + (f1_mu + lmbd*F_mu)

        return a1

    def rndCrop(self, input, W, H):
        lim = 64
        W_lim = rnd.randint(0, W-lim)
        H_lim = rnd.randint(0, H-lim)
        
        output = input[:,:,W_lim:W_lim+lim,H_lim:H_lim+lim]

        return output

    def stats(self, input, B, C, W, H):
        mu_small = torch.mean(input, dim = [2,3], keepdim=True)
        mu = mu_small.expand(B, C, input.size(dim=2), input.size(dim=3))
        eps = 10**(-10)
        sigma = torch.sqrt(torch.mean(torch.pow(input - mu, 2) + eps, dim = [2,3], keepdim=True))
        sigma = sigma.expand(B, C, W, H)
        mu = mu_small.expand(B, C, W, H)

        return mu, sigma
