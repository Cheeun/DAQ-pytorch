import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import quantize

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, a_bit, w_bit, qq_bit, finetune,
        bias=False, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        self.a_bit = a_bit 
        self.w_bit = w_bit
        

        # activation
        
        self.quant1 = quantize.Quantization(bit=self.a_bit, qq_bit=qq_bit, finetune=finetune)
        self.quant2 = quantize.Quantization(bit=self.a_bit, qq_bit=qq_bit, finetune=finetune)

        if w_bit ==32:
            self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
            self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        else:
            self.conv1 = quantize.Conv2d_Q(n_feats, n_feats, kernel_size, stride=1, padding=1, bias=bias, dilation=1, groups=1, w_bit=w_bit, finetune=finetune)
            self.conv2 = quantize.Conv2d_Q(n_feats, n_feats, kernel_size, stride=1, padding=1, bias=bias, dilation=1, groups=1, w_bit=w_bit, finetune=finetune)   

        self.bn1 = nn.BatchNorm2d(n_feats)
        self.act = act
        self.bn2 = nn.BatchNorm2d(n_feats)
        self.res_scale = res_scale


    def forward(self, x):
        if self.a_bit!=32:
            out= self.quant1(x)
        else:
            out=x
        
        out1 = self.act(self.bn1(self.conv1(out)))

        if self.a_bit!=32:
            out1 = self.quant2(out1)

        res = self.bn2(self.conv2(out1))
        # res = res.mul(self.res_scale)
        

        res += x
        
        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        scale = 4 # for SRResNet
        m = []
        m.append(conv(n_feats, 4 * n_feats, 3, bias=False))
        m.append(nn.PixelShuffle(2))
        m.append(nn.PReLU())
        # m.append(nn.LeakyReLU(0.2, inplace=True))
        m.append(conv(n_feats, 4 * n_feats, 3, bias=False))
        m.append(nn.PixelShuffle(2))
        # m.append(nn.LeakyReLU(0.2, inplace=True))
        m.append(nn.PReLU())
        
        super(Upsampler, self).__init__(*m)
