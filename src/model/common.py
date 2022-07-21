import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import quantize
import numpy as np

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

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)




class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, a_bit, w_bit, qq_bit, finetune, with_BN, 
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        self.a_bit = a_bit 
        self.w_bit = w_bit
        # self.bn = with_BN
        
        
        self.quant1 = quantize.Quantization(bit=self.a_bit, qq_bit=qq_bit, finetune=finetune)
        self.quant2 = quantize.Quantization(bit=self.a_bit, qq_bit=qq_bit, finetune=finetune)
        
        # convolution
        if w_bit ==32:
            self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
            self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        else:
            self.conv1 = quantize.Conv2d_Q(n_feats, n_feats, kernel_size, stride=1, padding=1, bias=bias, dilation=1, groups=1, w_bit=w_bit, finetune=finetune)
            self.conv2 = quantize.Conv2d_Q(n_feats, n_feats, kernel_size, stride=1, padding=1, bias=bias, dilation=1, groups=1, w_bit=w_bit, finetune=finetune)

        # if with_BN:
        #     self.BN1 = nn.BatchNorm2d(n_feats)
        #     self.BN2 = nn.BatchNorm2d(n_feats)

            
        self.act = act
        self.res_scale = res_scale


    def forward(self, x):        

        if self.a_bit!=32:
            out= self.quant1(x)
        else:
            out=x
        
        out = self.conv1(out)
        # if self.bn:
        #     out = self.BN1(out)


        out1 = self.act(out)


        if self.a_bit!=32:
            out1= self.quant2(out1)

        res = self.conv2(out1)
        # if self.bn:
        #     res = self.BN2(res)
        res = res.mul(self.res_scale)

        res += x
        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
