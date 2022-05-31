from model import common

import torch
import torch.nn as nn

from model import quantize

def make_model(args, parent=False):
    return RDN(args)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, a_bit, w_bit, qq_bit, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.a_bit=a_bit
        bias =True
        if w_bit ==32:
            self.conv = nn.Sequential(*[
                nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
                nn.ReLU()
            ])
        else:
            self.conv = nn.Sequential(*[
                quantize.Conv2d_Q(Cin, G, kSize, stride=1, padding=(kSize-1)//2, bias=bias, dilation=1, groups=1, w_bit=w_bit),
                nn.ReLU()
            ])
        
        
        self.quant = quantize.Quantization(bit=self.a_bit, qq_bit=qq_bit, finetune=False)

        
    def forward(self, x):
        if self.a_bit!=32:
            x1 = self.quant(x)
        else:
            x1 = x
        out = self.conv(x1)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers,  a_bit, w_bit, qq_bit, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G, a_bit, w_bit, qq_bit))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C,a_bit=args.quantize_a, w_bit=args.quantize_w, qq_bit=args.quantize_quantization)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        return self.UPNet(x)
