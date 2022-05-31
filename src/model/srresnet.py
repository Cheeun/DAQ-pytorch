from model import common_srresnet as common
import torch
import torch.nn as nn
import math
def make_model(args, parent=False):
    return SRResNet(args)

class SRResNet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SRResNet, self).__init__()

        a_bit = args.quantize_a
        w_bit = args.quantize_w
        qq_bit = args.quantize_quantization

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        m_head = [conv(args.n_colors, n_feats, kernel_size=9, bias=False)]
        # m_head.append(nn.LeakyReLU(0.2, inplace=True))
        m_head.append(nn.PReLU())

        act = nn.ReLU(True)
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, a_bit=a_bit, w_bit=w_bit, qq_bit=qq_bit, finetune=args.finetune, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append( conv(n_feats, n_feats, kernel_size, bias=False))
        m_body.append( nn.BatchNorm2d(n_feats))

        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size=9, bias=False)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
