import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )
    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubicL')
        else:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('', '.png')
    """
    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark')
        self.dir_hr = os.path.join(self.apath, self.name, 'HR')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubicL', self.name)
            #self.dir_lr = os.path.join(self.apath, 'LR_feqeL', self.name)
            #self.dir_lr = os.path.join(self.apath, 'SR_feqeL', self.name)
        else:
            self.dir_lr = os.path.join(self.apath, self.name, 'LR_bicubic')
            #self.dir_lr = os.path.join(self.apath, 'LR_feqe', self.name)
        self.ext = ('', '.png')
    """

