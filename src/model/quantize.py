import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np


uniform_steps = {0:2.0, 1: 1.0, 2: 0.5, 3:0.25, 4:0.125}
laplacian_steps = {0:2.0,1: 1.414, 2: 1.087, 3:0.731, 4:0.456}
gamma_steps = {0:2.0,1: 1.154, 2: 1.060, 3:0.796, 4:0.540}

gaussian_steps = {0:2.0, 1: 1.596, 2: 0.996, 3: 0.586, 4: 0.335, 5:0.188, 6:0.104, 7:0.057, 8:0.031, 15:(1/8), 16:(1/8)}

def gaussian_steps_func(b):
    s =torch.zeros(b.size()).cuda()
    for i in range(9):
        s += (b==i)*gaussian_steps[i]
    return s



class Quantization(nn.Module):
    def __init__(self, bit, qq_bit, finetune=False):
        super().__init__()
        self.a_bit = bit
        self.qq_bit = qq_bit
        if finetune:
            self.step= 1.596 # for 2 bit finetuning
            self.sig_step = 0.057 # 

        else:
            self.step = gaussian_steps_func(torch.tensor(self.a_bit-1))
            self.sig_step = gaussian_steps_func(torch.tensor(self.qq_bit-1))



    def forward(self, x ):
        a_bit = self.a_bit 
        if x.min() == 0 :
            ### After ReLU

            x_num = torch.sum(x>0,(2,3),True).float()+1 
            mu_gt = torch.sum(x,(2,3),True)/x_num
            x2_mean = torch.sum(x**2,(2,3),True)/x_num
            sigma_gt = (x2_mean-mu_gt**2)**0.5
            mu,sigma = mu_gt.detach(), sigma_gt.detach()

            if self.qq_bit!=32:
                mu_sigma = torch.mean(sigma,1,True).detach()
                sig_sigma = torch.std(sigma,1,True).view(sigma.size(0),1,1,1).detach()
                
                
                step = self.sig_step* sig_sigma
                thr = (2**self.qq_bit/2-0.5)*step
                step = step + (step==0).detach()*(-1)
                
                sig_c = sigma - mu_sigma

                y_sig = ((torch.round(sig_c/(step)+0.5)-0.5) * (step))*(step>0)
                y_sig = torch.min(y_sig, thr)
                y_sig = torch.max(y_sig, -thr)
                quantized_sig = y_sig + mu_sigma
                # int quantization

                sigma = quantized_sig

            
            lvls = 2 ** a_bit / 2
            step = self.step* sigma
            thr = (lvls-0.5)*step
            step = step + (step==0).detach()*(-1)

            x_c = x - thr

            y = ((torch.round(x_c/(step)+0.5)-0.5) * (step))*(step>0)
            
            # y = torch.min(y, thr + torch.max(thr-mu,thr*0))
            # y = torch.max(y, torch.max(-mu,-thr))
            quantized_x = y + thr

        else:
            mu_gt = torch.mean(x,(2,3),True)
            sigma_gt = torch.std(x,(2,3),True).view(x.size(0),x.size(1),1,1)
            # mu_gt = torch.mean(x,(1,2,3),True)
            # sigma_gt = torch.std(x,(1,2,3),True).view(x.size(0),1,1,1)
            mu,sigma = mu_gt.detach(), sigma_gt.detach()

            if self.qq_bit!=32:
                mu_sigma = torch.mean(sigma,1,True).detach()
                sig_sigma = torch.std(sigma,1,True).view(sigma.size(0),1,1,1).detach()

                sig_c = sigma - mu_sigma
                step = self.sig_step* sig_sigma
                thr = (2**self.qq_bit/2-0.5)*step
                step = step + (step==0).detach()*(-1)
                y_sig = ((torch.round(sig_c/(step)+0.5)-0.5) * (step))*(step>0)
                y_sig = torch.min(y_sig, thr)
                y_sig = torch.max(y_sig, -thr)
                quantized_sig = y_sig + mu_sigma
                sigma = quantized_sig

            x_c = x - mu
            lvls = 2 ** a_bit / 2
            step = self.step *sigma
            thr = (lvls-0.5)*step
            step = step + (step==0).detach()*(-1)
            
            y = ((torch.round(x_c/(step)+0.5)-0.5) * (step))*(step>0)
            
            y = torch.min(y, thr)
            y = torch.max(y, -thr)
            quantized_x = y + mu

        return quantized_x.detach() + x - x.detach()


        

class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                padding=1, bias=False, dilation=1, groups=1, w_bit=32, finetune=False):
        super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias, dilation, groups)
        
        self.w_bit = w_bit 

        if finetune:
            self.step= 1.596 # for 2 bit finetuning
        else:
            self.step = gaussian_steps[self.w_bit-1]
        
      

    def forward(self,x):
        # mu = torch.mean(self.weight,(0,1,2,3),True)
        # mu = torch.mean(self.weight,(0,2,3),True)
        # mu = torch.mean(self.weight,(2,3),True)
        mu = 0
        sigma = torch.std(self.weight,(0,1,2,3),True).view(1,1,1,1)
        # input channel
        # sigma = torch.std(self.weight,(0,2,3),True).view(1,self.weight.size(1),1,1)
        # output channel (filter)
        # sigma = torch.std(self.weight,(1,2,3),True).view(self.weight.size(0),1,1,1)
        # kernel
        # sigma = torch.std(self.weight,(2,3),True).view(self.weight.size(0),self.weight.size(1),1,1)

        w_bit = self.w_bit
        step = self.step
        
        
        w_z = (self.weight - mu)
        step = self.step * sigma
        lvls = (2 ** w_bit / 2)* (w_bit>0)
        thr = (lvls-0.5)*step*(w_bit>0)
        step = step + (step==0).detach()*(-1)
            
        y = ((torch.round(w_z/step+0.5)-0.5) * step)*(step>0)
        y = torch.min(y, thr+y*0)
        y = torch.max(y, -thr+y*0)
        
        w_q = y + mu

        self.dilation= (1,1) 
        
        return F.conv2d(x, self.weight - self.weight.detach()+  w_q.detach(), self.bias, self.stride, self.padding, self.dilation, self.groups)

