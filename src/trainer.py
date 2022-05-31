import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

import numpy as np

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.optimizer.schedule() 
        self.loss.step()
        epoch = self.optimizer.get_last_epoch()
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        

        timer_data, timer_model = utility.timer(), utility.timer()
        
        # print(len(self.loader_train))
        # self.loader_train.dataset.set_scale(0) 
        for batch, (lr, hr, idx_scale,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            
            
            params = self.model.named_parameters()
            for name1, params1 in params:
                params1.requires_grad=True
            
            sr = self.model(lr, idx_scale) #0
            loss = self.loss(sr, hr)

            loss.backward()
            #loss.backward(retain_graph=True) # False

            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release(), 
                ))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        # self.optimizer.schedule() #commented

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        timer_test = utility.timer()
        
        ############################## get_#_params ####################
        n_params = 0
        
        for p in list(self.model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            n_params += nn
        self.ckp.write_log('Parameters: {:.1f}K'.format(n_params/(10**3)))
        

       
        if self.args.save_results:
            self.ckp.begin_background()

       
        ############################## TEST FOR TEST SET #############################
        for idx_data, d in enumerate(self.loader_test):
            if True:
                for idx_scale, scale in enumerate(self.scale):
                    d.dataset.set_scale(idx_scale)
                    tot_ssim =0
                    i=0
                    for lr, hr, filename in tqdm(d, ncols=80):
                        i+=1
                        if True:
                            lr, hr = self.prepare(lr, hr)

                            sr = self.model(lr, idx_scale)
                            # sr = self.model(lr)

                            sr = utility.quantize(sr, self.args.rgb_range)

                           
                            save_list = [sr]

                         
                            psnr, ssim = utility.calc_psnr( sr, hr, scale, self.args.rgb_range, dataset=d)
                            
                            tot_ssim += ssim
                            self.ckp.log[-1, idx_data, idx_scale] += psnr
                            
                            if self.args.save_gt:
                                save_list.extend([lr, hr])

                            if self.args.save_results:
                                self.ckp.save_results(d, filename[0], save_list, scale)
                        
                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} \t SSIM: {:.4f} \t(Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            tot_ssim/i,
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                
        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        torch.set_grad_enabled(True) 

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            #writer.close()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            #writer.close()
            return epoch >= self.args.epochs

