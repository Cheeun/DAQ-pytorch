import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer

import torch
import torch.nn as nn

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


if checkpoint.ok:
    loader = data.Data(args)
    _model = model.Model(args, checkpoint)
    _loss = loss.Loss(args, checkpoint) if not args.test_only else None
    
    t = Trainer(args, loader, _model, _loss, checkpoint)
    
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()
