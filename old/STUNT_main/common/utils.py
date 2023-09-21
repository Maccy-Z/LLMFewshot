import os

import torch
import torch.optim as optim

from utils import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(P, model):
    params = model.parameters()
    optimizer = optim.Adam(params, lr=P.lr)
    return optimizer


def is_resume(P, model, optimizer):


    return is_best, start_step, best, acc


def load_model(P, model, logger=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log
    print(P.load_path)
    if P.load_path is not None:
        log_(f'Load model from {P.load_path}')
        checkpoint = torch.load(P.load_path)
        if P.rank != 0:
            model.__init_low_rank__(rank=P.rank)

        model.load_state_dict(checkpoint, strict=P.no_strict)
