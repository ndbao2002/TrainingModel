import torch
from torch import nn
from torchvision import models

import numpy as np
import math
import logging
import os
import shutil
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

        
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)
    
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)

    
def calc_psnr_and_ssim_torch_metric(hr, sr):
    sr = sr * 255
    hr = hr * 255
    if (sr.size() != hr.size()):
        h_min = min(sr.size(2), hr.size(2))
        w_min = min(sr.size(3), hr.size(3))
        sr = sr[:, :, :h_min, :w_min]
        hr = hr[:, :, :h_min, :w_min]

    sr = sr.round().to(torch.float64).cpu()
    hr = hr.round().to(torch.float64).cpu()

    sr[:,0,:,:] = sr[:,0,:,:] * 65.738/256.0
    sr[:,1,:,:] = sr[:,1,:,:] * 129.057/256.0
    sr[:,2,:,:] = sr[:,2,:,:] * 25.064/256.0
    sr = sr.sum(dim=1, keepdim=True) + 16.0

    hr[:,0,:,:] = hr[:,0,:,:] * 65.738/256.0
    hr[:,1,:,:] = hr[:,1,:,:] * 129.057/256.0
    hr[:,2,:,:] = hr[:,2,:,:] * 25.064/256.0
    hr = hr.sum(dim=1, keepdim=True) + 16.0

    ssim_cal = StructuralSimilarityIndexMeasure(data_range=255)
    psnr_cal = PeakSignalNoiseRatio(data_range=255)

    return psnr_cal(sr, hr), ssim_cal(sr, hr)

class Logger(object):
    def __init__(self, log_file_name, logger_name, log_level=logging.DEBUG):
        ### create a logger
        self.__logger = logging.getLogger(logger_name)

        ### set the log level
        self.__logger.setLevel(log_level)

        ### create a handler to write log file
        file_handler = logging.FileHandler(log_file_name)

        ### create a handler to print on console
        console_handler = logging.StreamHandler()

        ### define the output format of handlers
        formatter = logging.Formatter('[%(asctime)s] - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        ### add handler to logger
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger
    
def mkExpDir(args):
    if (os.path.exists(args.save)):
        if (not args.reset):
            raise SystemExit('Error: save_dir "' + args.save + '" already exists! Please set --reset True to delete the folder.')
        else:
            shutil.rmtree(args.save)

    os.makedirs(args.save)
    # os.makedirs(os.path.join(args.save, 'img'))
    os.makedirs(os.path.join(args.save, 'model'))
    os.mkdir(os.path.join(args.save, 'model_eval'))

    args_file = open(os.path.join(args.save, 'args.txt'), 'w')
    for k, v in vars(args).items():
        args_file.write(k.rjust(30,' ') + '\t' + str(v) + '\n')

    _logger = Logger(log_file_name=os.path.join(args.save, f'{args.model}.log'),
        logger_name=args.model).get_log()

    return _logger

def save_model(full_save, model_, optimizer_, scheduler_, count_, epoch_, log_loss_, eval_, file_name_):
    if full_save:
        data = {
            'model': model_.state_dict(),
            'opt': optimizer_.state_dict(),
            'scheduler': scheduler_.state_dict(),
            'step': count_,
            'epoch': epoch_,
            'loss': log_loss_,
            'eval': eval_,
        }

    else:
        data = {
            'model': model_.state_dict(),
            'step': count_,
            'epoch': epoch_,
            'loss': log_loss_,
            'eval': eval_,
        }
    torch.save(data, file_name_)