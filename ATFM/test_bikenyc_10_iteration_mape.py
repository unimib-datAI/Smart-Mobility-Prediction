import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from fvcore.nn import FlopCountAnalysis

from model.spn import ModelAttentionWithTimeaware as Model
from dataset.dataset import DatasetFactory
import torchmetrics
from torchsummary import summary


def APE(input, target, mmn, m_factor):
   
    def inverse_transform(X, mmn):
        X = (X + 1.) / 2.
        X = 1. * X * (mmn.max - mmn.min) + mmn.min
        return X
    
    
    target = inverse_transform(target, mmn)
    input = inverse_transform(input, mmn)
    idx = target > 10
    return torch.sum(torch.abs((target[idx] - input[idx]) / target[idx])) * 100

def MAPE(input, target, mmn, m_factor):
    def denormalize(X, mmn):
        return X * (mmn._max - mmn._min) / 2.
    
    def inverse_transform(X, mmn):
        X = (X + 1.) / 2.
        X = 1. * X * (mmn.max - mmn.min) + mmn.min
        return X
        
    target = inverse_transform(target, mmn)
    input = inverse_transform(input, mmn)
    idx = target > 10

    #mape = torch.mean(torch.abs((target[idx] - input[idx]) / target[idx]))
    mean_abs_percentage_error = torchmetrics.MeanAbsolutePercentageError().to('cuda')
    error = mean_abs_percentage_error(input[idx],target[idx])*100
    return error

class DataConfiguration:
    # Data
    name = 'BikeNYC'
    portion = 1.  # portion of data

    len_close = 4
    len_period = 2
    len_trend = 0
    pad_forward_period = 0
    pad_back_period = 0
    pad_forward_trend = 0
    pad_back_trend = 0

    len_all_close = len_close * 1
    len_all_period = len_period * (1 + pad_back_period + pad_forward_period)
    len_all_trend = len_trend * (1 + pad_back_trend + pad_forward_trend)

    len_seq = len_all_close + len_all_period + len_all_trend
    cpt = [len_all_close, len_all_period, len_all_trend]

    interval_period = 1
    interval_trend = 7

    ext_flag = True
    ext_time_flag = True
    rm_incomplete_flag = True
    fourty_eight = False
    previous_meteorol = True

    ext_dim = 33
    dim_h = 16
    dim_w = 8

def test(dconf):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ds_factory = DatasetFactory(dconf)
    test_ds = ds_factory.get_test_dataset()

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=240,
        shuffle=False,
        num_workers=1
    )

    model = Model(dconf)
    
    for iteration in range(10):
        np.random.seed(iteration*18)
        torch.manual_seed(iteration*18)
        try:
            model.load_state_dict(torch.load(f'./logs/bikenyc_420/seed_{iteration*18}_bikenyc.model'))
        except:
            model = torch.load(f'./logs/bikenyc_420/seed_{iteration*18}_bikenyc.model')
        model = model.cuda()   
        print(len(test_loader.dataset))

        criterion = nn.MSELoss().cuda()
        model.eval()
        mape1 = []
        ape1 = []
        mmn = ds_factory.ds.mmn
        with torch.no_grad():
            for i, (X, X_ext, Y, Y_ext) in enumerate(test_loader, 0):
                X = X.cuda()
                X_ext = X_ext.cuda() 
                Y = Y.cuda() 
                Y_ext = Y_ext.cuda()

                #### FLOPS & PARAMETERS                
                #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
                #input = (X, X_ext,Y_ext)
                #flops = FlopCountAnalysis(model, input)
                #print(flops.total())

                h = model(X, X_ext, Y_ext)
                loss = criterion(h, Y)
                
                mape = MAPE(h, Y, ds_factory.ds.mmn, ds_factory.dataset.m_factor)
                ape = APE(h, Y, ds_factory.ds.mmn, ds_factory.dataset.m_factor)
                
                #mape1.append(MAPE(h, Y, ds_factory.ds.mmn, ds_factory.dataset.m_factor))
                #ape1.append(APE(h, Y, ds_factory.ds.mmn, ds_factory.dataset.m_factor))
        #print('len mape 1', len(mape1))
        #mape = torch.stack(mape1, dim=0).sum()/ len(mape1)
        #ape = torch.stack(ape1, dim=0).sum()
        print("[interaction %d] mape: %.4f ape: %.4f" % (iteration, mape, ape))


if __name__ == '__main__':
    dconf = DataConfiguration()

    test(dconf)