import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from model.spn import ModelAttentionWithTimeaware as Model
from dataset.dataset import DatasetFactory
from utils import weights_init, RMSE, MAPE

log_name = 'logs/bikenyc_'

seed = 777

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

class TrainConfiguration:
    learning_rate = 0.0003
    milestones = {
        0:0.0003, 
        50:0.0002, 
        150:0.0001
    }
    batch_size = 64
    max_epoch = 150

    grad_threshold = 0.1

    pretrained = None

def save_conf(log_name, conf_name, conf):
    with open(log_name, "a") as log_file:
        log_file.write('[{}]'.format(conf_name))
        log_file.write('\n')
        for key in dir(conf):
            if not key.startswith('__'):
                log_file.write('{}={}'.format(key, getattr(conf, key)))
                log_file.write('\n')
        log_file.write('\n\n')

def save_confs(log_name, confs):
    for name, conf in confs.items():
        save_conf(log_name, name, conf)

def run(dconf, tconf):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ds_factory = DatasetFactory(dconf)
    train_ds = ds_factory.get_train_dataset()
    test_ds = ds_factory.get_test_dataset()

    train_loader = DataLoader(
        dataset=train_ds, 
        batch_size=tconf.batch_size,
        shuffle=True,
        num_workers=1
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=tconf.batch_size,
        shuffle=False,
        num_workers=1
    )

    model = Model(dconf)
    if tconf.pretrained is not None:
        print('load pretrained model...')
        try:
            model.load_state_dict(torch.load(tconf.pretrained))
        except Exception as e:
            model = torch.load(tconf.pretrained)
    else:
        model.apply(weights_init)
    model = model.cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=tconf.learning_rate)

    log_dir = log_name+str(dconf.len_close)+str(dconf.len_period)+str(dconf.len_trend)+'/'
    writer = SummaryWriter(log_dir)
    
    confs = {
        'Data Config': dconf, 
        'Train Config': tconf, 
        'Model Config': model.mconf
    }
    save_confs(log_dir+'confs', confs)
    for iteration in range(10):
        np.random.seed(iteration*18)
        torch.manual_seed(iteration*18)
        step = 0
        best_test_err = 1000.0
        early_stopping = 0
        for epoch in range(tconf.max_epoch):
            if epoch in tconf.milestones:
                print('Set lr=',tconf.milestones[epoch])
                for param_group in optimizer.param_groups:
                    param_group["lr"] = tconf.milestones[epoch]

            model.train()
            for i, (X, X_ext, Y, Y_ext) in enumerate(train_loader, 0):
                X = X.cuda()
                X_ext = X_ext.cuda() 
                Y = Y.cuda() 
                Y_ext = Y_ext.cuda()

                optimizer.zero_grad()

                h = model(X, X_ext, Y_ext)
                loss = criterion(h, Y)
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), tconf.grad_threshold)
                optimizer.step()

                #if step % 10 == 0:
                #    rmse = RMSE(h, Y, ds_factory.ds.mmn, ds_factory.dataset.m_factor)
                #    mape = MAPE(h, Y, ds_factory.ds.mmn, ds_factory.dataset.m_factor)
                    #print("[epoch %d][%d/%d] mse: %.4f rmse: %.4f mape: %.4f" % (epoch, i+1, len(train_loader), loss.item(), rmse.item(), mape.item()))
                #    writer.add_scalar('mse', loss.item(), step)
                #    writer.add_scalar('rmse', rmse.item(), step)
                #    writer.add_scalar('mape', rmse.item(), step)
                step += 1

            model.eval()
            mse = 0.0
            mape = 0.0
            with torch.no_grad():
                for i, (X, X_ext, Y, Y_ext) in enumerate(test_loader, 0):
                    X = X.cuda()
                    X_ext = X_ext.cuda() 
                    Y = Y.cuda() 
                    Y_ext = Y_ext.cuda()
                    h = model(X, X_ext, Y_ext)
                    loss = criterion(h, Y)
                    mse += X.size()[0] * loss.item()
                    idx = Y > (1. * (10. - ds_factory.ds.mmn.min) / (ds_factory.ds.mmn.max - ds_factory.ds.mmn.min))
                    if torch.sum(idx) != 0:
                        mape += torch.sum(torch.abs(Y[idx] - h[idx])/Y[idx])
            mse /= ds_factory.ds.X_test.shape[0]
            mape /= ds_factory.ds.X_test.shape[0]
            mape = mape * (ds_factory.ds.mmn.max - ds_factory.ds.mmn.min) / 2.
            rmse = math.sqrt(mse) * (ds_factory.ds.mmn.max - ds_factory.ds.mmn.min) / 2. * ds_factory.dataset.m_factor
            #print("[epoch %d] test_rmse: %.4f test_mape: %.4f\n" % (epoch, rmse, mape))
            #writer.add_scalar('test_rmse', rmse, epoch, iteration*18)
            #writer.add_scalar('test_mape', mape, epoch, iteration*18)

            save_perform = mse
            if save_perform <= best_test_err:
                early_stopping = 0
                best_test_err = save_perform
                torch.save(model.state_dict(), log_dir+f'seed_{iteration*18}_bikenyc.model')
            else:
                early_stopping += 1
            if early_stopping > 10 or epoch == (tconf.max_epoch-1):
                torch.save(model.state_dict(), log_dir+f'seed_{iteration*18}_bikenyc.model')
                break
        


        #################################
        # Testing
        #################################

        model.load_state_dict(torch.load(log_dir+f'seed_{iteration*18}_bikenyc.model'))
        model.eval()
        mse = 0.0
        mape = 0.0
        with torch.no_grad():
            for i, (X, X_ext, Y, Y_ext) in enumerate(test_loader, 0):
                X = X.cuda()
                X_ext = X_ext.cuda() 
                Y = Y.cuda() 
                Y_ext = Y_ext.cuda()
                h = model(X, X_ext, Y_ext)
                loss = criterion(h, Y)
                mse += X.size()[0] * loss.item()
                idx = Y > (1. * (10. - ds_factory.ds.mmn.min) / (ds_factory.ds.mmn.max - ds_factory.ds.mmn.min))
                if torch.sum(idx) != 0:
                    mape += torch.sum(torch.abs(Y[idx] - h[idx])/Y[idx])
            mse /= ds_factory.ds.X_test.shape[0]
            mape /= ds_factory.ds.X_test.shape[0]
            mape = mape * (ds_factory.ds.mmn.max - ds_factory.ds.mmn.min) / 2.
            rmse = math.sqrt(mse) * (ds_factory.ds.mmn.max - ds_factory.ds.mmn.min) / 2. * ds_factory.dataset.m_factor
        print("[interaction %d] test_rmse: %.4f test_mape: %.4f\n" % (iteration, rmse, mape))
        # save to csv
        csv_name = os.path.join('results', f'ATFN_bikenyc.csv')
        if not os.path.isfile(csv_name):
            if os.path.isdir('results') is False:
                os.mkdir('results')
            with open(csv_name, 'a', encoding="utf-8") as file:
                file.write('iteration,'
                           'rmse_tot,'
                           'mape_tot'
                           )
                file.write("\n")
                file.close()
        with open(csv_name, 'a', encoding="utf-8") as file:
            file.write(f'{iteration},{rmse},{mape}')
            file.write("\n")
            file.close()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    dconf = DataConfiguration()
    tconf = TrainConfiguration()

    run(dconf, tconf)
