#!/usr/bin/env python

import argparse
import os
import time
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default='3', help='Visible GPU id')
parser.add_argument('--model_version', default='DHGNN_v1', help='DHGNN model version, acceptable: DHGNN_v1, DHGNN_v2')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
import torch
import copy
import time
import random
from config import get_config
from datasets import source_select
from torch import nn
import torch.optim as optim
from models import model_select
import sklearn
from sklearn import neighbors
import numpy as np
from numpy import mean
from models.atten import ATTEN
from torch.autograd import Variable
import scipy.sparse as sp
import torch.nn.functional as F

from utils.construct_hypergraph import _edge_dict_to_H, _generate_G_from_H
from utils.plot import plot_embeddings
from news_pro import load_news_data

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, fts, lbls, idx_train, idx_val, edge_dict, G,
                   criterion, optimizer, scheduler, device,
                   num_epochs=25, print_freq=500):
    """
    gcn-style whole graph training
    :param model:
    :param fts:
    :param lbls:
    :param idx_train:
    :param idx_val:
    :param edge_dict:
    :param G: G for input HGNN layer
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param device:
    :param num_epochs:
    :param print_freq:
    :return: best model on validation set
    """
    since = time.time()

    state_dict_updates = 0          # number of epochs that updates state_dict

    model = model.cuda()

    model_wts_best_val_acc = copy.deepcopy(model.state_dict())
    model_wts_lowest_val_loss = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    loss_min = 100.0
    acc_epo = 0
    loss_epo = 0


    for epoch in range(num_epochs):
        epo = epoch

        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:


            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == 'train' else idx_val

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):


                outputs = model(ids=idx, feats=fts, edge_dict=edge_dict, G=G, ite=epo)
                #print(outputs)


                loss = criterion(outputs, lbls[idx]) * len(idx)
                _, preds = torch.max(outputs, 1)


                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss
            running_corrects += torch.sum(preds == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc

                model_wts_best_val_acc = copy.deepcopy(model.state_dict())

                acc_epo = epoch 
                state_dict_updates += 1

            if phase == 'val' and epoch_loss < loss_min:
                loss_min = epoch_loss

                model_wts_lowest_val_loss = copy.deepcopy(model.state_dict())

                loss_epo = epoch 
                state_dict_updates += 1


            if epoch % print_freq == 0 and phase == 'val':
                print(f'Best val Acc: {best_acc:4f}, Min val loss: {loss_min:4f}')
                print('-' * 20)
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'\nState dict updates {state_dict_updates}')
    print(f'Best val Acc: {best_acc:4f}')

    return (model_wts_best_val_acc, acc_epo), (model_wts_lowest_val_loss, loss_epo)


def test(model, best_model_wts, fts, lbls, n_category, idx_test, edge_dict, G, device, test_time = 1):
    """
    gcn-style whole graph test
    :param model_best:
    :param fts:
    :param lbls:
    :param idx_test:
    :param edge_dict:
    :param G: G for input HGNN layer
    :param device:
    :param test_time: test for several times and vote
    :return:
    """
    best_model_wts, epo = best_model_wts
    model = model.cuda()
    model.load_state_dict(best_model_wts)
    model.eval()

    running_corrects = 0.0

    outputs = torch.zeros(len(idx_test), n_category).cuda()

    for _ in range(test_time):

        with torch.no_grad():

            outputs += model(ids=idx_test, feats=fts, edge_dict=edge_dict, G=G, ite=epo)

    _, preds = torch.max(outputs, 1)

    running_corrects += torch.sum(preds == lbls.data[idx_test])
    test_acc = running_corrects.double() / len(idx_test)
    # lbls.data[idx_test] preds

    M = np.zeros((n_category,n_category))
    for i in range(len(idx_test)):
        M[preds[i]][lbls.data[idx_test][i]] +=1
    n = len(M)
    pre = []
    rec = []
    for i in range(n):
        rowsum, colsum = sum(M[i]), sum(M[r][i] for r in range(n))
        try:
            pre.append((M[i][i]/float(colsum)))
            rec.append((M[i][i]/float(colsum)))
        except ZeroDivisionError:
            pre.append(0)
            rec.append(0)
    w = np.ones(n_category)
    for i in range(n_category):
        w[i] = lbls.data.tolist().count(i)/len(lbls.data)
    pre_w = pre * w
    rec_w = rec * w
    print('precision_avg:%s'%mean(pre),'recall_avg:%s'%mean(rec))
    print('f1:%s'%((2*mean(pre)*mean(rec))/(mean(pre)+mean(rec))))
    print('Weight f1:%s' % ((2 * sum(pre_w) * sum(rec_w)) / (sum(pre_w) + sum(rec_w))))

    print('*' * 20)
    print(f'Test acc: {test_acc} @Epoch-{epo}')
    print('*' * 20)

    return test_acc, epo


def train_test_model(cfg):
    device = torch.device('cuda:0')
    # print(os.environ)

    source = source_select(cfg)
    print(f'Using {cfg["activate_dataset"]} dataset')
    fts, lbls, idx_train, idx_val, idx_test, n_category, _, edge_dict = source(cfg)
    ## new dataset loading
    # fts, lbls, idx_train, idx_val, idx_test, n_category, _, edge_dict = load_news_data();

    H = _edge_dict_to_H(edge_dict)   #2708*2708
    G = _generate_G_from_H(H)

    G = torch.Tensor(G).cuda()
    fts = torch.Tensor(fts).cuda()
    ################################
    a=1
    if a==1:
        lb = torch.Tensor(lbls).squeeze().long().cuda()
        mod = ATTEN(nfeat=fts.shape[1],
                      nhid=64,
                      nclass=n_category,  # Àà±ð×ÜÊý
                      dropout=0.6,
                      nheads=8,
                      alpha=0.2)
        # mod = torch.nn.DataParallel(mod, device_ids=device).cuda()
        optimizer = optim.Adam(mod.parameters(),
                               lr=0.005,
                               weight_decay=5e-4)
        mod.cuda()
        fts = fts.cuda()
        # fts, G, lb = Variable(fts), Variable(G), Variable(lb)
        #for  1000
        for i in range(768):  #768
            if i == 767:
                mod = ATTEN(nfeat=fts.shape[1],
                            nhid=64,
                            nclass=n_category,  # Àà±ð×ÜÊý
                            dropout=0.6,
                            nheads=8,
                            alpha=0.2,
                            flag=1)
            mod.cuda()
            mod.train()
            optimizer.zero_grad()
            output = mod(fts, G)
            output = output.cuda()
            lb = lb.cuda()
            loss_train = F.nll_loss(output, lb)
            loss_train.backward()
            optimizer.step()
            # mod.eval()  #***********************
            # output = mod(fts, G)
            # loss_val = F.nll_loss(output, lb)
            print('Epoch: {:04d}'.format(i + 1),
                  'loss_train: {:.4f}'.format(loss_train.data.item()),
                  # 'loss_val: {:.4f}'.format(loss_val.data.item())
                  )
    # print(output)
    #***********************************************

    lbls = torch.Tensor(lbls).squeeze().long().cuda()

    model = model_select(cfg['model'])\
        (dim_feat=fts.size(1),
        n_categories=n_category,
        k_structured=cfg['k_structured'],
        k_nearest=cfg['k_nearest'],
        k_cluster=cfg['k_cluster'],
        wu_knn=cfg['wu_knn'],
        wu_kmeans=cfg['wu_kmeans'],
        wu_struct=cfg['wu_struct'],
        clusters=cfg['clusters'],
        adjacent_centers=cfg['adjacent_centers'],
        n_layers=cfg['n_layers'],
        layer_spec=cfg['layer_spec'],
        dropout_rate=cfg['drop_out'],
        has_bias=cfg['has_bias'],
        alpha=cfg['alpha']
        )

    #initialize model
    state_dict = model.state_dict()
    for key in state_dict:
        if 'weight' in key:
            nn.init.xavier_uniform_(state_dict[key])
        elif 'bias' in key:
            state_dict[key] = state_dict[key].zero_()

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'],weight_decay=cfg['weight_decay'], eps=1e-20)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg['milestones'],
                                               gamma=cfg['gamma'])
    criterion = torch.nn.NLLLoss()


    # transductive learning mode
    model_wts_best_val_acc, model_wts_lowest_val_loss\
        = train(model, fts, lbls, idx_train, idx_val, edge_dict, G, criterion, optimizer, schedular, device,
                    cfg['max_epoch'], cfg['print_freq'])
    if idx_test is not None:
        print('**** Model of lowest val loss ****')
        test_acc_lvl, epo_lvl = test(model, model_wts_lowest_val_loss, fts, lbls, n_category, idx_test, edge_dict, G, device, cfg['test_time'])
        print('**** Model of best val acc ****')
        test_acc_bva, epo_bva = test(model, model_wts_best_val_acc, fts, lbls, n_category, idx_test, edge_dict, G, device, cfg['test_time'])
        # if test_acc_lvl>test_acc_bva:
        #     with open('result.txt','a') as r:
        #         r.write(str(cfg['max_epoch'])+':')
        #         r.write(str(test_acc_lvl)+';')
        #         r.write('\n')
        # else:
        #     with open('result.txt','a') as r:
        #         r.write(str(cfg['max_epoch'])+':')
        #         r.write(str(test_acc_bva)+';')
        #         r.write('\n')
        plot_embeddings(fts.tolist())
        return (test_acc_lvl, epo_lvl), (test_acc_bva, epo_bva)
    else:
        return None


if __name__ == '__main__':
    start_time = time.time()
    seed_num = 1000

    setup_seed(seed_num) 
    print('Using random seed: ', seed_num)

    cfg = get_config('config/config.yaml')
    cfg['model'] = args.model_version

    # k_cluster_list = [1,5,10,15,20,30,40,50,60,64,65,66,67,68,70,80,90]
    # for k_cluster in k_cluster_list:
    #     cfg['k-cluster'] = k_cluster
    #     train_test_model(cfg)
    # k_nearest_list = [1, 5, 10, 15, 20, 30, 40, 50, 60, 64, 65, 66, 67, 68, 70, 80, 90]
    # for k_nearest in k_nearest_list:
    #     cfg['k_nearest'] = k_nearest
    #     train_test_model(cfg)
    # alpha_list = [0,0.1,0.15,0.18,0.2,0.22,0.25,0.26,0.27,0.28,0.29,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # for alpha in alpha_list:
    #     cfg['alpha'] = alpha
    #     train_test_model(cfg)
    # for i in range(26):

    #     cfg['max_epoch'] = i
    #     train_test_model(cfg)
    train_test_model(cfg)

    end_time = time.time()
    run_time = end_time - start_time
    print('run time:%.2f s'%run_time)