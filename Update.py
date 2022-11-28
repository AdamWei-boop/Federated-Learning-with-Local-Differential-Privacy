#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn import metrics
import copy
from opacus import PrivacyEngine

import warnings

warnings.filterwarnings("ignore")
#matplotlib.use('Agg')

#matplotlib.use('Agg')

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        image, label = self.dataset[int(self.idxs[item])]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, tb):
        self.args = args
        self.tb = tb
        self.loss_func = nn.CrossEntropyLoss()
        # self.loss_func = nn.NLLLoss()
        self.ldr_train, self.ldr_test = self.train_val_test(dataset, list(idxs))          

    def train_val_test(self, dataset, idxs):
        # split train, and test
        idxs_train = idxs
        if (self.args.dataset == 'mnist') or (self.args.dataset == 'cifar'):
            idxs_test = idxs
            train = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
            #val = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val)/10), shuffle=True)
            test = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test)), shuffle=False)
        else:
            train = self.args.dataset_train[idxs]
            test = self.args.dataset_test[idxs]
        return train, test

    def update_weights(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0)
        
        # use this module to clip gradients
        privacy_engine = PrivacyEngine(
            net,
            sample_rate=0.01,
            alphas=list(np.arange(1.01, 20.0, 0.02)),
            noise_multiplier=0.0,
            max_grad_norm=self.args.clipthr,
            secure_rng = False,
            target_epsilon=2,
            target_delta = 1e-5,
            epochs = 1,
            )
        privacy_engine.attach(optimizer) 

        epoch_loss = []
        epoch_acc = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if self.args.gpu != -1:
                    images, labels = images.cuda(), labels.cuda()          
                    images, labels = autograd.Variable(images),\
                                    autograd.Variable(labels)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.gpu == -1:
                    loss = loss.cpu()
                self.tb.add_scalar('loss', loss.data.item())
                batch_loss.append(loss.data.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            acc, _, = self.test(net)
            epoch_acc.append(acc)
            if iter == 0:
                w_1st_ep = copy.deepcopy(net.state_dict())
        avg_loss = sum(epoch_loss)/len(epoch_loss)
        avg_acc = sum(epoch_acc)/len(epoch_acc)
        w = net.state_dict()         
        return w_1st_ep, w, avg_loss ,avg_acc               

    def test(self, net):
        loss = 0
        log_probs = []
        labels = []
        for batch_idx, (images, labels) in enumerate(self.ldr_test):
            if self.args.gpu != -1:
                images, labels = images.cuda(), labels.cuda()
            images, labels = autograd.Variable(images), autograd.Variable(labels)
            net = net.float()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            if self.args.gpu == -1:
                loss = loss.cpu()
                log_probs = log_probs.cpu()
                labels = labels.cpu()
        y_pred = np.argmax(log_probs.data, axis=1)
        acc = metrics.accuracy_score(y_true=labels.data, y_pred=y_pred)
        loss = loss.data.item()         
        return acc, loss