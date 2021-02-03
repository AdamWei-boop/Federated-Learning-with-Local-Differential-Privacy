# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 10:59:16 2019

@author: WEIKANG
"""

import numpy as np
import copy
import torch
import random
from Calculate import get_1_norm, get_2_norm, inner_product, avg_grads

def noise_add(args, noise_scale, w):
    w_noise = copy.deepcopy(w)
    if isinstance(w[0],np.ndarray) == True:
        noise = np.random.normal(0,noise_scale,w.size())
        w_noise = w_noise + noise
    else:
        for k in range(len(w)):
            for i in w[k].keys():
               noise = np.random.normal(0,noise_scale,w[k][i].size())
               if args.gpu != -1:
                   noise = torch.from_numpy(noise).float().cuda()
               else:
                   noise = torch.from_numpy(noise).float()
               w_noise[k][i] = w_noise[k][i] + noise
    return w_noise

def users_sampling(args, w, chosenUsers):
    if args.num_chosenUsers < args.num_users:
        w_locals = []
        for i in range(len(chosenUsers)):
            w_locals.append(w[chosenUsers[i]])
    else:
        w_locals = copy.deepcopy(w)
    return w_locals

def clipping(args, w):
    if get_1_norm(w) > args.clipthr:
        w_local = copy.deepcopy(w)
        for i in w.keys():
            w_local[i]=copy.deepcopy(w[i]*args.clipthr/get_1_norm(w))    
    else:
        w_local = copy.deepcopy(w)
    return w_local