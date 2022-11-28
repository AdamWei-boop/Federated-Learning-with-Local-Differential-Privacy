# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 20:37:06 2019

@author: WEIKANG
"""

import numpy as np
import copy
import torch
import math
from Calculate import f_zero


def Privacy_account(args, threshold_epochs, noise_list, iter):
    q_s = args.num_Chosenusers/args.num_users
    delta_s = 2*args.lr*args.clipthr/args.num_items_train
    if args.dp_mechanism != 'CRD':
        noise_scale = delta_s*np.sqrt(2*q_s*threshold_epochs*np.log(1/args.delta))/args.privacy_budget
    elif args.dp_mechanism == 'CRD':
        noise_sum = 0
        for i in range(len(noise_list)):
            noise_sum += pow(1/noise_list[i],2)
        if pow(args.privacy_budget/delta_s,2)/(2*q_s*np.log(1/args.delta))>noise_sum:
            noise_scale = np.sqrt((threshold_epochs-iter)/(pow(args.privacy_budget/delta_s,2)/(2*q_s*np.log(1/args.delta))-noise_sum))
        else:
            noise_scale = noise_list[-1]
    return noise_scale


def Adjust_T(args, loss_avg_list, threshold_epochs_list, iter):
    if loss_avg_list[iter-1]-loss_avg_list[iter-2]>=0:
        threshold_epochs = copy.deepcopy(math.floor( math.ceil(args.dec_cons*threshold_epochs_list[-1])))
        # print('\nThreshold epochs:', threshold_epochs_list)
    else:
        threshold_epochs = threshold_epochs_list[-1]
    return threshold_epochs

