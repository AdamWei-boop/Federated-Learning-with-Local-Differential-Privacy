# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:19:31 2019

@author: WEIKANG
"""
import torch
import numpy as np
import copy
import math

def inner_product(params_a, params_b):
    sum = 0
    for i in params_a.keys():
        sum += np.sum(np.multiply(params_a[i].cpu().numpy(),\
                params_b[i].cpu().numpy()))     
    return sum


def avg_grads(g):
    grad_avg = copy.deepcopy(g[0])
    for k in grad_avg.keys():
        for i in range(1, len(g)):
            grad_avg[k] += g[i][k]
        grad_avg[k] = torch.div(grad_avg[k], len(g))
    return grad_avg

def calculate_grads(args, w_before, w_new):
    grads = copy.deepcopy(w_before)
    for k in grads.keys():
        grads[k] =(w_before[k]-w_new[k]) * 1.0 / args.lr
    return grads


def f_zero(args, f, num_iter):
    x0 = 0
    x1 = args.max_epochs
    if f(x0)*f(x1)>=0:
        if abs(f(x0))>abs(f(x1)):
            x0 = copy.deepcopy(x1)
    else:
        y = copy.deepcopy(args.max_epochs)
        for i in range(100):
            if f(x0)*f(x1)<0:
                y = copy.deepcopy(x1)
                x1 = copy.deepcopy((x0+x1)/2)
            else:
                x1 = copy.deepcopy(y)
                x0 = copy.deepcopy((x0+x1)/2)
            if abs(x0-x1)<0.01:
                break
        if (x0+num_iter) > args.max_epochs:
            x0 = copy.deepcopy(args.max_epochs)    
    return x0

def get_l2_norm(args, params_a):
    sum = 0
    if args.gpu != -1:
        tmp_a = np.array([v.detach().cpu().numpy() for v in params_a])
    else:
        tmp_a = np.array([v.detach().numpy() for v in params_a])
    a = []
    for i in tmp_a:
        x = i.flatten()
        for k in x:
            a.append(k)
    for i in range(len(a)):
        sum += (a[i] - 0) ** 2
    norm = np.sqrt(sum)
    return norm

def get_1_norm(params_a):
    sum = 0
    if isinstance(params_a,np.ndarray) == True:
        sum += pow(np.linalg.norm(params_a, ord=2),2) 
    else:
        for i in params_a.keys():
            if len(params_a[i]) == 1:
                sum += pow(np.linalg.norm(params_a[i].cpu().numpy(), ord=2),2)
            else:
                a = copy.deepcopy(params_a[i].cpu().numpy())
                for j in a:
                    x = copy.deepcopy(j.flatten())
                    sum += pow(np.linalg.norm(x, ord=2),2)                  
    norm = np.sqrt(sum)
    return norm

def get_2_norm(params_a, params_b):
    sum = 0
    for i in params_a.keys():
        if len(params_a[i]) == 1:
            sum += pow(np.linalg.norm(params_a[i].cpu().numpy()-\
                params_b[i].cpu().numpy(), ord=2),2)
        else:
            a = copy.deepcopy(params_a[i].cpu().numpy())
            b = copy.deepcopy(params_b[i].cpu().numpy())
            x = []
            y = []
            for j in a:
                x.append(copy.deepcopy(j.flatten()))
            for k in b:          
                y.append(copy.deepcopy(k.flatten()))
            for m in range(len(x)):
                sum += pow(np.linalg.norm(x[m]-y[m], ord=2),2)            
    norm = np.sqrt(sum)
    return norm

def para_estimate(args, list_loss, loss_locals, w_glob_before, w_locals_before,\
                  w_locals, w_glob):
    Lipz_c = []
    Lipz_s = []
    beta = []
    delta = []
    norm_grads_locals = []
    Grads_locals = copy.deepcopy(w_locals)
    for idx in range(args.num_Chosenusers):
        ### Calculate▽F_i(w(t))=[w(t)-w_i(t)]/lr ###
        Grads_locals[idx] = copy.deepcopy(calculate_grads(args, w_glob, w_locals[idx]))
    ### Calculate▽F(w(t)) ###    
    Grads_glob =  copy.deepcopy(avg_grads(Grads_locals))    
    for idx in range(args.num_Chosenusers):
        ### Calculate ||w(t-1)-w(t)|| ###
        diff_weights_glob = copy.deepcopy(get_2_norm(w_glob_before, w_glob))
        ### Calculate ||▽F_i(w(t-1))-▽F_i(w(t))|| ###
        diff_grads = copy.deepcopy(get_2_norm(calculate_grads(args, w_glob_before, \
                w_locals_before[idx]), calculate_grads(args, w_glob, w_locals[idx])))
        ### Calculate ||w(t)-w_i(t)|| ###
        diff_weights_locals = copy.deepcopy(get_2_norm(w_glob, w_locals[idx]))
        ### Calculate ||▽F(w(t))-▽F_i(w(t))|| ###
        Grads_variance = copy.deepcopy(get_2_norm(Grads_glob, Grads_locals[idx]))
        ### Calculate ||▽F(w(t))|| ###
        norm_grads_glob = copy.deepcopy(get_1_norm(Grads_glob))
        ### Calculate ||▽F_i(w(t))|| ###
        norm_grads_locals.append(copy.deepcopy(get_1_norm(Grads_locals[idx])))
        ### Calculate Lipz_s=||▽F_i(w(t-1))-▽F_i(w(t))||/||w(t-1)-w(t)|| ###
        Lipz_s.append(copy.deepcopy(diff_grads/diff_weights_glob))
        ### Calculate Lipz_c=||F_i(w(t))-F_i(w_i(t))||/||w(t)-w_i(t)|| ###
        Lipz_c.append(copy.deepcopy(abs(list_loss[idx]-loss_locals[idx])/diff_weights_locals))
        ### Calculate delta= ||▽F(w(t))-▽F_i(w(t))||###
        delta.append(copy.deepcopy(Grads_variance))
    beta = copy.deepcopy(np.sqrt(sum(c*c for c in norm_grads_locals)/args.num_Chosenusers)/norm_grads_glob)
    return Lipz_s, Lipz_c, delta, beta, Grads_glob, Grads_locals, norm_grads_glob, norm_grads_locals
