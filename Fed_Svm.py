import argparse
import time
import copy
import numpy as np
import random
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import RepeatedKFold

def load_dat(filepath, minmax=None, normalize=False, bias_term=True):
    """ load a dat file

    args:
    minmax: tuple(min, max), dersired range of transformed data
    normalize: boolean, normalize samples individually to unit norm if True
    bias_term: boolean, add a dummy column of 1s
    """
    lines = np.loadtxt(filepath)
    labels = lines[:, -1]
    features = lines[:, :-1]

    N, dim = features.shape

    if minmax is not None:
        minmax = MinMaxScaler(feature_range=minmax, copy=False)
        minmax.fit_transform(features)

    if normalize:
        # make sure each entry's L2 norm is 1
        normalizer = Normalizer(copy=False)
        normalizer.fit_transform(features)

    if bias_term:
        X = np.hstack([np.ones(shape=(N, 1)), features])
    else:
        X = features

    return X, labels

def svm_grad(w, X, y, clip=-1):
    y2d = np.atleast_2d(y)

    ywx = y * np.dot(X, w)
    loc = ywx < 1
    per_grad = -1.0 * y2d[:, loc].T * X[loc]

    if clip > 0:
        norm = np.linalg.norm(per_grad, axis=1)
        to_clip = norm > clip
        per_grad[to_clip, :] = ((clip * per_grad[to_clip])
                                / np.atleast_2d(norm[to_clip]).T)
        grad = np.sum(per_grad, axis=0)
    else:
        grad = np.sum(per_grad, axis=0)

    return grad

def svm_loss(w, X, y, clip=-1):
    is_1d = w.ndim == 1

    w = np.atleast_2d(w)
    y = np.atleast_2d(y)

    wx = np.dot(X, w.T)
    obj = 1.0 - (y.T * wx)
    obj[obj < 0] = 0

    # clipping
    if clip > 0:
        obj[obj > clip] = clip

    # reg = lmbda * np.sum(np.square(w[:, 1:]), axis=1)
    loss = np.sum(obj, axis=0)

    # loss = hinge + reg

    if is_1d:
        loss = np.asscalar(loss)

    return loss

def svm_test(w, X, y):
    is_1d = w.ndim == 1

    N = X.shape[0]
    w2d = np.atleast_2d(w)
    y2d = np.atleast_2d(y)

    wx = np.dot(X, w2d.T)
    sign = y2d.T * wx

    cnt = np.count_nonzero(sign > 0, axis=0)

    if is_1d:
        cnt = np.squeeze(cnt)

    return cnt / float(N)

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

def clipping(w, clipthr):
    if get_1_norm(w) > clipthr:
        w_local = copy.deepcopy(w)
        for i in w.keys():
            w_local[i]=copy.deepcopy(w[i]*clipthr/get_1_norm(w))    
    else:
        w_local = copy.deepcopy(w)
    return w_local

def noise_add(w, noise_scale,dim):
    w_noise = copy.deepcopy(w)
    noise = np.random.normal(0, noise_scale ,dim)
    noise = np.clip(noise,-3*noise_scale,3*noise_scale)
    w_noise = w_noise + noise
    return w_noise


def main(args):
    fpath = "./dataset/{0}.dat".format(args.dname)
    X, y = load_dat(fpath, minmax=(0, 1), normalize=False, bias_term=True)
    N, dim = X.shape
    y[y < 1] = -1

    grad_clip = args.grad_clip
    set_num_epochs = args.num_epochs
    reg_coeff = args.reg_coeff
    step_size = args.step_size 
    
    delta = args.delta
    set_privacy_budget = args.set_privacy_budget
    clipthr = args.clipthr    
    
    num_experiments = args.num_experiments

    num_users = args.num_users
    num_Chosenusers = args.num_Chosenusers
    num_train = args.num_train
    
    for eps in range(len(set_privacy_budget)):
        privacy_budget = copy.deepcopy(set_privacy_budget[eps])
        print('Privacy budget:{}, Clients number:{},Chosen clients:{}, Dataset size:{}, Experiment times:{}\n'.format(privacy_budget,\
              num_users, num_Chosenusers, num_train, num_experiments))
        epo_acc, epo_obj = [], [0]
        for j in range(len(set_num_epochs)):
            num_epochs = copy.deepcopy(set_num_epochs[j])
            avg_acc, avg_obj =[], []
            q_s = num_Chosenusers/num_users
            if privacy_budget>10000:
                noise_scale = 0
            else:
                noise_scale = 2*clipthr*np.sqrt(2*q_s*num_epochs*np.log(1/delta))/(privacy_budget*num_train)
            # noise_scale = 0
            for num_exper in range(num_experiments):
                acc_test = np.zeros(num_users)
                obj_test = np.zeros(num_users)
                avg_acc_test, avg_obj_test = [], []
                acc_train = np.zeros(num_users)
                obj_train = np.zeros(num_users)
                avg_acc_train, avg_obj_train = [], []
                sol_glob = copy.deepcopy(np.zeros(dim))                
                for i in range(num_epochs):
                    sol_locals = []
                    if  num_Chosenusers < num_users:
                        chosenUsers = random.sample(range(1,num_users),num_Chosenusers)
                        chosenUsers.sort()
                    else:
                        chosenUsers = range(num_users)
                    # print("\nChosen users:", chosenUsers)                     
                    for k in chosenUsers:
                        train_X, train_y = X[k*num_train:(k+1)*num_train,:], y[k*num_train:(k+1)*num_train]
                        test_X, test_y = X[num_users*num_train:,:], y[num_users*num_train:]
                    
                        n_train = train_X.shape[0]
                        n_test = test_X.shape[0]
                        N, dim = train_X.shape
                        sol = copy.deepcopy(sol_glob) 
                        if args.batch_size > 0:
                            # build a mini-batch
                            idx_len = int(np.ceil(N/args.batch_size))
                            for batch_idx in range(idx_len):
                                mini_X = X[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size, :]
                                mini_y = y[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
                                grad = svm_grad(sol, mini_X, mini_y, grad_clip)
                                if reg_coeff > 0:
                                    grad += reg_coeff * sol                
                                sol += - step_size * grad
                            # rand_idx = np.random.choice(N, size=args.batch_size, replace=False)
                        else:
                            mini_X = X
                            mini_y = y
                            grad = svm_grad(sol, mini_X, mini_y, grad_clip)
                            if reg_coeff > 0:
                                grad += reg_coeff * sol                
                            sol += - step_size * grad
                        sol_locals.append(sol)
                        obj_train[k] = svm_loss(sol, train_X, train_y) / n_train
                        acc_train[k] = svm_test(sol, train_X, train_y) * 100.0   
                        
                    sol_glob = np.zeros(dim)
                    for k in range(len(chosenUsers)):
                        ### Clipping  ###            
                        sol_locals[k] = copy.deepcopy(clipping(sol_locals[k], clipthr))
                        # print('\nLocal parameters' ,w_locals[i])
                        ### Add noise  ###                
                        sol_locals[k] = copy.deepcopy(noise_add(sol_locals[k], noise_scale, dim))                                        
                        sol_glob += sol_locals[k]
                    sol_glob = sol_glob/len(chosenUsers)
                    
                    obj_test = svm_loss(sol_glob, test_X, test_y) / n_test
                    acc_test = svm_test(sol_glob, test_X, test_y) * 100.0 
                    
                    avg_acc_train.append(sum(acc_train)/len(acc_train))    
                    avg_obj_train.append(sum(obj_train)/len(obj_train))        
                    avg_acc_test.append(acc_test)    
                    avg_obj_test.append(obj_test)
                    
                avg_acc.append(avg_acc_test[-1])
                avg_obj.append(avg_obj_test[-1])
            epo_obj.append(sum(avg_obj)/len(avg_obj)) 
            epo_acc.append(sum(avg_acc)/len(avg_acc))
            # print('*' * 20,f'Epoch[{i+1}/{num_epochs}]','*' * 20)               
            print(f'loss: {sum(avg_obj)/len(avg_obj):.6f}, acc: {sum(avg_acc)/len(avg_acc):.6f}, STD: {noise_scale}')
            if (epo_obj[-1]+epo_obj[-2])/2>epo_obj[1]:
                break
        print('Total loss:{}\n'.format(epo_obj))  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='adaptive sgd')
    # parser.add_argument('--dname', help='IPUMS-US')
    parser.add_argument('--dname', default='ADULT')
    parser.add_argument('--rep', type=int, default=1)
    parser.add_argument('--step_size', type=float, default=0.001)
    parser.add_argument('--grad_clip', type=float, default=3.0)
    parser.add_argument('--num_epochs', type=list, default=range(10,205,10))
    parser.add_argument('--reg_coeff', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_users', type=int, default=50)
    parser.add_argument('--num_Chosenusers', type=int, default=50)    
    parser.add_argument('--num_train', type=int, default=128)
    parser.add_argument('--num_experiments', type=int, default=2)
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--set_privacy_budget', type=int, default=[100000000])
    parser.add_argument('--clipthr', type=int, default=10)    

    args = parser.parse_args()
    
    # fpath = "./dataset/{0}.dat".format(args.dname)
    # X, y = load_dat(fpath, minmax=(0, 1), normalize=False, bias_term=True)

    print("Running the program ... [{0}]".format(
        time.strftime("%m/%d/%Y %H:%M:%S")))
    print("Parameters")
    print("----------")

    for arg in vars(args):
        print(" - {0:22s}: {1}".format(arg, getattr(args, arg)))

    #start_time = time.clock()

    main(args)

    #elapsed = time.clock() - start_time
    #mins, sec = divmod(elapsed, 60)
    #hrs, mins = divmod(mins, 60)

    print("The program finished. [{0}]".format(
        time.strftime("%m/%d/%Y %H:%M:%S")))
    #print("Elasepd time: %d:%02d:%02d" % (hrs, mins, sec))
