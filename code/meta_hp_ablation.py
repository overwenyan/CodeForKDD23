

# from hashlib import new
from curses import meta
from random import Random
import GPUtil
import socket
import math
from time import localtime, sleep, strftime, time
import os
import argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import  RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
# from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

import torch
import torch.utils
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import multiprocessing as mp # 多线程工
# from zmq import PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_MESSAGE

from main import get_single_model_performance, get_single_model_performance_time
from controller import sample_arch_cf, sample_arch_cf_signal, sample_arch_cf_test
from dataset import get_data_queue_cf, get_data_queue_cf_nonsparse
from dataset import get_data_queue_efficiently, get_data_queue_negsampling_efficiently
from dataset import get_data_queue_subsampling_efficiently, get_data_queue_subsampling_efficiently_explicit # sample

# parser of 
parser = argparse.ArgumentParser(description="Generate configs")
parser.add_argument('--data_type', type=str, default='implicit', help='explicit or implicit(default)')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--valid_portion', type=float, default=0.25, help='portion of validation data')
parser.add_argument('--mode', type=str, default='random_single', help='search or single mode')
parser.add_argument('--save', type=str, default='save/', help='experiment name')
parser.add_argument('--use_gpu', type=int, default=1, help='whether use gpu')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--train_epochs', type=int, default=2000, help='num of training epochs')
parser.add_argument('--search_epochs', type=int, default=1000, help='num of searching epochs')
parser.add_argument('--loss_func', type=str, default='bprloss', help='Implicit loss function')
parser.add_argument('--device', type=int, default=0, help='GPU device')
parser.add_argument('--batch_size', type=int, default=3000, help='batch size')


parser.add_argument('--file_id', type=int, default=206, help='size of anchor sample number')
parser.add_argument('--remaining_arches', type=str, default='src/arch.txt', help='')
parser.add_argument('--if_valid', type=int, default=1, help='use validation set for tuning single architecture or not')
parser.add_argument('--mark', type=str, default='') # 
parser.add_argument('--seed', type=int, default=1, help='random seed')
# sub sample test model 
parser.add_argument('--dataset', type=str, default='ml-100k', help='dataset')
parser.add_argument('--sample_portion', type=float, default=0.1, help='portion of data subgraph')
parser.add_argument('--anchor_num', type=int, default=60, help='size of anchor sample number')
parser.add_argument('--sample_mode', type=str, default='distribute', help='topk or distribute mode of sampling')

args = parser.parse_args()

def arch_dict_to_encoding(arch_dict):
    arch_str = '{}_{}_{}_{}_{}'.format(arch_dict['cf'], arch_dict['emb']['u'], arch_dict['emb']['i'], arch_dict['ifc'], arch_dict['pred'])
    return arch_str

def generate_random_hparam(data_size, space_mode='origin'):
    if space_mode == 'reduced':
        opt_list = np.random.choice(['Adagrad', 'Adam'], data_size)
        lr_list = np.random.uniform(low=1e-4, high=1e-2, size=data_size) # smaller range of learning rate
        embedding_dim_list = np.random.choice(range(1, 64+1, 1), data_size)
        weight_decay_list = np.random.uniform(low=1e-4, high=1e-1, size=data_size)
    else: # origin hp space
        opt_list = np.random.choice(['Adagrad', 'Adam', 'SGD'], data_size)
        lr_list = np.random.uniform(low=1e-6, high=1e0, size=data_size)
        embedding_dim_list = np.random.choice(range(1, 256+1, 1), data_size)
        weight_decay_list = np.random.uniform(low=1e-5, high=1e-0, size=data_size)

    if data_size == 1:
        return opt_list[0], lr_list[0], embedding_dim_list[0], weight_decay_list[0]
    else:
        return opt_list, lr_list, embedding_dim_list, weight_decay_list

def generate_random_arch(data_size, remaining_arches_encoding):
    if data_size == 1:
        arch_encoding = np.random.choice(remaining_arches_encoding, data_size)
        arch_single = sample_arch_cf()
        arch_single['cf'], arch_single['emb']['u'], arch_single['emb']['i'], arch_single['ifc'], arch_single['pred'] = arch_encoding[0].split('_')
        return arch_single
    else:
        arch_single_list = [] # 详细
        arch_encode_list = np.random.choice(remaining_arches_encoding, data_size)
        for arch_encoding in arch_encode_list:
            arch_single = sample_arch_cf()
            arch_single['cf'], arch_single['emb']['u'], arch_single['emb']['i'], arch_single['ifc'], arch_single['pred'] = arch_encoding.split('_')
            # print(type(arch_single))
            arch_single_list.append(dict(arch_single))
        return arch_single_list

def get_top_performance_list(performance_origin, time_origin):
    performance_origin_top1 = np.array([performance_origin[0]] + [np.max(performance_origin[:i]) for i in range(1, len(performance_origin))])
    performance_origin_top5 = []
    for i in range(len(performance_origin)):
        if i == 0:
            item = performance_origin[0]
        elif i <= 5:
            item = np.mean(performance_origin[:i])
        else:
            top5_list = performance_origin[np.argpartition(performance_origin[:i], -5)[-5:]]
            item = np.mean(top5_list)
        performance_origin_top5.append(item)
    time_origin_accumulated = np.array([np.sum(time_origin[:i]) for i in range(1,time_origin.shape[0]+1)])
    return performance_origin_top1, performance_origin_top5, time_origin_accumulated

class MetaHPLearner:
    def __init__(self, meta_mode) -> None:
        self.meta_mode = meta_mode
        self.hparams_list = []
        self.performance_list = []
        # for RF regression
        self.features = []
        self.targets = []
        self.labels = [] # for z targets

        self.data_size = 0
        self.quantile = 0.35

        self.losses = []
        self.scores = []

        self.ohenc = OneHotEncoder()
        self.scaler_enc = StandardScaler()
        self.regressor = None
        if self.meta_mode == 'bore_rf':
            self.regressor = RandomForestRegressor(n_estimators=200)
        elif self.meta_mode == 'bore_adaboost':
            self.regressor = AdaBoostRegressor()
        elif self.meta_mode == 'bore_mlp':
            self.regressor = MLPRegressor()
        elif self.meta_mode == 'bore_gp':
            self.regressor = GaussianProcessRegressor()
        else:
            pass


    def data_init(self, hparams_list, performance_list):
        self.hparams_list = hparams_list
        self.performance_list = performance_list
        print(f'self.performance_list: {self.performance_list}')
        self.data_size = len(hparams_list)
        
        data_list = [{'opt': hparams_list[i][0], 'lr': hparams_list[i][1], \
                'embedding_dim': hparams_list[i][2] ,\
               'weigh_decay': hparams_list[i][3], 'performance': performance_list[i] } for i in range(len(performance_list))]
        df = pd.DataFrame(data_list)
        df.drop(columns=['performance', 'opt'], inplace=True)

        opt_list = np.array([hparams_list[i][0] for i in range(len(hparams_list))])
        features_opt = self.ohenc.fit_transform(opt_list.reshape(-1, 1))
        features_opt = np.array(features_opt.todense())
        features_other = self.scaler_enc.fit_transform(df)
        self.features = np.concatenate((features_opt, features_other), axis=1)

        self.targets = [-item for item in self.performance_list] # which we want to maximize
        tau = np.quantile(self.targets, q=self.quantile)
        self.labels = np.less(self.targets, tau) # y<=tau => 1; y>tau => 0 
        self.labels = np.array(self.labels, dtype='int')
        return

        
    def update_data_density(self, new_hparam, new_performance):
        self.hparams_list.append(new_hparam)
        self.performance_list.append(new_performance)
        self.data_size = len(self.hparams_list)

        data_list = [{'opt': self.hparams_list[i][0], 'lr': self.hparams_list[i][1], 'embedding_dim': self.hparams_list[i][2], 'weigh_decay': self.hparams_list[i][3], 'performance': self.performance_list[i] } for i in range(self.data_size)]
        df = pd.DataFrame(data_list)
        df.drop(columns=['performance', 'opt'], inplace=True)

        opt_list = np.array([self.hparams_list[i][0] for i in range(len(self.hparams_list))])
        features_opt = self.ohenc.fit_transform(opt_list.reshape(-1, 1))
        features_opt = np.array(features_opt.todense())
        # '''
        # features_opt = []
        # for i in range(len(self.hparams_list)):
        #     if self.hparams_list[i] == 'Adagrad':
        #         features_opt.append([0,1])
        #     elif self.hparams_list[i] == 'Adam':
        #         features_opt.append([1,0])
        # features_opt = np.array(features_opt)
        # '''
        # features_opt = np.array([self.hparams_list[i][0] for i in range(len(self.hparams_list))])
        features_other = self.scaler_enc.fit_transform(df)
        self.features = np.concatenate((features_opt, features_other), axis=1)
        print(f'feature shape: {self.features.shape}')

        self.targets = [-item for item in self.performance_list] # which we want to minimize
        tau = np.quantile(self.targets, q=self.quantile)
        # print("q: {}, tau: {}".format(q, tau))
        self.labels = np.less(self.targets, tau) # y<=tau => 1; y>tau => 0 
        self.labels = np.array(self.labels, dtype='int')
        return
    
    def get_next_hparams(self):
        hparams_next = []
        if self.meta_mode == 'random':
            hparams_next = generate_random_hparam(data_size=1)
        elif self.meta_mode == 'bore_mlp':
            pass
        elif self.meta_mode in ['bore_rf' ,'bore_gp' ,'bore_mlp', 'bore_adaboost']:
            # TODO: construct a bore + rf meta learner, consist of a bore and a random forest classifier
            X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.1, random_state=0)
            self.regressor.fit(X_train, y_train)
            self.losses.append( self.regressor.loss)


            # print("X_train.shape: {}".format(X_train.shape))
            score = self.regressor.score(X_test, y_test)
            # result = self.regressor.predict(X_test)
            # score = self.regressor.score(X_test, y_test)
            print(f'score: {score}')
            self.scores.append(score)
            test_data_size = 200    
            opt_list, lr_list, embedding_dim_list, weight_decay_list = generate_random_hparam(data_size=test_data_size,space_mode='reduced')
            data_list = [{'opt': opt_list[i], 'lr': lr_list[i], 'embedding_dim': embedding_dim_list[i] ,'weigh_decay': weight_decay_list[i]} for i in range(len(opt_list))]
            # data_list
            df = pd.DataFrame(data_list)
            df.drop(columns=['opt'], inplace=True)
            # ohenc = OneHotEncoder(categories='auto')
            test_features_opt = self.ohenc.fit_transform(opt_list.reshape(-1, 1))
            test_features_opt = np.array(test_features_opt.todense())
            # scaler = StandardScaler()
            test_features_other = self.scaler_enc.fit_transform(df)
            # print(test_features_opt.shape, test_features_other.shape)
            # print(f'feature test size {test_features_opt.shape}, {test_features_other.shape}')
            feature_test = np.concatenate((test_features_opt, test_features_other), axis=1)


            # feature_test = ohenc.fit_transform(df)
            # print(f'feature test size {feature_test.shape}')
            result_test = self.regressor.predict(feature_test)
            # best_idx = np.argmax(result_test)
            best_idx = np.argmin(result_test)
            hparams_next = [opt_list[best_idx], lr_list[best_idx], embedding_dim_list[best_idx], weight_decay_list[best_idx]]
            # next_hparams, result_test[best_idx]
        else:
            pass
        return hparams_next

if __name__ == '__main__':
    # find best hyper-parameters(hparams) on the original dataset
    # it is a arch tuning program on fixed model and original dataset
    storage_nums = 50 # 80
    trails_nums = 30
    if args.dataset == 'ml-100k': # default
        num_users = 943
        num_items = 1682
    elif args.dataset == 'ml-1m':
        num_users = 6040
        num_items = 3952
    elif args.dataset == 'ml-10m':
        num_users = 71567
        num_items = 65133
    elif args.dataset == 'ml-20m':
        num_users = 138493 # 138493, 131262 in dataset.py
        num_items = 131262
    elif args.dataset == 'amazon-book':
        num_users = 11899
        num_items = 16196
    elif args.dataset == 'yelp':
        num_users = 6102
        num_items = 18599 #  density: 0.3926%
    else:
        pass
    args.num_users = num_users
    args.num_items = num_items
    data_path = args.dataset + '/'
    if args.data_type == 'implicit': 
        train_queue_pair, valid_queue, test_queue = get_data_queue_negsampling_efficiently(data_path, args)
    else: 
        train_queue, valid_queue, test_queue = get_data_queue_efficiently(data_path, args)

    
    remaining_arches_encoding = open(args.remaining_arches, 'r').readlines() # origin arch space
    remaining_arches_encoding = list(map(lambda x: x.strip(), remaining_arches_encoding))
    
    arch_single_list = generate_random_arch(storage_nums, remaining_arches_encoding) # random selected model
    # arch_single_list = arch_single_list[0]

    stage1_start = time()
    # different search space
    opt_list, lr_list, embedding_dim_list, weight_decay_list = generate_random_hparam(data_size=storage_nums+trails_nums, space_mode='reduced') # random search hp
    opt_meta, lr_meta, embedding_dim_meta, weight_decay_meta = generate_random_hparam(data_size=storage_nums, space_mode='reduced')

    # store some hparams-perf pairs
    # we don't need subgraph here  maybe we can try some multi process technique here
    hostname = socket.gethostname()
    avaliable_device_ids = [5,6] # rtx3090
    print(f"train and eval on host {hostname} avaliable_device_ids: {avaliable_device_ids}")
    threads_per_device = 1
    assigned_device_ids = [val for val in avaliable_device_ids for i in range(threads_per_device)]  # final gpu-ids x 3
    task_number = math.ceil(storage_nums / len(assigned_device_ids))  # in each task, three thread on one GPU
    task_split = list(range(0, storage_nums, len(assigned_device_ids)))
    task_split.append(storage_nums)
    task_index = [list(range(task_split[i], task_split[i+1])) for i in range(task_number)]

    rmse_list1 = [] # start of stage1
    recall20_list_origin = []
    recall20_list_meta = []
    for tasks in task_index:
        with mp.Pool(processes=len(tasks)) as p:
            print('\nStage1: getting storage on subsample dataset')
            p.name = 'test on arch tuning'
            if args.data_type == 'implicit':# 装载
                args.num_users, args.num_items = num_users, num_items
                jobs = []
                jobs_meta = []
                for i in tasks:
                    origin_hparams = opt_list[i], lr_list[i], embedding_dim_list[i], weight_decay_list[i]
                    args.opt, args.lr, args.embedding_dim, args.weight_decay = origin_hparams
                    jobs.append([arch_single_list[0], num_users, num_items, [], valid_queue, test_queue, train_queue_pair, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] )
                    # meta hp learner storage 
                    meta_hparams = opt_meta[i], lr_meta[i], embedding_dim_meta[i], weight_decay_meta[i]
                    args.opt, args.lr, args.embedding_dim, args.weight_decay = meta_hparams
                    jobs_meta.append([arch_single_list[0], num_users, num_items, [], valid_queue, test_queue, train_queue_pair, args, [lr_meta[i], embedding_dim_meta[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] )

            else: # explicit
                args.num_users, args.num_items = num_users, num_items
                curr_hparams = opt_list[i], lr_list[i], embedding_dim_list[i], weight_decay_list[i]
                args.opt, args.lr, args.embedding_dim, args.weight_decay = curr_hparams
                jobs = [[arch_single_list[i], num_users, num_items, train_queue, valid_queue, test_queue, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] for i in tasks]
                rmse_list1 += p.map(get_single_model_performance_time, jobs)
        
            recall20_list_origin += p.map(get_single_model_performance_time, jobs)
            recall20_list_meta += p.map(get_single_model_performance_time, jobs_meta)
            p.close()
            p.terminate()
    stage1_end = time()
    ################## stage 1 end stage 2 start 

    print(f'recall20_list_origin: {recall20_list_origin}')
    print(f'recall20_list_meta: {recall20_list_meta}')

    with open('results_meta.txt', 'a', encoding='utf-8') as f:
        f.writelines('recall20_list_origin = ' + str(recall20_list_origin) + '\n')

    meta_hp_classifier = MetaHPLearner(meta_mode='bore_adaboost') # bore_mlp
    hparams_meta_init = [[opt_meta[i], lr_meta[i], embedding_dim_meta[i], weight_decay_meta[i]] for i in range(storage_nums)]
    performance_meta_init = [item[0][0] for item in recall20_list_meta]

    print(hparams_meta_init, performance_meta_init)
    meta_hp_classifier.data_init(hparams_meta_init, performance_meta_init)
    hparams_next = meta_hp_classifier.get_next_hparams()

    for i in range(trails_nums):
        args.opt, args.lr, args.embedding_dim, args.weight_decay = hparams_next
        if args.data_type == 'implicit':
            args.num_users, args.num_items = num_users, num_items
            job_meta = [arch_single_list[0], num_users, num_items, [], valid_queue, test_queue, train_queue_pair, args, [hparams_next[1], hparams_next[2]], args.device, args.if_valid]
            recall20_list_meta.append(tuple(get_single_model_performance_time(job_meta)))

            args.opt, args.lr, args.embedding_dim, args.weight_decay = opt_list[storage_nums+i], lr_list[storage_nums+i], embedding_dim_list[storage_nums+i], weight_decay_list[storage_nums+i]
            job_origin = [arch_single_list[0], num_users, num_items, [], valid_queue, test_queue, train_queue_pair, args, [lr_list[storage_nums+i], embedding_dim_list[storage_nums+i]], args.device, args.if_valid]
            recall20_list_origin.append(tuple(get_single_model_performance_time(job_origin)))
        
        print(f'recall20_list_meta: {recall20_list_meta}')
        new_performance = recall20_list_meta[storage_nums+i][0][0]
        meta_hp_classifier.update_data_density(hparams_next, new_performance)
        hparams_next = meta_hp_classifier.get_next_hparams()
    print(f'self.losses: {meta_hp_classifier.losses}')
    print(f'self.scores: {meta_hp_classifier.scores}')


    #################### data process
    performance_origin = np.array([item[0][0] for item in recall20_list_origin])
    time_origin = np.array([item[1] for item in recall20_list_origin])
    performance_origin_top1, performance_origin_top5, time_origin_accumulated = get_top_performance_list(performance_origin, time_origin)

    performance_metahp = np.array([item[0][0] for item in recall20_list_meta])
    time_metahp = np.array([item[1] for item in recall20_list_meta])
    performance_metahp_top1, performance_metahp_top5, time_metahp_accumulated = get_top_performance_list(performance_metahp, time_metahp)

    
    with open('results_hp_meta.txt', 'a', encoding='utf-8') as f:
        f.writelines('recall20_list_origin = ' + str(recall20_list_origin) + '\n')

    plt.plot(time_origin_accumulated, performance_origin_top1, label='rank1@random@origin dataset', c='r')
    plt.plot(time_origin_accumulated, performance_origin_top5, label='top5avg@random@origin dataset', c='tomato')
    plt.plot(time_metahp_accumulated, performance_metahp_top1, label=f'rank1@{meta_hp_classifier.meta_mode}@origin dataset', c='g')
    plt.plot(time_metahp_accumulated, performance_metahp_top5, label=f'top5avg@{meta_hp_classifier.meta_mode}@origin dataset', c='springgreen')

    plt.xlabel('time/s')
    plt.ylabel('Recall@20')
    plt.title(f'perf meta({meta_hp_classifier.meta_mode}) hp learner ablation@{args.dataset}')
    plt.legend()

    current_time = strftime("%Y-%m-%d-%H:%M:%S", localtime())
    plt.savefig(os.path.join('ablation_figs', f'meta_hp_ablation_{args.dataset}_{current_time}.jpg'))

