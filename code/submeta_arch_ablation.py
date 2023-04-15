

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
parser.add_argument('--sample_portion', type=float, default=0.2, help='portion of data subgraph')
parser.add_argument('--anchor_num', type=int, default=60, help='size of anchor sample number')
parser.add_argument('--sample_mode', type=str, default='topk', help='topk or distribute mode of sampling')

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
        return arch_single_list # in dictionary mode

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

class MetaArchLearner:
    def __init__(self, meta_mode) -> None:
        self.meta_mode = meta_mode
        self.hparams_list = []
        self.performance_list = []
        # for RF regression
        self.features = []
        self.targets = []
        self.labels = [] # for z targets

        self.data_size = 0
        self.quantile = 0.1
        self.losses = []
        self.scores = []
        self.criterion = []

        self.ohenc = OneHotEncoder()
        self.scaler_enc = StandardScaler()
        self.arch_encoder = OneHotEncoder()
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

    def get_feature_label_from_arch_encodes(self):
        # print(self.arch_single_list[0]['cf'][0], self.arch_single_list[0]['cf'][1])
        data_list = [{
            'u_cf_emb': self.arch_single_list[i]['cf'][0]+'_'+self.arch_single_list[i]['emb']['u'],
            'i_cf_emb': self.arch_single_list[i]['cf'][1]+'_'+self.arch_single_list[i]['emb']['i'],
            'ifc': self.arch_single_list[i]['ifc'],
            'pred': self.arch_single_list[i]['pred'],
            'performance': self.performance_list[i] } for i in range(self.data_size)]
        # print('data_list')
        df = pd.DataFrame(data_list)
        df.drop(columns=['performance'], inplace=True)
        self.features = self.arch_encoder.fit_transform(df)
        self.targets = self.performance_list # which we want to maximize
        tau = np.quantile(self.targets, q=self.quantile)
        self.labels = np.less(self.targets, tau) # y<=tau => 1; y>tau => 0 
        self.labels = np.array(self.labels, dtype='int')
        return 

    def data_init(self, arch_single_list, performance_list, remaining_arches_encoding):
        # self.arch_encode_list = arch_encode_list
        print(len(arch_single_list), len(performance_list))
        self.arch_single_list = arch_single_list
        self.performance_list = performance_list
        self.data_size = len(self.arch_single_list)
        self.remaining_arches_encoding = remaining_arches_encoding
        self.get_feature_label_from_arch_encodes()
        return

    def update_data_density(self, new_arch, new_performance):
        self.arch_single_list.append(new_arch)
        self.performance_list.append(new_performance)
        self.data_size = len(self.arch_single_list)
        self.get_feature_label_from_arch_encodes()
        return
    
    def get_next_arch_single(self):
        if self.meta_mode == 'random':
            arch_next = generate_random_arch(1, self.remaining_arches_encoding)
        elif self.meta_mode == 'bore_rf' or self.meta_mode == 'bore_gp' or self.meta_mode == 'bore_mlp': 
            # TODO: construct a bore + rf meta learner, consist of a bore and a random forest classifier
            X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.1, random_state=0)
            self.regressor.fit(X_train, y_train)
            # TODO: add loss function of random forest
            # score = self.regressor.score(X_test, y_test)
            score = self.regressor.score(X_train, y_train)
            # print(f'score: {score}')
            self.scores.append(score)
            
            test_data_size = 200
            arch_next_list = generate_random_arch(test_data_size, self.remaining_arches_encoding)

            data_list = [{
                'u_cf_emb': arch_next_list[i]['cf'][0]+'_'+arch_next_list[i]['emb']['u'],
                'i_cf_emb': arch_next_list[i]['cf'][1]+'_'+arch_next_list[i]['emb']['i'],
                'ifc': arch_next_list[i]['ifc'],
                'pred': arch_next_list[i]['pred'] } for i in range(test_data_size)]
            df_test = pd.DataFrame(data_list)

            feature_test = self.arch_encoder.fit_transform(df_test)
            result_test = self.regressor.predict(feature_test)
            best_idx = np.argmin(result_test)
            arch_next = arch_next_list[best_idx]
        else:
            pass
        return arch_next

    def get_next_hparams(self):
        hparams_next = []
        if self.meta_mode == 'random':
            hparams_next = generate_random_hparam(data_size=1)
        elif self.meta_mode == 'bore_mlp':
            pass
        elif self.meta_mode == 'bore_rf':
            X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.1, random_state=0)
            self.regressor.fit(X_train, y_train)
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
    storage_nums = 21 # 80
    trails_nums = 3 # can be changed
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
        train_queue_pair_sub, valid_queue_sub, test_queue_sub = get_data_queue_subsampling_efficiently(data_path, args, item_down_sample_portion=0.2)
        num_user_sub = args.num_users
        num_item_sub = args.num_items
        print(f'{num_user_sub}x{num_item_sub}...submeta....')
    else: 
        train_queue, valid_queue, test_queue = get_data_queue_efficiently(data_path, args)
        train_queue_sub, valid_queue_sub, test_queue_sub = get_data_queue_subsampling_efficiently_explicit(data_path, args)
        num_user_sub = args.num_users
        num_item_sub = args.num_items

    remaining_arches_encoding = open(args.remaining_arches, 'r').readlines() # origin arch space
    remaining_arches_encoding = list(map(lambda x: x.strip(), remaining_arches_encoding))
    
    # arch_single_list = generate_random_arch(storage_nums+trails_nums, remaining_arches_encoding) # random selected model
    arch_single_sub = generate_random_arch(storage_nums, remaining_arches_encoding)
    # arch_single_list = arch_single_list[0]
    arch_single_rs = generate_random_arch(storage_nums+trails_nums, remaining_arches_encoding)

    stage1_start = time()
    # different search space
    opt_list, lr_list, embedding_dim_list, weight_decay_list = generate_random_hparam(data_size=storage_nums+trails_nums, space_mode='reduced') # random search hp
    
    # store some hparams-perf pairs
    # we don't need subgraph here  maybe we can try some multi process technique here
    hostname = socket.gethostname()
    avaliable_device_ids = [3] # rtx3090
    print(f"train and eval on host {hostname} avaliable_device_ids: {avaliable_device_ids}")
    threads_per_device = 1 # 每个卡上的线程, ml-100k可以加到3
    assigned_device_ids = [val for val in avaliable_device_ids for i in range(threads_per_device)]  # final gpu-ids x 3
    task_number = math.ceil(storage_nums / len(assigned_device_ids))  # in each task, three thread on one GPU
    task_split = list(range(0, storage_nums, len(assigned_device_ids)))
    task_split.append(storage_nums)
    task_index = [list(range(task_split[i], task_split[i+1])) for i in range(task_number)]

    rmse_list1 = [] # start of stage1
    recall20_list_sub = []
    recall20_list_rs = []
    for tasks in task_index:
        with mp.Pool(processes=len(tasks)) as p:
            print('\nStage1: getting storage on dataset')
            p.name = 'test on arch tuning'
            if args.data_type == 'implicit':# 装载
                # args.num_users, args.num_items = num_users, num_items
                jobs = []
                jobs_rs = []
                jobs_sub = []
                for i in tasks:
                    origin_hparams = opt_list[i], lr_list[i], embedding_dim_list[i], weight_decay_list[i]
                    args.opt, args.lr, args.embedding_dim, args.weight_decay = origin_hparams
                    args.num_users, args.num_items = num_user_sub, num_item_sub
                    jobs_sub.append([arch_single_sub[i], num_user_sub, num_item_sub, [], valid_queue_sub, test_queue_sub, train_queue_pair_sub, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid])
                    recall20_list_sub += p.map(get_single_model_performance_time, jobs_sub)
                    
                    args.num_users, args.num_items = num_users, num_items
                    jobs_rs.append([arch_single_rs[i], num_users, num_items, [], valid_queue, test_queue, train_queue_pair, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid])
                    recall20_list_rs += p.map(get_single_model_performance_time, jobs_rs)
                print(f'jobs_sub len: {len(jobs_sub)}')
            else: # explicit
                args.num_users, args.num_items = num_users, num_items
                curr_hparams = opt_list[i], lr_list[i], embedding_dim_list[i], weight_decay_list[i]
                args.opt, args.lr, args.embedding_dim, args.weight_decay = curr_hparams
                args.num_users, args.num_items = num_user_sub, num_item_sub
                jobs_sub.append([[arch_single_sub[i], num_users, num_items, train_queue, valid_queue, test_queue, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] for i in tasks])
                # rmse_list1 += p.map(get_single_model_performance_time, jobs)
            p.close()
            p.terminate()
    stage1_end = time()
    ################## stage 1 end stage 2 start 

    # print(f'recall20_list_sub: {recall20_list_sub}, len: {len(recall20_list_sub)}')
    # print(f'recall20_list_rs: {recall20_list_rs}')

    with open('results_meta.txt', 'a', encoding='utf-8') as f:
        f.writelines('recall20_list_sub = ' + str(recall20_list_sub) + '\n')

    meta_arch_learner = MetaArchLearner(meta_mode='bore_rf') # use the best one on subgraph
    performance_sub = [item[0][0] for item in recall20_list_sub]
    print(f'performance_sub: {performance_sub}')
    print(f'len perf sub: {len(performance_sub)}; len arch sub: {len(arch_single_sub)}, len recall20_list_sub: {len(recall20_list_sub)}')

    meta_arch_learner.data_init(arch_single_sub, performance_sub, remaining_arches_encoding)
    arch_next = meta_arch_learner.get_next_arch_single()
    arch_sub = arch_single_sub.copy()

    # used for data
    for i in range(trails_nums):
        args.opt, args.lr, args.embedding_dim, args.weight_decay = opt_list[storage_nums+i], lr_list[storage_nums+i], embedding_dim_list[storage_nums+i], weight_decay_list[storage_nums+i]
        arch_sub.append(arch_next)
        if args.data_type == 'implicit':
            args.num_users, args.num_items = num_user_sub, num_item_sub
            job_meta_sub = [arch_sub[storage_nums+i], num_user_sub, num_item_sub, [], valid_queue_sub, test_queue_sub, train_queue_pair_sub, args, [lr_list[storage_nums+i], embedding_dim_list[storage_nums+i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid]
            recall20_list_sub.append(tuple(get_single_model_performance_time(job_meta_sub)))

            args.num_users, args.num_items = num_users, num_items
            job_origin = [arch_single_rs[storage_nums+i], num_users, num_items, [], valid_queue, test_queue, train_queue_pair, args, [lr_list[storage_nums+i], embedding_dim_list[storage_nums+i]], args.device, args.if_valid]
            recall20_list_rs.append(tuple(get_single_model_performance_time(job_origin)))
        
        print(f'recall20_list_sub: {recall20_list_sub}')
        new_performance = recall20_list_sub[storage_nums+i][0][0]
        meta_arch_learner.update_data_density(arch_next, new_performance)
        arch_next = meta_arch_learner.get_next_arch_single()
        
    #################### data process
    performance_sub = np.array([item[0][0] for item in recall20_list_sub])
    time_sub = np.array([item[1] for item in recall20_list_sub])    
    performance_sub_top1, performance_sub_top5, time_sub_accumulated = get_top_performance_list(performance_sub, time_sub)
    with open(f'results_sub_2.txt', 'a', encoding='utf-8') as f:
        # f.writelines('recall20_list_origin = ' + str(recall20_list_sub) + '\n')
        f.write('arch_encoding, performance, time\n')
        for i in range(len(performance_sub)):
            f.write(f'{arch_dict_to_encoding(arch_sub[i])}, {performance_sub[i]}, {time_sub[i]}\n')
    


    performance_origin = np.array([item[0][0] for item in recall20_list_rs])
    time_origin = np.array([item[1] for item in recall20_list_rs])    
    performance_rs_arch_top1, performance_rs_arch_top5, time_rs_arch_accumulated = get_top_performance_list(performance_origin, time_origin)

    plt.plot(time_sub_accumulated, performance_sub_top1, label=f'rank1@random@sub_{args.dataset}', c='r')
    plt.plot(time_sub_accumulated, performance_sub_top5, label=f'top5avg@random@sub_{args.dataset}', c='tomato')

    plt.plot(time_rs_arch_accumulated, performance_rs_arch_top1, label=f'rank1@RS@origin dataset', c='g')
    plt.plot(time_rs_arch_accumulated, performance_rs_arch_top5, label=f'top5avg@RS@origin dataset', c='springgreen')

    plt.xlabel('time/s')
    plt.ylabel('Recall@20')
    plt.title(f'perf comparison@sub_{args.dataset}@{meta_arch_learner.meta_mode}')
    plt.legend()

    current_time = strftime("%Y-%m-%d-%H:%M:%S", localtime())
    plt.savefig(os.path.join('sub_figs', f'meta_arch_{args.dataset}_{current_time}.jpg'))

    # print(f"meta_arch_learner.scores: {meta_arch_learner.scores}")
    # plt.figure()
    # plt.plot(meta_arch_learner.scores)
    # plt.xlabel(f"meta arch learner ({meta_arch_learner.meta_mode}) epoch ")
    # plt.ylabel("score")
    # plt.title(f"meta arch learner ({meta_arch_learner.meta_mode}) training scores")
    # plt.savefig(os.path.join('ablation_figs', f'metalearner_arch_{args.dataset}_{current_time}.jpg'))

