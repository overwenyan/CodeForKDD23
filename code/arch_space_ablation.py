

# from hashlib import new
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


if __name__ == '__main__':
    # find best hyper-parameters(hparams) on the original dataset
    # it is a arch tuning program on fixed model and original dataset
    storage_nums = 80 # 80
    # trial_nums = 20 # we will draw a figure: trails vs performances
    # trial_nums += storage_nums
    # construct dataset 
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
        # doing arch tuning, we had better to tuning on a subgraph
        train_queue_pair, valid_queue, test_queue = get_data_queue_negsampling_efficiently(data_path, args)
        # train_queue_pair_sub, valid_queue_sub, test_queue_sub = get_data_queue_subsampling_efficiently(data_path, args, item_down_sample_portion=0.3)
        # num_user_sub, num_item_sub = args.num_users, args.num_items 
    else: 
        train_queue, valid_queue, test_queue = get_data_queue_efficiently(data_path, args)
        # train_queue_sub, valid_queue_sub, test_queue_sub = get_data_queue_subsampling_efficiently_explicit(data_path, args)
        # num_user_sub, num_item_sub = args.num_users, args.num_items 

    # construct a model, maybe can be changed to many models and average or maximize the performance
    remaining_arches_encoding = open(args.remaining_arches, 'r').readlines() # origin arch space
    arches_encoding_reduced = open('save/arches_reduced.txt', 'r').readlines() # reduced arch space

    remaining_arches_encoding = list(map(lambda x: x.strip(), remaining_arches_encoding))
    arches_encoding_reduced = list(map(lambda x: x.strip(), arches_encoding_reduced))
    # arches_encoding_reduced = arches_encoding_reduced[:30]
    arches_encoding_reduced = remaining_arches_encoding
    # ''' # start of stage1
    arch_single_list = generate_random_arch(storage_nums, remaining_arches_encoding) # random selected model
    arch_single_reduced = generate_random_arch(storage_nums, arches_encoding_reduced)

    # arch_single_list = [arch_single_list[0]]*storage_nums # changed to a fixed model in a run
    stage1_start = time()

    # different search space
    opt_list, lr_list, embedding_dim_list, weight_decay_list = generate_random_hparam(data_size=storage_nums, space_mode='reduced')
    # opt_reduced, lr_reduced, embedding_dim_reduced, weight_decay_reduced = generate_random_hparam(data_size=storage_nums, space_mode='reduced')
    # store some hparams-perf pairs
    # we don't need subgraph here  maybe we can try some multi process technique here
    hostname = socket.gethostname()
    # print("hostname: {}".format(hostname))
    avaliable_device_ids = [2,3] # 
    print("train and eval on host {} avaliable_device_ids: {}".format(hostname, avaliable_device_ids))

    # avaliable_device_ids = avaliable_device_ids[:3]
    threads_per_device = 1
    assigned_device_ids = [val for val in avaliable_device_ids for i in range(threads_per_device)]  # final gpu-ids x 3
    task_number = math.ceil(storage_nums / len(assigned_device_ids))  # in each task, three thread on one GPU
    task_split = list(range(0, storage_nums, len(assigned_device_ids)))
    task_split.append(storage_nums)
    task_index = [list(range(task_split[i], task_split[i+1])) for i in range(task_number)]

    rmse_list1 = [] # start of stage1
    recall20_list_origin = []
    recall20_list_reduced = []
    for tasks in task_index:
        with mp.Pool(processes=len(tasks)) as p:
            print('\nStage1: getting storage on subsample dataset')
            p.name = 'test on arch tuning'
            if args.data_type == 'implicit':# 装载
                args.num_users, args.num_items = num_users, num_items
                jobs_sub = []
                jobs = []
                jobs_reduced = []
                for i in tasks:
                    origin_hparams = opt_list[i], lr_list[i], embedding_dim_list[i], weight_decay_list[i]
                    # reduced_hparams = opt_reduced[i], lr_reduced[i], embedding_dim_reduced[i], weight_decay_reduced[i]
                    args.opt, args.lr, args.embedding_dim, args.weight_decay = origin_hparams
                    # args.num_users, args.num_items = num_user_sub, num_item_sub
                    # jobs_sub.append([arch_single_list[i], num_user_sub, num_item_sub, [], valid_queue_sub, test_queue_sub, train_queue_pair_sub, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid])
                    jobs.append([arch_single_list[i], num_users, num_items, [], valid_queue, test_queue, train_queue_pair, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] )

                    # args.opt, args.lr, args.embedding_dim, args.weight_decay = reduced_hparams
                    jobs_reduced.append([arch_single_reduced[i], num_users, num_items, [], valid_queue, test_queue, train_queue_pair, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] )
                    
                # rmse_list1 += p.map(get_single_model_performance_time, jobs_sub)
                # rmse_list1 += p.map(get_single_model_performance_time, jobs)

            else: # explicit
                args.num_users, args.num_items = num_users, num_items
                curr_hparams = opt_list[i], lr_list[i], embedding_dim_list[i], weight_decay_list[i]
                args.opt, args.lr, args.embedding_dim, args.weight_decay = curr_hparams
                jobs = [[arch_single_list[i], num_users, num_items, train_queue, valid_queue, test_queue, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] for i in tasks]
                # jobs = [[arch_single_list[i], num_users, num_items, train_queue_sub, valid_queue_sub, test_queue_sub, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] for i in tasks]
                # rmse_list1 += p.map(get_single_model_performance, jobs)
                rmse_list1 += p.map(get_single_model_performance_time, jobs)
        
            recall20_list_origin += p.map(get_single_model_performance_time, jobs)
            recall20_list_reduced += p.map(get_single_model_performance_time, jobs_reduced)
            p.close()
            p.terminate()
    stage1_end = time()
    print(f'recall20_list_origin: {recall20_list_origin}')
    print(f'recall20_list_reduced: {recall20_list_reduced}')
    # TODO: select top10
    
    performance_origin = np.array([item[0][0] for item in recall20_list_origin])
    time_origin = np.array([item[1] for item in recall20_list_origin])
    performance_origin_top1 = np.array([performance_origin[0]] + [np.max(performance_origin[:i]) for i in range(1, len(performance_origin))])
    # TODO:how to choose top 5
    # performance_origin_top5 = np.array([performance_origin[0]] + [np.max(performance_origin[:i]) for i in range(1, len(performance_origin))]) 
    performance_origin_top5 = []
    for i in range(len(performance_origin)):
        if i == 0:
            item = performance_origin[0]
            # item = 0.05
        elif i <= 5:
            item = np.mean(performance_origin[:i])
            # item = 0.05
            # item = performance_origin[0]
        else:
            top5_list = performance_origin[np.argpartition(performance_origin[:i], -5)[-5:]]
            item = np.mean(top5_list)
        performance_origin_top5.append(item)
    time_origin_accumulated = np.array([np.sum(time_origin[:i]) for i in range(1,time_origin.shape[0]+1)])

    
    performance_reduced = np.array([item[0][0] for item in recall20_list_reduced])
    time_reduced = np.array([item[1] for item in recall20_list_reduced])
    performance_reduced_top1 = np.array([performance_reduced[0]] + [np.max(performance_reduced[:i]) for i in range(1, len(performance_reduced))])
    performance_reduced_top5 = []
    for i in range(len(performance_reduced)):
        if i == 0:
            item = performance_reduced[0]
            # item = 0.05
        elif i <= 5:
            item = np.mean(performance_reduced[:i])
            # item = performance_reduced[0]
            # item = 0.05
        else:
            top5_list = performance_reduced[np.argpartition(performance_reduced[:i], -5)[-5:]]
            item = np.mean(top5_list)
        performance_reduced_top5.append(item)
    time_reduced_accumulated = np.array([np.sum(time_reduced[:i]) for i in range(1,time_reduced.shape[0]+1)])

    with open('results.txt', 'a', encoding='utf-8') as f:
        # f.writelines(str(performance_list_stage1) + '\n')
        # f.writelines(str(time_list_stage1) + '\n')
        f.writelines('recall20_list_origin = ' + str(recall20_list_origin) + '\n')
        f.writelines('recall20_list_reduced = ' + str(recall20_list_reduced) + '\n\n')

    plt.plot(time_origin_accumulated, performance_origin_top1, label='rank1@random@origin arch', c='r')
    plt.plot(time_origin_accumulated, performance_origin_top5, label='top5avg@random@origin arch', c='tomato')
    plt.plot(time_reduced_accumulated, performance_reduced_top1, label='rank1@random@reduced arch', c='g')
    plt.plot(time_reduced_accumulated, performance_reduced_top5, label='top5avg@random@reduced arch', c='springgreen')

    plt.xlabel('time/s')
    plt.ylabel('Recall@20')
    plt.title(f'performance arch space ablation @{args.dataset}')
    plt.legend()
    # recall20_list = [p[0][0] for p in rmse_list1]
    # print(f'recall20_list: {recall20_list}')
    # recall20_curr_best_list = [recall20_list[0]] + [np.max(recall20_list[:i]) for i in range(1, len(recall20_list))]
    # print(f'recall20_curr_best_list: {recall20_curr_best_list}')
    # plt.plot(list(range(1, trial_nums+1, 1)), recall20_curr_best_list)
    # print(f'loss: {meta_arch_classifier.losses}')
    # print(f'scores: {meta_arch_classifier.scores}')
    # plt.plot(list(range(1, trial_nums+1, 1)), [max_perf, max_perf], )
    # plt.axhline(max_perf)

    current_time = strftime("%Y-%m-%d-%H:%M:%S", localtime())
    plt.savefig(os.path.join('ablation_figs', f'arches_ablation_{args.dataset}_{current_time}.jpg'))
    # plt.show()
    # plt.close()
    # ''' # end of stage 2
