

from hashlib import new
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
from scipy import stats
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
from dataset import get_data_queue_efficiently, get_data_queue_negsampling_efficiently, get_data_queue_subsampling_efficiently, get_data_queue_subsampling_efficiently_explicit

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


def generate_random_hparam(data_size):
    if data_size == 1:
        opt_list = np.random.choice(['Adagrad', 'Adam'], data_size)
        lr_list = np.random.uniform(low=1e-4, high=1e-2, size=data_size)
        embedding_dim_list = np.random.choice(range(1, 64+1, 1), data_size)
        weight_decay_list = np.random.uniform(low=1e-4, high=1e-1, size=data_size)
        return opt_list[0], lr_list[0], embedding_dim_list[0], weight_decay_list[0]
    else:
        opt_list = np.random.choice(['Adagrad', 'Adam'], data_size)
        lr_list = np.random.uniform(low=1e-4, high=1e-2, size=data_size)
        embedding_dim_list = np.random.choice(range(1, 64+1, 1), data_size)
        weight_decay_list = np.random.uniform(low=1e-4, high=1e-1, size=data_size)
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

class MetaArchClassifier:
    def __init__(self, meta_mode) -> None:
        self.meta_mode = meta_mode
        self.hparams_list = []
        self.arch_encode_list = [] # encoding in '_' 
        self.arch_single_list = [] # dict
        self.performance_list = []

        # for RF regression
        self.features = []
        self.targets = []
        self.labels = [] # for z targets

        self.data_size = 0
        self.quantile = 0.35
        self.losses = []
        self.scores = []
        self.criterion = []

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
        # print(df)
        # df.drop(columns=['performance', 'opt'], inplace=True)
        
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
            # add loss function TODO
            # loss = self.regressor.loss() # non
            # print("X_train.shape: {}".format(X_train.shape))
            score = self.regressor.score(X_test, y_test)
            print(f'score: {score}')
            self.scores.append(score)
            # criterion = 
            # self.criterion.append()
            # result = self.regressor.predict(X_test)
            
            # loss = 1
            # print(f'loss: {loss}')
            # self.losses.append(loss)
            
            test_data_size = 200
            arch_next_list = generate_random_arch(test_data_size, self.remaining_arches_encoding)

            data_list = [{
                'u_cf_emb': arch_next_list[i]['cf'][0]+'_'+arch_next_list[i]['emb']['u'],
                'i_cf_emb': arch_next_list[i]['cf'][1]+'_'+arch_next_list[i]['emb']['i'],
                'ifc': arch_next_list[i]['ifc'],
                'pred': arch_next_list[i]['pred'] } for i in range(test_data_size)]
            df_test = pd.DataFrame(data_list)
            # print(test_features_opt.shape, test_features_other.shape)
            # feature_test = np.concatenate((test_features_opt, test_features_other), axis=1)

            feature_test = self.arch_encoder.fit_transform(df_test)
            # feature_test = ohenc.fit_transform(df)
            result_test = self.regressor.predict(feature_test)
            # best_idx = np.argmax(result_test)
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
            # TODO: mlp tuner
            pass
        elif self.meta_mode == 'bore_rf':
            # TODO: construct a bore + rf meta learner, consist of a bore and a random forest classifier
            X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.1, random_state=0)
            self.regressor.fit(X_train, y_train)
            # print("X_train.shape: {}".format(X_train.shape))
            score = self.regressor.score(X_test, y_test)
            result = self.regressor.predict(X_test)
            
            test_data_size = 200
            generate_random_arch(test_data_size, self.remaining_arches_encoding)
            data_list = [{'opt': opt_list[i], 'lr': lr_list[i], 'embedding_dim': embedding_dim_list[i] ,'weigh_decay': weight_decay_list[i]} for i in range(len(opt_list))]
            # data_list
            df = pd.DataFrame(data_list)
            df.drop(columns=['opt'], inplace=True)
            feature_test = self.arch_encoder.fit_transform(df)

            # feature_test = ohenc.fit_transform(df)
            result_test = self.regressor.predict(feature_test)
            best_idx = np.argmax(result_test)
            hparams_next = [opt_list[best_idx], lr_list[best_idx], embedding_dim_list[best_idx], weight_decay_list[best_idx]]
            # next_hparams, result_test[best_idx]
        else:
            pass
        return hparams_next


if __name__ == '__main__':
    # find best hyper-parameters(hparams) on the original dataset
    # it is a arch tuning program on fixed model and original dataset
    storage_nums = 64 # 80
    trial_nums = 20 # we will draw a figure: trails vs performances
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
        num_users = 138493
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
        train_queue_pair_sub, valid_queue_sub, test_queue_sub = get_data_queue_subsampling_efficiently(data_path, args, item_down_sample_portion=0.3)
        num_user_sub = args.num_users
        num_item_sub = args.num_items
        
    else: 
        # train_queue, valid_queue, test_queue = get_data_queue_efficiently(data_path, args)
        train_queue_sub, valid_queue_sub, test_queue_sub = get_data_queue_subsampling_efficiently_explicit(data_path, args)
        num_user_sub = args.num_users
        num_item_sub = args.num_items

    # construct a model, maybe can be changed to many models and average or maximize the performance
    remaining_arches_encoding = open(args.remaining_arches, 'r').readlines() # opten the file of arch
    remaining_arches_encoding = list(map(lambda x: x.strip(), remaining_arches_encoding))
    # print("remaining_arches_encoding: {}".format(remaining_arches_encoding))
    # ''' # start of stage1
    arch_single_list = generate_random_arch(storage_nums, remaining_arches_encoding)
    stage1_start = time()
    # hparams choice: random start, all in numpy frame
    opt_list, lr_list, embedding_dim_list, weight_decay_list = generate_random_hparam(data_size=storage_nums)
    # store some hparams-perf pairs
    # we don't need subgraph here  maybe we can try some multi process technique here
    hostname = socket.gethostname()
    # print("hostname: {}".format(hostname))
    avaliable_device_ids = [6, 5] # 
    print("on host {} avaliable_device_ids: {}".format(hostname, avaliable_device_ids))

    # avaliable_device_ids = avaliable_device_ids[:3]
    threads_per_device = 2
    assigned_device_ids = [val for val in avaliable_device_ids for i in range(threads_per_device)]  # final gpu-ids x 3
    task_number = math.ceil(storage_nums / len(assigned_device_ids))  # in each task, three thread on one GPU
    task_split = list(range(0, storage_nums, len(assigned_device_ids)))
    task_split.append(storage_nums)
    task_index = [list(range(task_split[i], task_split[i+1])) for i in range(task_number)]

    rmse_list1 = [] # start of stage1
    for tasks in task_index:
        with mp.Pool(processes=len(tasks)) as p:
            print('\nStage1: getting storage on subsample dataset')
            p.name = 'test on arch tuning'
            if args.data_type == 'implicit':# 装载
                args.num_users, args.num_items = num_users, num_items
                jobs_sub = []
                jobs = []
                for i in tasks:
                    curr_hparams = opt_list[i], lr_list[i], embedding_dim_list[i], weight_decay_list[i]
                    args.opt, args.lr, args.embedding_dim, args.weight_decay = curr_hparams
                    args.num_users, args.num_items = num_user_sub, num_item_sub
                    
                    jobs_sub.append([arch_single_list[i], num_user_sub, num_item_sub, [], valid_queue_sub, test_queue_sub, train_queue_pair_sub, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid])
                    # jobs.append([arch_single_list[i], num_users, num_items, [], valid_queue, test_queue, train_queue_pair, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] )
                rmse_list1 += p.map(get_single_model_performance_time, jobs_sub)
                # rmse_list1 += p.map(get_single_model_performance_time, jobs)


            else: # explicit
                args.num_users, args.num_items = num_users, num_items
                curr_hparams = opt_list[i], lr_list[i], embedding_dim_list[i], weight_decay_list[i]
                args.opt, args.lr, args.embedding_dim, args.weight_decay = curr_hparams
                jobs = [[arch_single_list[i], num_users, num_items, train_queue_sub, valid_queue_sub, test_queue_sub, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] for i in tasks]
                # rmse_list1 += p.map(get_single_model_performance, jobs)
                rmse_list1 += p.map(get_single_model_performance_time(), jobs)
            p.close()
            p.terminate()
    
    stage1_end = time()
    print(f'rmse_list1 in stage 1: {rmse_list1}')
    # hparams_list = [[opt_list[i], lr_list[i], embedding_dim_list[i], weight_decay_list[i]] for i in range(storage_nums)]
    if args.data_type == 'implicit':
        performance_list_stage1 = [rmse[0][0] for rmse in rmse_list1]
        time_list_stage1 = [rmse[1] for rmse in rmse_list1]
    else:
        # performance_list = rmse_list1
        performance_list_stage1 = [rmse[0] for rmse in rmse_list1]
        time_list_stage1 = [rmse[1] for rmse in rmse_list1]
    
    # save storage data
    results_dict = dict()
    results_dict['performance_list'] = performance_list_stage1
    results_dict['arch_single_list'] = arch_single_list
    with open('results.txt', 'a', encoding='utf-8') as f:
        f.writelines(str(performance_list_stage1) + '\n')
        f.writelines(str(time_list_stage1) + '\n')
    # TODO: select top10


    json_str = json.dumps(results_dict, indent=4)
    with open('arch_bore_rf_meta_log1_{}.json'.format(args.dataset), 'w') as json_file:
        json_file.write(json_str)
    
    # ''' # end of stage 1
    

    # ''' # start of stage 2
    with open(f'arch_bore_rf_meta_log1_{args.dataset}.json','r') as f:
        data = json.load(f)
    performance_list = data['performance_list']
    arch_single_list = data['arch_single_list']
    
    # setting meta_learner
    meta_arch_classifier = MetaArchClassifier(meta_mode='bore_rf')
    # meta_arch_classifier = MetaArchClassifier(meta_mode='random')
    remaining_arches_encoding = open(args.remaining_arches, 'r').readlines() # opten the file of arch
    remaining_arches_encoding = list(map(lambda x: x.strip(), remaining_arches_encoding))
    meta_arch_classifier.data_init(arch_single_list, performance_list, remaining_arches_encoding)
    arch_single_next = meta_arch_classifier.get_next_arch_single()

    max_perf_idx, max_perf = np.argmax(performance_list), np.max(performance_list)
    print('best arch: {}, performance: {}'.format(arch_single_list[max_perf_idx], max_perf))
    # print("hparams_next: {}".format(hparams_next))
    # meta_classifier.classifier.fit(hparams_list, performance_list)

    # optimizing hparams by surrogate model 
    # we don't need different host in this stage
    if args.data_type == 'implicit': 
        # doing arch tuning, we had better to tuning on a subgraph
        train_queue_pair, valid_queue, test_queue = get_data_queue_negsampling_efficiently(data_path, args)
        # num_users, num_items = args.num_users, args.num_items
    else: 
        train_queue, valid_queue, test_queue = get_data_queue_efficiently(data_path, args)
        # num_users, num_items = args.num_users, args.num_items
        # train_queue_sub, valid_queue_sub, test_queue_sub = get_data_queue_subsampling_efficiently_explicit(data_path, args)
        # num_user_sub = args.num_users
        # num_item_sub = args.num_items

    stage2_start = time()
    rmse_list1 = []
    performance_list_stage2 = []
    time_list_stage2 = []
    for i in range(trial_nums):
        # score = get_arch_performance_cf_signal_param_device(args)
        curr_arch = arch_single_next
        curr_hparams = generate_random_hparam(data_size=1) # random hp
        args.opt, args.lr, args.embedding_dim, args.weight_decay = curr_hparams
        if args.data_type == 'implicit':# 装载
            # args.num_users, args.num_items = num_user_sub, num_item_sub
            # jobs = [arch_single_next, num_user_sub, num_item_sub, [], valid_queue_sub, test_queue_sub, train_queue_pair_sub, args, [curr_hparams[1], curr_hparams[2]], args.device, args.if_valid]
            
            args.num_users, args.num_items = num_users, num_items
            jobs = [arch_single_next, num_users, num_items, [], valid_queue, test_queue, train_queue_pair, args, [curr_hparams[1], curr_hparams[2]], args.device, args.if_valid]
            # rmse_list1.append(list(get_single_model_performance(jobs)))
            rmse_list1.append(list(get_single_model_performance_time(jobs)))

        else: # explicit
            # args.num_users, args.num_items = num_user_sub, num_item_sub
            # jobs = [arch_single_next, num_user_sub, num_item_sub, train_queue_sub, valid_queue_sub, test_queue_sub, args, [curr_hparams[1], curr_hparams[2]], args.device, args.if_valid] 
            args.num_users, args.num_items = num_users, num_items
            jobs = [arch_single_next, num_users, num_items, train_queue, valid_queue, test_queue, args, [curr_hparams[1], curr_hparams[2]], args.device, args.if_valid] 
            # rmse_list1.append(list(get_single_model_performance(jobs)))
            rmse_list1.append(list(get_single_model_performance_time(jobs)))

        stage2_end = time()
        new_performance = rmse_list1[i][0][0] # should be generated by model
        new_time_s2 = rmse_list1[i][1] # time
        performance_list_stage2.append(new_performance)
        time_list_stage2.append(new_time_s2)

        print(f"arch_single_next: {arch_dict_to_encoding(arch_single_next)}, new_performance: {new_performance}|{max_perf}, time in s2: {new_time_s2}")
        meta_arch_classifier.update_data_density(arch_single_next, new_performance)
        arch_single_next = meta_arch_classifier.get_next_arch_single()
        # print('encoding: {}'.format(arch_dict_to_encoding(arch_single_next))) 
        performance_list = rmse_list1

    
    print(f'stage1 time: {stage1_end-stage1_start}, stage2 time: {stage2_end-stage2_start}')
    print(f'performance_list_stage1: {performance_list_stage1}')
    
    with open('results.txt', 'a', encoding='utf-8') as f:
        # f.writelines(str(performance_list_stage1) + '\n')
        # f.writelines(str(time_list_stage1) + '\n')
        f.writelines(str(performance_list_stage2) + '\n')
        f.writelines(str(time_list_stage2) + '\n\n')

    recall20_list = [p[0][0] for p in rmse_list1]
    print(f'recall20_list: {recall20_list}')
    recall20_curr_best_list = [recall20_list[0]] + [np.max(recall20_list[:i]) for i in range(1, len(recall20_list))]
    # print(f'recall20_curr_best_list: {recall20_curr_best_list}')
    # plt.plot(list(range(1, trial_nums+1, 1)), recall20_curr_best_list)
    # print(f'loss: {meta_arch_classifier.losses}')
    print(f'scores: {meta_arch_classifier.scores}')
    # plt.plot(list(range(1, trial_nums+1, 1)), [max_perf, max_perf], )
    # plt.axhline(max_perf)

    plt.xlabel('trials')
    plt.ylabel('recall@20')
    plt.title('tuning on {}'.format(args.dataset))
    best_arch_stage1_str = arch_dict_to_encoding(arch_single_list[max_perf_idx])
    current_time = strftime("%Y-%m-%d-%H:%M:%S", localtime())
    plt.savefig(os.path.join('bore_meta', 'test_arch_bore2_rf_{}_{}_{}.jpg'.format(best_arch_stage1_str, args.dataset,current_time)))
    plt.show()
    plt.close()
    # ''' # end of stage 2
