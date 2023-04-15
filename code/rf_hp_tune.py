

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
import os
import GPUtil

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

import torch
import torch.utils
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import multiprocessing as mp # 多线程


from main import get_single_model_performance
from controller import sample_arch_cf, sample_arch_cf_signal, sample_arch_cf_test
from dataset import get_data_queue_cf, get_data_queue_cf_nonsparse, get_data_queue_efficiently, get_data_queue_negsampling_efficiently, get_data_queue_subsampling_efficiently, get_data_queue_subsampling_efficiently_explicit

# parser
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
parser.add_argument('--device', type=int, default=4, help='GPU device')


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


class MetaClassifier:
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

        self.ohenc = OneHotEncoder()
        # self.quantile_transformer = QuantileTransformer()
        self.scaler_enc = StandardScaler()
        # self.classifier = None
        self.regressor = None
        if self.meta_mode == 'bore_rf':
            # self.classifier = RandomForestClassifier()
            self.regressor = RandomForestRegressor(n_estimators=200)
            # self.regressor = AdaBoostRegressor()

    def data_init(self, hparams_list, performance_list):
        self.hparams_list = hparams_list
        self.performance_list = performance_list
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
        # print("q: {}, tau: {}".format(q, tau))
        self.labels = np.less(self.targets, tau) # y<=tau => 1; y>tau => 0 
        self.labels = np.array(self.labels, dtype='int')
        return

        
    def update_data_density(self, new_hparam, new_performance):
        self.hparams_list.append(new_hparam)
        self.performance_list.append(new_performance)
        self.data_size = len(hparams_list)

        data_list = [{'opt': self.hparams_list[i][0], 'lr': self.hparams_list[i][1], 'embedding_dim': self.hparams_list[i][2], 'weigh_decay': self.hparams_list[i][3], 'performance': self.performance_list[i] } for i in range(self.data_size)]
        df = pd.DataFrame(data_list)
        df.drop(columns=['performance', 'opt'], inplace=True)

        opt_list = np.array([self.hparams_list[i][0] for i in range(len(self.hparams_list))])
        features_opt = self.ohenc.fit_transform(opt_list.reshape(-1, 1))
        features_opt = np.array(features_opt.todense())
        features_other = self.scaler_enc.fit_transform(df)
        self.features = np.concatenate((features_opt, features_other), axis=1)

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
            # TODO: how to train a mlp classifier with non-float hparams type, it can't be graded down
            # hyper parameters need s
            pass
        elif self.meta_mode == 'bore_rf':
            # TODO: construct a bore + rf meta learner, consist of a bore and a random forest classifier
            X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.1, random_state=0)
            self.regressor.fit(X_train, y_train)
            # print("X_train.shape: {}".format(X_train.shape))
            score = self.regressor.score(X_test, y_test)
            result = self.regressor.predict(X_test)
            
            test_data_size = 200
            opt_list, lr_list, embedding_dim_list, weight_decay_list = generate_random_hparam(data_size=test_data_size)
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
            feature_test = np.concatenate((test_features_opt, test_features_other), axis=1)


            # feature_test = ohenc.fit_transform(df)
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
    # it is a hp tuning program on fixed model and original dataset
    
    # hparams_list = []
    # performance_list = []

    storage_nums = 64
    trial_nums = 20 # we will draw a figure: trails vs performances
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
        # doing hp optimization, we only need do it on the whole data queue
        train_queue_pair, valid_queue, test_queue = get_data_queue_negsampling_efficiently(data_path, args)
    else: # train queue，显式推荐
        train_queue, valid_queue, test_queue = get_data_queue_efficiently(data_path, args)

    # construct a model, maybe can be changed to many models and average or maximize the performance
    remaining_arches_encoding = open(args.remaining_arches, 'r').readlines() # opten the file of arch
    remaining_arches_encoding = list(map(lambda x: x.strip(), remaining_arches_encoding))
    # print("remaining_arches_encoding: {}".format(remaining_arches_encoding))
    # arch_encode_list = np.random.choice(remaining_arches_encoding, trial_nums)
    # arch_single_list = [] # 详细
    # for arch_encoding in arch_encode_list:
    #     arch_single = sample_arch_cf()
    #     arch_single['cf'], arch_single['emb']['u'], arch_single['emb']['i'], arch_single['ifc'], arch_single['pred'] = arch_encoding.split('_')
    #     # print(type(arch_single))
    #     arch_single_list.append(dict(arch_single))
    arch_start = time()
    arch_encoding = np.random.choice(remaining_arches_encoding, 1)
    # arch_encoding = ['rr_mat_mlp_plus_i']
    arch_single = sample_arch_cf()
    arch_single['cf'], arch_single['emb']['u'], arch_single['emb']['i'], arch_single['ifc'], arch_single['pred'] = arch_encoding[0].split('_')
    # '''
    # hparams choice: random start, all in numpy frame
    opt_list, lr_list, embedding_dim_list, weight_decay_list = generate_random_hparam(data_size=storage_nums)
    # store some hparams-perf pairs
    # we don't need subgraph here
    # maybe we can try some multi process technique here
    hostname = socket.gethostname()
    print("hostname: {}".format(hostname))
    
    avaliable_device_ids = GPUtil.getAvailable(order = 'first', limit = 8, maxLoad = 0.5, maxMemory = 0.2, includeNan=False, excludeID=[], excludeUUID=[])
    avaliable_device_ids = [6,7] # gpu
    print("host {} avaliable_device_ids: {}".format(hostname, avaliable_device_ids))
    # avaliable_device_ids = avaliable_device_ids[:3]
    assigned_device_ids = [val for val in avaliable_device_ids for i in range(2)]  # final gpu-ids x 3
    task_number = math.ceil(storage_nums / len(assigned_device_ids))  # in each task, three thread on one GPU
    task_split = list(range(0, storage_nums, len(assigned_device_ids)))
    task_split.append(storage_nums)
    task_index = [list(range(task_split[i], task_split[i+1])) for i in range(task_number)]

    rmse_list1 = []
    for tasks in task_index:
        with mp.Pool(processes=len(tasks)) as p:
            print('\nStage1: getting storage')
            p.name = 'test on {}'.format(arch_single)
            if args.data_type == 'implicit':# 装载
                args.num_users, args.num_items = num_users, num_items
                jobs = []
                for i in tasks:
                    curr_hparams = opt_list[i], lr_list[i], embedding_dim_list[i], weight_decay_list[i]
                    args.opt, args.lr, args.embedding_dim, args.weight_decay = curr_hparams
                    jobs.append([arch_single, num_users, num_items, [], valid_queue, test_queue, train_queue_pair, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid])
                # jobs = [[arch_single, num_users, num_items, [], valid_queue, test_queue, train_queue_pair, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] for i in tasks]
                rmse_list1 += p.map(get_single_model_performance, jobs)

            else: # explicit
                args.num_users, args.num_items = num_users, num_items
                jobs = [[arch_single, num_users, num_items, train_queue, valid_queue, test_queue, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] for i in tasks]
                rmse_list1 += p.map(get_single_model_performance, jobs)
            p.close()
            p.terminate()
    hparams_list = [[opt_list[i], lr_list[i], embedding_dim_list[i], weight_decay_list[i]] for i in range(storage_nums)]
    if args.data_type == 'implicit':
        performance_list = [rmse[0] for rmse in rmse_list1]
    else:
        performance_list = rmse_list1
    print("hparams_list: {}".format(hparams_list))
    print("performance_list: {}".format(performance_list))
    max_perf_idx, max_perf = np.argmax(performance_list), np.max(performance_list)
    print('best hp: {}, performance: {}'.format(hparams_list[max_perf_idx], max_perf))

    # save storage data
    # results_dict = dict()
    # results_dict['performance_list'] = performance_list
    # results_dict['hparams_list'] = hparams_list
    # json_str = json.dumps(results_dict, indent=4)
    # with open('bore_rf_meta_log.json', 'w') as json_file:
    #     json_file.write(json_str)
    with open('bore_rf_meta_log.txt', 'a') as f:
        # f.write("hparams_list: {}\n".format(hparams_list))
        # f.write("performance_list: {}".format(performance_list))
        f.write("{}\n{}\n".format(hparams_list, performance_list))
    # end of storing data stage
    '''
    with open('bore_rf_meta_log1.json','r') as f:
        data = json.load(f)
    performance_list = data['performance_list']
    hparams_list = data['hparams_list']
    '''
    # setting meta_learner
    meta_classifier = MetaClassifier(meta_mode='bore_rf')
    meta_classifier.data_init(hparams_list, performance_list)
    hparams_next = meta_classifier.get_next_hparams()
    print("hparams_next: {}".format(hparams_next))
    # meta_classifier.classifier.fit(hparams_list, performance_list)

    # optimizing hparams by surrogate model 
    # we don't need different host in this stage
    rmse_list1 = []
    for i in range(trial_nums):
        # score = get_arch_performance_cf_signal_param_device(args)
        curr_hparams = hparams_next
        args.opt, args.lr, args.embedding_dim, args.weight_decay = hparams_next
        if args.data_type == 'implicit':# 装载
            args.num_users, args.num_items = num_users, num_items
            jobs = [arch_single, num_users, num_items, [], valid_queue, test_queue, train_queue_pair, args, [curr_hparams[1], curr_hparams[2]], args.device, args.if_valid]
            rmse_list1.append(list(get_single_model_performance(jobs)))

        else: # explicit
            args.num_users, args.num_items = num_users, num_items
            jobs = [arch_single, num_users, num_items, train_queue, valid_queue, test_queue, args, [curr_hparams[1], curr_hparams[2]], args.device, args.if_valid] 
            rmse_list1.append(list(get_single_model_performance(jobs)))

        
        new_performance = rmse_list1[i][0] # should be generated by model
        print("curr_hparams: {}, new_performance: {}".format(curr_hparams,new_performance))
        meta_classifier.update_data_density(hparams_next, new_performance)
        hparams_next = meta_classifier.get_next_hparams()
        # performance_list = rmse_list1

    recall20_list = [p[0] for p in rmse_list1]
    print(recall20_list)
    print(performance_list)
    plt.plot(list(range(1, trial_nums+1, 1)), recall20_list)
    # plt.plot(list(range(1, trial_nums+1, 1)), [max_perf, max_perf], )
    plt.axhline(max_perf)
    plt.xlabel('trials')
    plt.ylabel('recall@20')
    plt.title('tuning on model {}'.format(arch_encoding[0]))
    plt.savefig(os.path.join('bore_meta', 'test_bore_rf_{}.jpg'.format(arch_encoding[0])))
    plt.show()
    plt.close()
    
