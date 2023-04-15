import GPUtil
import socket
import math
from time import localtime, sleep, strftime, time
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'


import numpy as np
import argparse
import torch
import torch.utils
from torch import multiprocessing as mp # 多线程工
from scipy import stats

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


if __name__ == '__main__':
    # torch.set_default_tensor_type(torch.FloatTensor)
    anchor_config_num = args.anchor_num
    # hp
    opt_list = np.random.choice(['Adagrad', 'Adam'], anchor_config_num)
    weight_decay_list = np.random.uniform(low=1e-4, high=1e-1, size=anchor_config_num)
    lr_list = np.random.uniform(low=1e-4, high=1e-2, size=anchor_config_num)
    embedding_dim_list = np.random.choice(range(1, 64+1, 1), anchor_config_num)
    # arch
    remaining_arches_encoding = open(args.remaining_arches, 'r').readlines() # opten the file of arch
    remaining_arches_encoding = list(map(lambda x: x.strip(), remaining_arches_encoding))
    # print("remaining_arches_encoding: {}".format(remaining_arches_encoding))
    arch_encode_list = np.random.choice(remaining_arches_encoding, anchor_config_num)

    performance = {}
    # setting datasets,  default='ml-100k'
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
    num_user_sub = num_users
    num_item_sub = num_items

    data_path = args.dataset + '/'
    if args.data_type == 'implicit': # 主要使用这一行，隐式推荐
        train_queue_pair, valid_queue, test_queue = get_data_queue_negsampling_efficiently(data_path, args)
        # train_queue_pair_sub, valid_queue_sub, test_queue_sub = get_data_queue_subsampling_efficiently(data_path, args)
        num_user_sub = args.num_users
        num_item_sub = args.num_items
        print('implicit', num_user_sub, num_item_sub)
    else: # train queue，显式推荐
        train_queue, valid_queue, test_queue = get_data_queue_efficiently(data_path, args)
        # train_queue_sub, valid_queue_sub, test_queue_sub = get_data_queue_subsampling_efficiently_explicit(data_path, args)
        num_user_sub = args.num_users
        num_item_sub = args.num_items
        print('explicit', num_user_sub, num_item_sub)

    
    arch_single_list = [] # 详细
    for arch_encoding in arch_encode_list:
        arch_single = sample_arch_cf()
        arch_single['cf'], arch_single['emb']['u'], arch_single['emb']['i'], arch_single['ifc'], arch_single['pred'] = arch_encoding.split('_')
        # print(type(arch_single))
        arch_single_list.append(dict(arch_single))
    # performance[str(arch_single)] = []
    # print(type(arch_single_list[0]))

    arch_start = time()
    hostname = socket.gethostname()
    print("hostname: {}".format(hostname))
    if hostname == 'fib':
        avaliable_device_ids = [0, 1, 2, 3]
    elif hostname == 'abc':
        # avaliable_device_ids = [0,1,2,3,4,5,6,7]
        avaliable_device_ids = GPUtil.getAvailable(order='first', limit=8, maxLoad=0.5, maxMemory=0.2, includeNan=False, excludeID=[], excludeUUID=[])
    else:
        pass
    print("host abc avaliable_device_ids: {}".format(avaliable_device_ids))
    if len(avaliable_device_ids) < 2:
        avaliable_device_ids = [4,6] # irl
        print("Not enough avaliable_device_ids, we use default: {}!".format(avaliable_device_ids))
    elif len(avaliable_device_ids) >= 5:
        avaliable_device_ids = avaliable_device_ids[:4]
    avaliable_device_ids = [3, 5] # irl
    print("host abc avaliable_device_ids: {}".format(avaliable_device_ids))
    # avaliable_device_ids = avaliable_device_ids[:3]
    assigned_device_ids = avaliable_device_ids# final gpu-ids
    task_number = math.ceil(anchor_config_num / len(assigned_device_ids)) #anchor_config_num总任务数量
    task_split = list(range(0, anchor_config_num, len(assigned_device_ids)))
    task_split.append(anchor_config_num)
    task_index = [list(range(task_split[i], task_split[i+1])) for i in range(task_number)]
    
    # doing 16 tasks - mutiprocess for total dataset
    rmse_list1 = []
    rmse_list_sub = []
    for tasks in task_index:
        with mp.Pool(processes=len(tasks)) as p:
            print('\nStage1')
            p.name = 'test'
            if args.data_type == 'implicit':# 装载
                args.num_users, args.num_items = num_users, num_items
                jobs = [[arch_single_list[i], num_users, num_items, [], valid_queue, test_queue, train_queue_pair, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] for i in tasks]
                rmse_list1 += p.map(get_single_model_performance, jobs)

                args.num_users, args.num_items = num_user_sub, num_item_sub
                jobs_sub = [[arch_single_list[i], num_user_sub, num_item_sub, [], valid_queue_sub, test_queue_sub, train_queue_pair_sub, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] for i in tasks]
                # train_queue_pair_sub, valid_queue_sub, test_queue_sub
                rmse_list_sub += p.map(get_single_model_performance, jobs_sub)

            else: # explicit
                args.num_users, args.num_items = num_users, num_items
                jobs = [[arch_single_list[i], num_users, num_items, train_queue, valid_queue, test_queue, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] for i in tasks]
                rmse_list1 += p.map(get_single_model_performance, jobs)

                args.num_users, args.num_items = num_user_sub, num_item_sub
                jobs_sub = [[arch_single_list[i], num_user_sub, num_item_sub, train_queue_sub, valid_queue_sub, test_queue_sub, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] for i in tasks]
                rmse_list_sub += p.map(get_single_model_performance, jobs_sub)
            p.close()
    
    # print('rmse_list1: {}'.format(rmse_list1))
    # print('rmse_list_sub: {}'.format(rmse_list_sub))
    srcc_sp = 0.0
    if args.data_type == 'implicit': 
        recall20_list = np.array([item[0] for item in rmse_list1])
        recall20_list_sub = np.array([item[0] for item in rmse_list_sub])
        srcc_sp = stats.spearmanr(recall20_list, recall20_list_sub).correlation
    else:
        rmse_list = np.array(rmse_list1)
        rmse_list_sub = np.array(rmse_list_sub)
        srcc_sp = stats.spearmanr(rmse_list, rmse_list_sub).correlation
        
    print("sample_portion: {}, srcc: {}".format(args.sample_portion, srcc_sp))
    save_dir = 'subsample_srcc' + '_' + args.dataset + '_' + args.sample_mode + '_' + args.data_type + '.txt'
    with open(save_dir, 'a') as f:
        f.write('{}, {}, {}\n'.format(args.anchor_num, args.sample_portion, srcc_sp))

    
