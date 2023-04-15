import argparse
from audioop import rms
import logging
import os
from re import L
import sys
from itertools import product
from time import localtime, sleep, strftime, time

import json
import numpy as np
import setproctitle # to set the name of process
import torch
import torch.utils
from tensorboardX import SummaryWriter
from torch import multiprocessing as mp # 多线程工作

from dataset import get_data_queue_cf, get_data_queue_cf_nonsparse, get_data_queue_efficiently, get_data_queue_negsampling_efficiently, get_data_queue_subsampling_efficiently
from models import (CML, DELF, DMF, FISM, GMF, MLP, SVD, JNCF_Cat, JNCF_Dot, SVD_plus_plus, SPACE, BaseModel, Virtue_CF)
from controller import sample_arch_cf, sample_arch_cf_signal, sample_arch_cf_test
from train_eval import (evaluate_cf, evaluate_cf_efficiently, evaluate_cf_efficiently_implicit, get_arch_performance_cf_signal_param_device, get_arch_performance_single_device, train_single_cf, train_single_cf_efficiently,get_arch_performance_implicit_single_device,evaluate_cf_efficiently_implicit_minibatch)

import GPUtil
import socket
import math

parser = argparse.ArgumentParser(description="Run.")
parser.add_argument('--lr', type=float, default=0.05, help='init learning rate')
parser.add_argument('--arch_lr', type=float, default=0.05, help='learning rate for arch encoding')
parser.add_argument('--controller_lr', type=float, default=1e-1, help='learning rate for controller')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--update_freq', type=int, default=1, help='frequency of updating architeture')
parser.add_argument('--opt', type=str, default='Adagrad', help='choice of opt')
parser.add_argument('--use_gpu', type=int, default=1, help='whether use gpu')
parser.add_argument('--minibatch', type=int, default=1, help='whether use minibatch')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--train_epochs', type=int, default=2000, help='num of training epochs')
parser.add_argument('--search_epochs', type=int, default=1000, help='num of searching epochs')
parser.add_argument('--save', type=str, default='save/', help='experiment name')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--valid_portion', type=float, default=0.25, help='portion of validation data')
parser.add_argument('--dataset', type=str, default='ml-100k', help='dataset')
parser.add_argument('--mode', type=str, default='random_single', help='search or single mode')
parser.add_argument('--process_name', type=str, default='AutoCF@wenyan', help='process name')
parser.add_argument('--embedding_dim', type=int, default=2, help='dimension of embedding')
parser.add_argument('--controller', type=str, default='PURE', help='structure of controller')
parser.add_argument('--controller_batch_size', type=int, default=4, help='batch size for updating controller')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--max_batch', type=int, default=65536, help='max batch during training')
parser.add_argument('--device', type=int, default=0, help='GPU device')
parser.add_argument('--multi', type=int, default=0, help='using multi-training for single architecture')
parser.add_argument('--if_valid', type=int, default=1, help='use validation set for tuning single architecture or not')
parser.add_argument('--breakpoint', type=str, default='save/log.txt', help='the log file storing existing results')
parser.add_argument('--arch_file', type=str, default='src/arch.txt', help='all arches')
parser.add_argument('--remaining_arches', type=str, default='src/arch.txt', help='')
parser.add_argument('--arch_assign', type=str, default='[0,3]', help='')
parser.add_argument('--data_type', type=str, default='implicit', help='explicit or implicit(default)')
parser.add_argument('--loss_func', type=str, default='bprloss', help='Implicit loss function')
parser.add_argument('--mark', type=str, default='') # 
parser.add_argument('--batch_size', type=int, default=5000, help='batch size')
parser.add_argument('--file_id', type=int, default=100, help='file id')
parser.add_argument('--sample_portion', type=float, default=0.20, help='portion of data subgraph')
args = parser.parse_args()
mp.set_start_method('spawn', force=True) # 一种多任务运行方法
# torch.multiprocessing.set_start_method('spawn', force=True)

# 一些辅助函数
def get_hyperparam_performance(x):
    arch, num_users, num_items, train_queue, test_queue, args, param, device_id = x

    return get_arch_performance_cf_signal_param_device(arch, num_users, num_items, train_queue, test_queue, args, param, device_id)

def get_single_model_performance(x):
    if len(x) == 10: # for explicit
        arch, num_users, num_items, train_queue, valid_queue, test_queue, args, param, device_id, if_valid = x
        return get_arch_performance_single_device(arch, num_users, num_items, train_queue, valid_queue, test_queue, args, param, device_id, if_valid)
    else: # for implicit
        arch, num_users, num_items, train_queue, valid_queue, test_queue, train_queue_pair, args, param, device_id, if_valid = x
        return get_arch_performance_implicit_single_device(arch, num_users, num_items, train_queue, valid_queue, test_queue, train_queue_pair, args, param, device_id, if_valid)


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.FloatTensor)
    setproctitle.setproctitle(args.process_name) # 设定进程名称
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    log_format = '%(asctime)s %(message)s' # 记录精确的实践
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, filemode='w', format=log_format, datefmt='%m/%d %I:%M:%S %p')
    current_time = strftime("%Y-%m-%d-%H:%M:%S", localtime())
    # args.save = 'save/'
    save_name = args.mode + '_' + args.dataset + '_' + str(args.embedding_dim) + '_' + args.opt + str(args.lr)
    save_name += '_' + str(args.data_type)

    if args.mode == 'reinforce':
        save_name += '_' + str(args.controller_lr) + '_' + args.controller + '_' + str(args.controller_batch_size)
    else:
        # save_name += '_' + str(args.weight_decay) # default=1e-5
        save_name += '_' + ('%.6f' % (args.weight_decay)) 
    save_name += '_' + str(args.seed) # default=1
    save_name += '_' + current_time
    
    # 创建log路径
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.exists(args.save + '/log'):
        os.makedirs(args.save + '/log')
    if not os.path.exists(args.save + '/log_sub'):
        os.makedirs(args.save + '/log_sub')
    if os.path.exists(os.path.join(args.save, save_name + '.txt')):
        os.remove(os.path.join(args.save, save_name + '.txt'))
    
    # fh表示
    fh = logging.FileHandler(os.path.join(args.save + 'log', save_name + '.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    writer = SummaryWriter(log_dir=args.save + 'tensorboard/{}'.format(save_name))
    if args.use_gpu: # default = True
        torch.cuda.set_device(args.gpu)
        logging.info('gpu device = %d' % args.gpu)
    else:
        logging.info('no gpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_start = time()
    dim = 2
    data_path = args.dataset + '/'

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
    elif args.dataset == 'youtube_small':
        num_ps = 600
        num_qs = 14340
        num_rs = 5
        dim = 3
    elif args.dataset == 'youtube':
        num_ps = 15088
        num_qs = 15088
        num_rs = 5
        dim = 3
    elif args.dataset == 'amazon-book':
        num_users = 11899
        num_items = 16196
    elif args.dataset == 'yelp':
        num_users = 6102
        num_items = 18599 #  density: 0.3926%
    elif args.dataset == 'yelp2':
        num_users = 15496
        num_items = 12666
        # yelp generated from: https://www.kaggle.com/yelp-dataset/yelp-dataset
    elif args.dataset == 'yelp-10k':
        num_users = 9357-1
        num_items = 4299
    elif args.dataset == 'yelp-50k':
        num_users = 42919-1
        num_items = 9033
    elif args.dataset == 'yelp-100k':
        num_users = 80388-1
        num_items = 11223
    elif args.dataset == 'yelp-1m':
        num_users = 551747-1
        num_items = 28085
    elif args.dataset == 'yelp-10m':
        num_users = 1483546-1
        num_items = 90315
    elif args.dataset == 'yelp-all':
        num_users = 1483546-1
        num_items = 90315
    elif args.dataset == 'yelp-all2':
        num_users = 6102-1
        num_items = 18599
    else:
        pass
    args.num_users = num_users
    args.num_items = num_items

    if args.data_type == 'implicit': # 主要使用这一行，隐式推荐
        # train_queue_pair, valid_queue, test_queue = get_data_queue_negsampling_efficiently(data_path, args)
        train_queue_pair, valid_queue, test_queue = get_data_queue_subsampling_efficiently(data_path, args)
    else: # train queue，显式推荐
        train_queue, valid_queue, test_queue = get_data_queue_efficiently(data_path, args)
        # train_queue_pair, valid_queue, test_queue = get_data_queue_subsampling_efficiently(data_path, args)
    # print(train_queue)
    logging.info('prepare data finish! [%f]' % (time()-data_start))
    stored_arches = {} # log ging表示添加到记录中
    # python main.py --mode GMF --dataset ml-100k --data_type implicit --gpu 5 --device 5 
    # python main.py --mode random_single --dataset ml-100k --data_type implicit --gpu 5 --device 5  

    # 分不同的mode运行代码  default='random_single', help='search or single mode'
    # print("args.mode: {}".format(args.mode)) # 应该只要运行一次即可
    print('Debug in model {}, sample_portion={}'.format(args.mode, args.sample_portion))
    
    
    # 单个模型的NAS，采用random search，第一阶段用不到
    if args.mode == 'random_single':
        num_users = args.num_users 
        num_items = args.num_items 
        search_start = time()
        performance = {}
        best_arch, best_rmse = None, 100000
        arch_batch_size = 1  
        args.search_epochs = min(args.search_epochs, SPACE) # SPACE=135
    
        remaining_arches_encoding = open(args.remaining_arches, 'r').readlines() # opten the file of arch
        remaining_arches_encoding = list(map(lambda x: x.strip(), remaining_arches_encoding))
        if not args.arch_assign:
            remaining_arches_encoding = remaining_arches_encoding
        else:
            start, end = eval(args.arch_assign)
            remaining_arches_encoding = remaining_arches_encoding[start:end]
        arch_count = 0
        print("remaining_arches_encoding: {}".format(remaining_arches_encoding))
        while True:
            # 对不同的arch进行采样，可以选择遍历所有的arch， 数量为arch_count加到len(remaining_arches_encoding)
            # 循环内部arch不变
            if arch_count >= len(remaining_arches_encoding):
                break
            # sample an arch
            arch_encoding = remaining_arches_encoding[arch_count]
            arch_single = sample_arch_cf()
            arch_single['cf'], arch_single['emb']['u'], arch_single['emb']['i'], arch_single['ifc'], arch_single['pred'] = arch_encoding.split('_')
            arch_count += 1
            
            
            if len(performance) >= len(remaining_arches_encoding):
                break
            performance[str(arch_single)] = []

            arch_start = time()
            avaliable_device_ids = [0,1,2,3]
            hostname = socket.gethostname()
            print("hostname: {}".format(hostname))
            if hostname == 'rl3':
                avaliable_device_ids = [0, 1, 2, 3]
            elif hostname == 'fib-dl':
                avaliable_device_ids = [0, 1, 2, 3]
            elif hostname == 'fib-dl3':
                avaliable_device_ids = [2,3,5]
            elif hostname == 'fib':
                avaliable_device_ids = [0, 1, 2, 3]
            elif hostname =='rl2':
                avaliable_device_ids = [0, 1, 2, 3]
            elif hostname == 'abc':
                # avaliable_device_ids = [4,5,6,7]
                avaliable_device_ids = [0,1,2,3,4,5,6,7]
            else:
                pass

            # lr_candidates = [0.01, 0.02, 0.05, 0.1] #
            # rank_candidates = [2, 4, 8, 16] # embeding_dim #后面的16来源于两者之积
            lr_candidates = [0.001, 0.005, 0.01, 0.05, 0.1] #
            rank_candidates = [1, 2, 4, 8, 16] # embeding_dim #后面的16来源于两者之积
            hyper_parameters = list(product(lr_candidates, rank_candidates)) #product 笛卡尔积
            run_one_model = 0
            anchor_config_num = len(hyper_parameters)


            while True:
                # 对于每一个arch，这个while应该只循环一次
                avaliable_device_ids = GPUtil.getAvailable(order = 'first', limit = 8, maxLoad = 0.5, maxMemory = 0.2, includeNan=False, excludeID=[], excludeUUID=[])
                if hostname == 'fib':
                    avaliable_device_ids = [0, 1, 2, 3]
                elif hostname == 'abc':
                    avaliable_device_ids = [0,1,2,3,4,5,6,7]
                    avaliable_device_ids = GPUtil.getAvailable(order = 'first', limit = 8, maxLoad = 0.5, maxMemory = 0.2, includeNan=False, excludeID=[], excludeUUID=[])
                else:
                    pass
                
                print("host abc avaliable_device_ids: {}".format(avaliable_device_ids))
                if len(avaliable_device_ids) == 0:
                    avaliable_device_ids = [0, 2] # 默认的两张卡，最少情况，程序一定要跑下去!不能中途中断！
                    print("Not enough avaliable_device_ids, we use default: {}!".format(avaliable_device_ids))

                assigned_device_ids = avaliable_device_ids# final gpu-ids
                if run_one_model > 0:
                    print('已经跑完这个模型')
                    break # 代表单独的一个arch跑完了所有的tasks, 
                task_number = math.ceil(anchor_config_num / len(assigned_device_ids)) 
                task_split = list(range(0, anchor_config_num, len(assigned_device_ids)))
                task_split.append(anchor_config_num)
                task_index = [list(range(task_split[i], task_split[i+1])) for i in range(task_number)]
                # doing 16 tasks - mutiprocess
                for tasks in task_index:
                    with mp.Pool(processes=len(tasks)) as p:
                        print('Stage1')
                        p.name = 'test'
                        if args.data_type == 'implicit':# 装载
                            jobs = [[arch_single, num_users, num_items, [], valid_queue, test_queue, train_queue_pair, args, hyper_parameters[i], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] for i in tasks]
                        else:
                            jobs = [[arch_single, num_users, num_items, train_queue, valid_queue, test_queue, args, hyper_parameters[i], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] for i in tasks]
                        rmse_list1 = p.map(get_single_model_performance, jobs)
                        # performance[str(arch_single)].append(rmse_list1)
                        performance[str(arch_single)] += rmse_list1
                        run_one_model += 1
                        p.close()
                
                if len(performance[str(arch_single)]) < anchor_config_num:
                    print('大小出问题')

                # 这一部分不知道干什么
                # for k in range(len(hyper_parameters)):
                #     arch_encoding = '{}_{}_{}_{}_{}'.format(arch_single['cf'], arch_single['emb']['u'], arch_single['emb']['i'], arch_single['ifc'], arch_single['pred'])
        # recall20, recall10
        info_json = json.dumps(performance,sort_keys=False, indent=4, separators=(',', ': '))
        # f = open(os.path.join(args.save, 'random_nas_perf17.json' ), 'w')
        f = open(os.path.join(args.save, 'subgraph'+'_'+str(args.sample_portion)
        +'_'+'random_nas_'+args.dataset+'_'+args.data_type+'_perf_'+str(args.file_id)+'.json' ), 'w')
        f.write(info_json)
                

    
    else:
        print('No such mode.')
    print(save_name)
