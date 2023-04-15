import argparse
import logging
import os
import sys
from time import localtime, sleep, strftime, time
import json
import random

import numpy as np
import setproctitle # to set the name of process
import torch
import torch.utils
from tensorboardX import SummaryWriter
from torch import multiprocessing as mp # 多线程工作

from dataset import get_data_queue_efficiently, get_data_queue_negsampling_efficiently
from models import (SPACE)
from controller import sample_arch_cf, sample_arch_cf_signal, sample_arch_cf_test
from main import get_single_model_performance

import GPUtil
import socket
import math

parser = argparse.ArgumentParser(description="Test Run.")
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
parser.add_argument('--anchor_hp', type=str, default='embedding_dim', help='setting an anchor hyper-parameter')
parser.add_argument('--anchor_num', type=int, default=6, help='size of anchor sample number')
parser.add_argument('--file_id', type=int, default=206, help='size of anchor sample number')


args = parser.parse_args()
# args, unknown = parser.parse_known_args()
mp.set_start_method('spawn', force=True) # 一种多任务运行方法
os.system("export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'")


if __name__ == '__main__':
    logfilePath = './weight_random/'  
    if not os.path.exists(logfilePath):
        os.makedirs(logfilePath)

    args.save = logfilePath
    args.mode = 'wd_random_nas'
    # for lr test random run
    anchor_config_num = args.anchor_num
    torch.set_default_tensor_type(torch.FloatTensor)
    setproctitle.setproctitle(args.process_name) # 设定进程名称
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    log_format = '%(asctime)s %(message)s' # 记录精确的实践
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, filemode='w', format=log_format, datefmt='%m/%d %I:%M:%S %p')
    current_time = strftime("%Y-%m-%d-%H:%M:%S", localtime())
    

    save_name = args.mode + '_' + args.dataset + '_' + str(args.embedding_dim) + '_' + args.opt + str(args.lr)
    save_name += '_' + str(args.data_type)
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
        # num_users = 715
        # num_items = 653
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
        # num_users = 26829
        # num_items = 20344
        # 31668, item_set: 1237259
        num_users = 31668 
        num_items = 38048
    elif args.dataset == 'yelp2':
        num_users = 15496
        num_items = 12666
    # following yelp generated from: https://www.kaggle.com/yelp-dataset/yelp-dataset
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
    else:
        pass
    args.num_users = num_users
    args.num_items = num_items

    if args.data_type == 'implicit': # 主要使用这一行，隐式推荐
        train_queue_pair, valid_queue, test_queue = get_data_queue_negsampling_efficiently(data_path, args)
    else: # train queue，显式推荐
        train_queue, valid_queue, test_queue = get_data_queue_efficiently(data_path, args)
    logging.info('prepare data finish! [%f]' % (time()-data_start))


    performance = {}
    best_arch, best_rmse = None, 100000
    arch_batch_size = 1  
    args.search_epochs = min(args.search_epochs, SPACE) # SPACE=135

    remaining_arches_encoding = open(args.remaining_arches, 'r').readlines() # opten the file of arch
    remaining_arches_encoding = list(map(lambda x: x.strip(), remaining_arches_encoding))
    remaining_arches_encoding = random.sample(remaining_arches_encoding, anchor_config_num)
    
    arch_count = 0
    print("remaining_arches(in hp analysis with random arch): {}".format(remaining_arches_encoding))
    # embs_hp_list = ['1','2', '4','8','16','32','64', '128', '256', '512']
    weight_decay_hp_list = ['1e-06','1e-05', '0.0001','0.001','0.01']
    weight_decay_result_list_dict = {}
    for wd in weight_decay_hp_list:
        weight_decay_result_list_dict[wd] = []


    opt_list = np.random.choice(['Adagrad', 'Adam'], anchor_config_num)
    # weight_decay_list = np.random.uniform(low=1e-4, high=1e2, size=anchor_config_num)
    lr_list = np.random.uniform(low=1e-2, high=2.0, size=anchor_config_num)
    embedding_dim_list = np.random.choice(range(1, 64+1, 1), anchor_config_num)
    
    hyper_parameters = []
    for i in range(anchor_config_num):
        hparam = [0.0, args.embedding_dim]
        hyper_parameters.append(hparam)

    arch_single_list = []
    for arch_count in range(anchor_config_num):
        arch_encoding = remaining_arches_encoding[arch_count]
        arch_single = sample_arch_cf()
        arch_single['cf'], arch_single['emb']['u'], arch_single['emb']['i'], arch_single['ifc'], arch_single['pred'] = arch_encoding.split('_')
        arch_count += 1
        arch_single_list.append(arch_single)
        # performance[str(arch_single)] = 0
    # if len(performance) >= len(remaining_arches_encoding):
    #     break

    arch_start = time()
    avaliable_device_ids = [0,1,2,3]
    hostname = socket.gethostname()
    print("hostname: {}".format(hostname))
    if hostname == 'abc':
        avaliable_device_ids = [0,1,2,3,4,5,6,7]
    else:
        pass

    # run_one_model = 0
    for wd in weight_decay_hp_list:
        avaliable_device_ids = GPUtil.getAvailable(order = 'first', limit = 8, maxLoad = 0.5, maxMemory = 0.2, includeNan=False, excludeID=[], excludeUUID=[])
        if hostname == 'fib':
            avaliable_device_ids = [0, 1, 2, 3]
        elif hostname == 'abc':
            avaliable_device_ids = [0, 1, 2, 3]
            avaliable_device_ids = GPUtil.getAvailable(order = 'first', limit = 8, maxLoad = 0.5, maxMemory = 0.2, includeNan=False, excludeID=[], excludeUUID=[])
            # avaliable_device_ids = [0,1,2,3,4,5,6,7]
        else:
            pass
        # avaliable_device_ids = GPUtil.getAvailable(order = 'first', limit = 8, maxLoad = 0.5, maxMemory = 0.2, includeNan=False, excludeID=[], excludeUUID=[])
        print('|------ START of weight_decay ({}) optimization ------|'.format(wd))
        args.weight_decay = float(wd)
        print("abc avaliable_device_ids: {}".format(avaliable_device_ids))
        if len(avaliable_device_ids) == 0:
            avaliable_device_ids = [0, 2] # 默认的两张卡，最少情况，程序一定要跑下去!不能中途中断！
            print("Not enough avaliable_device_ids, we use default: {}!".format(avaliable_device_ids))
            # sys.exit(1)
        # if run_one_model > 0:
        #     break # 代表单独的一个arch跑完了所有的tasks, 就可以结束了
    
        assigned_device_ids = avaliable_device_ids
        # 16表示对一个arch进行的任务数量
        # 这里改为anchor_num
        # task_number表示一张卡上的任务量
        task_number = math.ceil(anchor_config_num / len(assigned_device_ids)) 
        task_split = list(range(0, anchor_config_num, len(assigned_device_ids)))
        task_split.append(anchor_config_num)
        task_index = [list(range(task_split[i], task_split[i+1])) for i in range(task_number)]

        # doing anchor_num tasks - mutiprocess
        for tasks in task_index: #下面对同一种架构进行不同超参的并行 每个task对应在不同GPU上的工作
            with mp.Pool(processes=len(tasks),maxtasksperchild=1) as p:
                print('weight_decay: {}, subtasks start!, tasks: {} on gpu-ids {}'.format(wd, tasks, assigned_device_ids))
                p.name = 'test'
                if args.data_type == 'implicit':# 装载任务
                    jobs = [[arch_single_list[i], num_users, num_items, [], valid_queue, test_queue, train_queue_pair, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] for i in tasks]
                else:
                    jobs = [[arch_single_list[i], num_users, num_items, train_queue, valid_queue, test_queue, args, [lr_list[i], embedding_dim_list[i]], assigned_device_ids[i % len(assigned_device_ids)], args.if_valid] for i in tasks] # 这一组task的lr都相同，其它随机，按照顺序
                rmse_list1 = p.map(get_single_model_performance, jobs) # 返回的数值是tasks的合计
                print('weight_decay: {}, rmse_list1: {}'.format(wd, rmse_list1))
                print("task {} ended\n".format(tasks))
                # print() # 每一个lr对应一组rmse_list（实际上也可能是recall list）
                # run_one_model += 1
                # lr_result_list_dict
                recall_list1 = [rmse1[0] for rmse1 in rmse_list1]
                weight_decay_result_list_dict[wd] += recall_list1
                p.close()
        print('weight decay',wd,' end!','weight_decay_result_list_dict', weight_decay_result_list_dict)
        if len(weight_decay_result_list_dict[wd]) < anchor_config_num:
            print('大小出问题')
        print('|------  END of weight_decay ({}) optimization  ------|\n\n\n'.format(wd))
    
    
    print('weight_decay_result_list_dict', weight_decay_result_list_dict)
    info_json = json.dumps(weight_decay_result_list_dict,sort_keys=False, indent=4, separators=(',', ': '))
    # 显示数据类型
    # print(type(info_json))
    f = open(logfilePath+'wdinfo'+str(args.file_id)+'.json', 'w')
    f.write(info_json)
    
    




    