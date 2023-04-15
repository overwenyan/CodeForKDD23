import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
# from torch import multiprocessing as mp # 多线程工作
from single_model import single_model_run
from time import localtime, strftime

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

args = parser.parse_args()
# mp.set_start_method('spawn', force=True) # 一种多任务运行方法
if __name__ == '__main__':
    logfilePath = './val_structure/'  
    if not os.path.exists(logfilePath):
        os.makedirs(logfilePath)
    
    # # for lr test random run
    # lr_list = np.logspace(-3,3,7,base=10)
    # embedding_dim_list = np.logspace(0,10,11,base=2)

    lr_list = np.logspace(-3,3,5,base=10)
    embedding_dim_list = np.logspace(0,10,5,base=2)

    anchor_config_num = len(lr_list) * len(embedding_dim_list)
    opt_list = np.random.choice(['Adagrad', 'Adam'], anchor_config_num)
    weight_decay_list = np.random.uniform(low=1e-4, high=1e-1, size=anchor_config_num)

    gpu_num = 2
    args.gpu = gpu_num
    args.device = gpu_num
    args.mode = 'GMF'
    args.save = logfilePath
    # args.dataset = 'ml-1m'
    args.dataset = 'ml-100k'
    args.data_type = 'implicit'
    xx, yy = np.meshgrid(lr_list, embedding_dim_list)
    loss_z = xx + yy #设置z由xx+yy生成，当然也可以是其他函数如sin（xx,yy）等
    # print (loss_z.shape)
    # print(loss_z[0,0])
    cnt = 0
    current_time = strftime("%Y-%m-%d-%H:%M:%S", localtime())
    for i in range(len(lr_list)):
        for j in range(len(embedding_dim_list)):
            print("[i:{}|{}, j:{}|{}]".format(i,len(lr_list), j, len(embedding_dim_list)))

            # run
            # os.system('python ./main.py  --mode GMF --dataset {} --use_gpu 1 --data_type implicit --gpu {} --device {} --save {} --opt {} \
            #     --lr {:.4f} --embedding_dim {} --weight_decay {:.6f}'.format(dataset, gpu_num, gpu_num, logfilePath, opt_list[i], lr_list[i], int(embsize), weight_decay_list[i]))
            args.lr = lr_list[i]
            args.embedding_dim = int(embedding_dim_list[j])
            print("args.lr: {}, args.embedding_dim: {}".format(args.lr, args.embedding_dim))
            args.weight_decay = weight_decay_list[cnt]
            args.opt = opt_list[cnt]
            rmse_list, loss_list = single_model_run(args)
            
            # analyze
            recall20_array = np.array(rmse_list)
            loss_array = np.array(loss_list)
            max_epoch  = np.argmax(recall20_array)
            max_recall = np.max(recall20_array)
            # embedding_dim_result_list_dict[embsize].append(max_recall)
            cnt += 1
            print("[Index: {}|{}], max_epoch: {},  max_recall: {}".format(cnt,anchor_config_num, max_epoch, max_recall))
            loss_z[j][i] = np.min(loss_array)
    print("End of Train and Validation!")
    plt.contourf(xx,yy,loss_z)
    plt.contour(xx,yy,loss_z)#在z=3.5,4,4.4,5处绘制等高线，等高线位置必须递增排序
    plt.xlabel('lr')
    plt.ylabel('embedding_dim')
    plt.show()   
    plt.savefig(os.path.join('box_violin_plot', os.path.join('val_contour_ml-100k' + current_time)))
    # print('embsize_result_list_dict: {}'.format(embedding_dim_result_list_dict))
    # dumps 将数据转换成字符串
    # info_json = json.dumps(embedding_dim_result_list_dict,sort_keys=False,indent=4,separators=(',', ': '))
    # 显示数据类型
    # print(type(info_json))
    # f = open(logfilePath+'embinfo40.json', 'w')
    # f = open(logfilePath+'embinfo74.json', 'w')
    # f.write(info_json)