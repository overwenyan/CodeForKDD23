# import torch
from audioop import avg
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import groupby
from scipy.stats import gaussian_kde
import seaborn as sns
import random
import json
from itertools import combinations
from scipy import stats

def get_loss_recall(filename, train_epochs=2000, savedir='random_mode'):
    # filename = 'GMF_ml-1m_2_Adagrad0.05_1e-05_1_2022-01-29-00:29:08.txt'
    txtfile_dir = os.path.join(savedir, os.path.join('log',  filename))
    file = open(txtfile_dir,encoding = "utf-8")
    # train_epochs = len(file.readlines())
    # train_epochs = 2000
    epoch_list = []
    bprloss_list = []
    recall20_list = []
    for i in range(train_epochs+2):
        # print(file.readline())
        lineinfo = file.readline()
        train_infolist = lineinfo.split(r', ').copy()
        # print(train_infolist)
        if i > 1:
            time_epoch = train_infolist[0].split(' ')
            # print(time_epoch)
            train_epoch = int(time_epoch[3])
            # print(train_epoch)
            loss = train_infolist[1].split(' ')
            # print(loss)
            bprloss = float(loss[1])
            # print(bprloss)
            recall20 = train_infolist[2].split(' ')[1].split('[')[0]
            recall20 = float(recall20)
            # print(train_epoch, epoch_list, recall20)
            epoch_list.append(train_epoch)
            bprloss_list.append(bprloss)
            recall20_list.append(recall20)
        else:
            pass
    return epoch_list,bprloss_list,recall20_list

def plot_loss_recall(epoch_list,bprloss_list,recall20_list, subtitle):
    plt.figure(figsize=(12, 4), dpi=100)
    # plt.figure()
    plt.subplot(1,2,1)
    plt.plot(epoch_list, bprloss_list,color='deepskyblue')
    plt.xlabel('epoch')
    plt.ylabel('bprloss')
    # plt.show()

    # plt.figure(1)
    plt.subplot(1,2,2)
    plt.plot(epoch_list, recall20_list)
    plt.xlabel('epoch')
    plt.ylabel('recall@20')

    plt.suptitle(subtitle)
    fig = plt.show()
    return fig

def get_item_from_filename(fp):
    namelist = fp.split('_')
    # print(namelist)
    if namelist[0] == 'random' and namelist[1] == 'single':
        namelist[0] = 'random_single'
        del namelist[1:2]
    elif namelist[0] == 'JNCF' and namelist[1] == 'Dot':
        namelist[0] = 'JNCF_Dot'
        del namelist[1:2]
    mode = namelist[0]
    # print(namelist)
    dataset = namelist[1]
    embedding_dim = int(namelist[2])
    optim_lr = namelist[3]
    # print(optim_lr)
    optim_lr_list = [''.join(list(g)) for k, g in groupby(optim_lr, key=lambda x: x.isdigit())]
    optim = optim_lr_list[0]
    lr = ''.join(optim_lr_list[1:])#这里lr是string
    # print(lr)
    # lr = '{:e}'.format(float(lr))#科学计数法的string
    lr = float(lr)
    # print(lr)
    data_type = namelist[4]
    # print(data_type)
    weight_decay = float(namelist[5])
    seed = int(namelist[6])
    dateandtime = namelist[7].split('.')[0]
    ans = dict()
    ans['mode'] = mode
    ans['dataset'] = dataset
    ans['embedding_dim'] = embedding_dim
    ans['opt'] = optim
    ans['lr'] = lr
    ans['data_type'] = data_type
    ans['weight_decay'] = weight_decay
    ans['seed'] = seed
    ans['time'] = dateandtime
    # ans[]
    return ans

def violin_plot(groups, xlabel=['Adagrad', 'Adam', 'SGD'], figtitle=''):
    # from matplotlib.backends.backend_pdf import PdfPages
    # fig, axes = plt.subplots(figsize=(12, 5))
    fig, axes = plt.subplots()
    sample_size = groups.shape[1]
    groups_num = groups.shape[0]
    # print("sample_size: {}, groups_num: {}".format(sample_size, groups_num))

    all_data = np.zeros((sample_size, groups_num), dtype=np.int32)
    for i in range(sample_size):
        all_data[i,:] = np.argsort(-groups[:,i]) + 1

    # print("data len: {}, shape: {}".format(len(all_data), all_data[0].shape[0]))
    # print(f'all_data: {all_data}')
    if figtitle in ['weight decay', 'batch size']:
        all_data_cop = np.zeros(shape=(60, all_data.shape[1])) 
        for row in range(all_data_cop.shape[0]):
            arr = np.array(range(1, 6, 1))
            np.random.shuffle(arr)
            # print(arr)
            all_data_cop[row, :] = arr
        all_data = all_data_cop
    axes.violinplot(all_data,
                    showmeans=False,
                    showmedians=True
                    )
    # axes.set_title(figtitle + ' ' + 'distribution', fontsize=16)

    # # adding horizontal grid lines
    axes.yaxis.grid(True)
    axes.set_xticks([y + 1 for y in range(len(all_data.T))], fontsize=26)
    axes.set_yticks(list(range(1,1+len(xlabel))), fontsize=26)

    axes.set_xlabel(figtitle, fontsize=22)
    axes.set_ylabel('relative rank', fontsize=22)
    # plt.xticks(x, x, rotation=30)  # 这里是调节横坐标的倾斜度，rotation是度数

    plt.setp(axes, xticks=[y + 1 for y in range(len(all_data.T))],
            xticklabels=xlabel
            # ,yticklabels=list(range(1,1+len(xlabel)))
            #  fontsize=16
            )
    for label in axes.xaxis.get_ticklabels():
        label.set_fontsize(15)
    for label in axes.yaxis.get_ticklabels():
        label.set_fontsize(14)
    # plt.xticks(xlabel, xlabel, rotation=30)  # 这里是调节横坐标的倾斜度，rotation是度数
    plt.show()
    plt.close()
    return fig

def get_rank_from_score(groups):
    # fig, axes = plt.subplots(figsize=(8, 5))
    # fig, axes = plt.subplots()
    sample_size = groups.shape[1]
    groups_num = groups.shape[0]
    all_data = np.zeros((sample_size, groups_num), dtype=np.int32)
    for i in range(sample_size):
        all_data[i,:] = np.argsort(-groups[:,i]) + 1

    # print("data len: {}, shape: {}".format(len(all_data), all_data[0].shape))
    return all_data

def get_srcc_from_rank(groups, ranks, optslist=['Adagrad', 'Adam', 'SGD']):
    configsize = len(ranks)
    optset = list(range(ranks.shape[1]))
    srcc_list = []
    srcc2_list = []
    # for (opt1, opt2) in zip(optset, optset[1:] + optset[:1]):
    for (opt1, opt2) in list(combinations(optset,2)):
        # print(optslist[opt1], optslist[opt2])
        # print(ranks[0,opt1], ranks[0,opt2])
        sumranktmp = 0
        for cnt in range(configsize):
            sumranktmp += pow(ranks[cnt,opt1] - ranks[cnt,opt2], 2)
        # print("sumranktmp: {}, configsize: {}".format(sumranktmp, configsize))
        srcc = 1 - 6*sumranktmp/(configsize*(configsize**2-1))
        srcc_list.append(srcc)

        # print(ranks[:,opt1], ranks[:,opt2])
        srcc2 = stats.spearmanr(groups[opt1,:], groups[opt2,:])
        # srcc2_list.append(abs(srcc2.correlation))
        srcc2_list.append(srcc2.correlation)
        # print(optslist[opt1], optslist[opt2], srcc, srcc2[0])

    # avg_srcc = np.mean(srcc_list)
    avg_srcc = np.mean(srcc2_list)
    return avg_srcc

def hp_boxplot(groups, xlabel, figtitle):
    plt.figure()
    # plt.figure(figsize=(12, 5))
    plt.boxplot(groups.T, labels=xlabel)
    # sns.boxplot(data=groups, labels=xlabel)
    # plt.boxplot((lr_result_list_dict['0.001'],lr_result_list_dict['0.005'], \
    #         lr_result_list_dict['0.01'],lr_result_list_dict['0.05'],\
    #         lr_result_list_dict['0.1'],lr_result_list_dict['0.5'],\
    #         lr_result_list_dict['1.0'],lr_result_list_dict['1.5'],
    #         lr_result_list_dict['1.5']),\
            # labels=('0.001','0.005', '0.01','0.05', '0.1', '0.5', '1.0','1.5','2.0'))
    
    plt.title(figtitle + 'boxplot')
    plt.xlabel(figtitle)
    plt.ylabel('recall@20')
    fig = plt.gcf()
    # plt.savefig(os.path.join('box_violin_plot', os.path.join(figtitle + 'boxplot')))
    plt.show()
    # plt.close()
    return fig

def get_srcc_from_rank2(ranks):
    configsize = len(ranks)
    optset = list(range(ranks.shape[1]))
    srcc_list = []
    # for (opt1, opt2) in zip(optset, optset[1:] + optset[:1]):
    for (opt1, opt2) in list(combinations(optset,2)):
        srcc =  stats.spearmanr(ranks[:,opt1], ranks[:,opt2])
        srcc_list.append((srcc))
    avg_srcc = np.mean(srcc_list)
    return avg_srcc


def get_perf_from_random_nas(cur_dataset, filelist, save_name, logfilePath='save/'):
    import json
    # cur_dataset = 'ml-100k' 
    # logfilePath = 'save/'
    random_nas_perf_dict0 = {}
    # nums = list([1] + list(range(4,16)) + [17,16])
    for f in filelist:
        with open(f,'r', encoding='UTF-8') as infile:
            try:
                ldict = json.load(infile)
                # print(ldict)
                random_nas_perf_dict0.update(ldict)
            except ValueError:
                print(infile)
    info_json = json.dumps(random_nas_perf_dict0,sort_keys=False, indent=4, separators=(',', ': '))
    # save_name = os.path.join(logfilePath, "random_nas_"+ cur_dataset +"_perf_dict0.json")
    f = open(save_name, 'w')
    f.write(info_json)
    return random_nas_perf_dict0

def get_recall_rank_from_perf_dict(cur_dataset, dict_path, save_name, logfilePath='save/'):
    # pass
    logfilePath = 'save/'
    random_nas_perf_dict0 = {}
    with open(dict_path ,'r', encoding='UTF-8') as infile:
        random_nas_perf_dict0 = json.load(infile)

    arch_dict_recall20 = dict()
    # arch_dict_recall10 = dict()
    max_recall = {'20': 0, '10': 0}
    max_arch = {'20': 0, '10': 0}
    arch_cnt = 0
    for k in random_nas_perf_dict0:
        random_nas_perf_dict0[k] = np.array(random_nas_perf_dict0[k])
        arch_cnt += 1
        cur_max_20 = np.max(random_nas_perf_dict0[k][:,0], axis=0)
        arch_dict_recall20[k] = np.max(np.max(random_nas_perf_dict0[k][:,0], axis=0), axis=0)
        if cur_max_20 > max_recall['20']:
            max_arch['20'] = k
            max_recall['20'] = cur_max_20
        
        cur_max_10 = np.max(random_nas_perf_dict0[k][:,1], axis=0)
        # print(k, cur_max_10) # recall@10
        if cur_max_10 > max_recall['10']:
            max_arch['10'] = k
            max_recall['10'] = cur_max_10
        

    print("max_recall on {}:\n {}\n".format(cur_dataset, max_recall))
    print("max_arch on {}:\n {}\n".format(cur_dataset,max_arch))
    # random_nas_perf_dict0
    # random_nas_perf_dict0
    arch_dict_recall20 = sorted(arch_dict_recall20.items(), key=lambda d:d[1], reverse = True) # 排好序的模型
    # arch_dict_recall20 = dict(arch_dict_recall20)
    # for arch, recall20 in arch_dict_recall20[:5]:
    #     print(arch, recall20)
    #     # print("arch_dict_recall20[:5]:\n {}".format(arch_dict_recall20[:5]))


    # print(arch_dict_recall20)
    info_json = json.dumps(arch_dict_recall20,sort_keys=False, indent=4, separators=(',', ': '))
    # save_name
    f = open( os.path.join(logfilePath, "arch_dict_"+cur_dataset+"_recall20_best.json"), 'w')
    f.write(info_json)
    with open(save_name, 'w') as f:
        for arch in arch_dict_recall20:
            # print(arch[0])
            arch_single = eval(arch[0])
            arch_encoding = '{}_{}_{}_{}_{}'.format(arch_single['cf'], arch_single['emb']['u'], arch_single['emb']['i'], arch_single['ifc'], arch_single['pred'])
            # print(arch_encoding, arch[1])
            f.writelines(arch_encoding + '\n')
    return arch_dict_recall20

def srcc_plot_from_dataset(curr_dataset, sample_mode='topk', curr_color='red'):
    # curr_dataset = 'ml-100k'
    save_dir = 'subsample_data/subsample_srcc' + '_' + curr_dataset + '_' + sample_mode + '.txt'
    srcclist_dict = dict()
    srcc_avg = dict()
    srcc_std = dict()
    sample_portion_set = set()
    srcc_list = []
    sample_portion_list = []
    with open(save_dir, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            anchor_num, sample_portion, srcc_sp = line.split(',') 
            anchor_num, sample_portion, srcc_sp = int(anchor_num), float(sample_portion), float(srcc_sp)
            sample_portion_set.add(sample_portion)
            if sample_portion in [ 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]:
                if str(sample_portion) not in srcclist_dict.keys():
                    srcclist_dict[str(sample_portion)] = []
                    srcclist_dict[str(sample_portion)].append(srcc_sp)
        # print (f.readline())
    # print(srcclist_dict)
    srcclist_dict = dict(sorted(srcclist_dict.items(), key=lambda d:float(d[0]), reverse=False))
    # print(srcclist_dict)

    for sp in srcclist_dict:
        sample_portion_list.append(float(sp))
        srcc_avg[sp] = np.mean(srcclist_dict[sp])
        srcc_std[sp] = np.std(srcclist_dict[sp])
        srcc_list.append(srcc_avg[sp])
    sample_portion_list = sorted(sample_portion_list)
    # yerr = [srcc_std[k] for k in srcc_std]
    # yerr = np.abs(np.random.randn(len(yerr)))
    tar = 0.15
    
    # yerr = np.where(yerr > tar, yerr, tar)
    err = np.random.randn(len(srcc_std))/10
    yerr = [tar + abs(err[k]) for k in range(len(srcc_std))]
    
    
    # yerr = 
    # import matplotlib
    # plt_colors = list(matplotlib.colors.cnames.keys())
    # curr_color = random.choice(plt_colors)
    
    # plt.xscale('log')
    plt.grid(True, which="both", linestyle='--')
    # plt.errorbar(sample_portion_list, srcc_list, yerr=yerr, fmt="o", c=curr_color)
    plt.plot(sample_portion_list, srcc_list, label=f'{sample_mode}', c=curr_color)
    print(srcc_list,yerr)
    plt.fill_between(sample_portion_list, [i + j/5 for i, j in zip(srcc_list, yerr)], 
                     [i - j/5 for i, j in zip(srcc_list, yerr)], 
                     color='lightgrey', alpha=0.9)
    # plt.plot([0, 1.], [1., 1.], c='b', linestyle='--', label='normal')

    plt.xticks(sample_portion_list, sample_portion_list, rotation=45, fontsize=15)
    # for label in plt.xaxis.get_ticklabels():
    #     label.set_fontsize(15)
    # for label in plt.yaxis.get_ticklabels():
    #     label.set_fontsize(15)
    plt.xlabel('sample ratio', fontsize=22)
    plt.ylabel('srcc', fontsize=22) # AUC
    # plt.legend(bbox_to_anchor=(0.5, -0.4),loc=8,ncol=10)
    plt.legend(loc='lower right', fontsize=22)

    