import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from models import CF_MODEL, CF_EMB, CF_IFC, CF_PRED, CF_EMB_LIST
from itertools import chain


def sample_arch_cf_signal():
    '''
    对于所有的CF_MODEL
    '''
    # CF_EMB = ['mat', 'mlp']
    arch_cf_list = dict() # 一共四种模式，r表示此处对应的Embedding是用random
    arch_cf_list['ui'], arch_cf_list['ur'], arch_cf_list['ri'], arch_cf_list['rr'] = dict(
    ), dict(), dict(), dict()
    for cf_signal in arch_cf_list: 
        arch_cf = dict()
        arch_cf['cf'] = cf_signal # 一个cf_signal相当于一个Input Encoding
        arch_cf['emb'] = dict()
        if arch_cf['cf'] == 'ui':
            arch_cf['emb']['u'] = 'mat'
            arch_cf['emb']['i'] = 'mat'
        elif arch_cf['cf'] == 'ur':
            arch_cf['emb']['u'] = 'mat'
            arch_cf['emb']['i'] = CF_EMB[np.random.randint(len(CF_EMB))]
        elif arch_cf['cf'] == 'ri':
            arch_cf['emb']['u'] = CF_EMB[np.random.randint(len(CF_EMB))]
            arch_cf['emb']['i'] = 'mat'
        else:
            arch_cf['emb']['u'] = CF_EMB[np.random.randint(len(CF_EMB))]
            arch_cf['emb']['i'] = CF_EMB[np.random.randint(len(CF_EMB))]
        arch_cf['ifc'] = CF_IFC[np.random.randint(len(CF_IFC))]
        arch_cf['pred'] = CF_PRED[np.random.randint(len(CF_PRED))]
        arch_cf_list[cf_signal] = arch_cf
    weight = np.random.rand(4)
    arch_cf_list['weight'] = list(weight / weight.sum())
    return arch_cf_list


def sample_arch_cf(cf_model_num=1): 
    '''
    针对emb部分进行sample，根据arch_cf['cf']进行分类
    '''
    arch_cf = dict()
    if cf_model_num == 1:
        arch_cf['cf'] = CF_MODEL[np.random.randint(len(CF_MODEL))]
    arch_cf['emb'] = dict()

    if arch_cf['cf'] == 'ui':
        arch_cf['emb']['u'] = 'mat'
        arch_cf['emb']['i'] = 'mat'
    elif arch_cf['cf'] == 'ur':
        arch_cf['emb']['u'] = 'mat'
        arch_cf['emb']['i'] = CF_EMB[np.random.randint(len(CF_EMB))]#CF_EMB = ['mat', 'mlp']
    elif arch_cf['cf'] == 'ri':
        arch_cf['emb']['u'] = CF_EMB[np.random.randint(len(CF_EMB))]
        arch_cf['emb']['i'] = 'mat'
    else:
        arch_cf['emb']['u'] = CF_EMB[np.random.randint(len(CF_EMB))]
        arch_cf['emb']['i'] = CF_EMB[np.random.randint(len(CF_EMB))]
    arch_cf['ifc'] = CF_IFC[np.random.randint(len(CF_IFC))]
    arch_cf['pred'] = CF_PRED[np.random.randint(len(CF_PRED))]
    return arch_cf

def sample_arch_cf_ifc(ifc_mode): 
    '''
    针对emb部分进行sample，根据arch_cf['cf']进行分类
    '''
    
    CF_IFC = ['max', 'min', 'plus', 'mul', 'concat'] # Interacton
    if ifc_mode not in CF_IFC:
        print("错误")
        return []
    arch_cf_list = []
    # arch_cf = dict()
    arch_cf = {'cf':'', 
                'emb':{'u':'', 'i':''},
                'ifc':'',
                'pred':''}
    arch_cf['ifc'] = ifc_mode
    CF_MODEL = ['ui', 'ur', 'ri', 'rr'] # Input Encoding
    CF_EMB = ['mat', 'mlp'] # Embedding Function
    CF_PRED = ['i', 'h', 'mlp'] # Prediction Function
    
    for pred in CF_PRED:
        # arch_cf['ifc'] = CF_IFC[np.random.randint(len(CF_IFC))]
        # arch_cf['pred'] = CF_PRED[np.random.randint(len(CF_PRED))]
        arch_cf['pred'] = pred
        for cf in CF_MODEL:
            arch_cf['cf'] = cf
            # arch_cf['emb'] = dict()

            if arch_cf['cf'] == 'ui':
                arch_cf['emb']['u'] = 'mat'
                arch_cf['emb']['i'] = 'mat'
                arch_cf_list.append(arch_cf)
            elif arch_cf['cf'] == 'ur':
                arch_cf['emb']['u'] = 'mat'
                for iemb in CF_EMB:
                    arch_cf['emb']['i'] = iemb
                    arch_cf_list.append(arch_cf)
            elif arch_cf['cf'] == 'ri':
                for uemb in CF_EMB:
                    arch_cf['emb']['u'] = uemb
                    arch_cf_list.append(arch_cf)
                arch_cf['emb']['i'] = 'mat'
            else:
                for uemb in CF_EMB:
                    for iemb in CF_EMB:
                        arch_cf['emb']['u'] = uemb
                        arch_cf['emb']['i'] = iemb
                        arch_cf_list.append(arch_cf)
                
    return arch_cf_list


def sample_arch_cf_pred(pred_mode): 
    '''
    根据arch_cf['pred']进行分类
    '''
    # CF_MODEL = ['ui', 'ur', 'ri', 'rr'] # Input Encoding
    # CF_EMB = ['mat', 'mlp'] # Embedding Function
    # CF_PRED = ['i', 'h', 'mlp'] # Prediction Function
    # CF_PRED = ['i', 'h', 'mlp'] # Interacton
    if pred_mode not in CF_PRED:
        print("输入错误")
        return []
    arch_cf_list = []
    arch_cf = {'cf':'', 
                'emb':{'u':'', 'i':''},
                'ifc':'',
                'pred':''}
    arch_cf['pred'] = pred_mode
    
    for ifc in CF_IFC:
        arch_cf['ifc'] = ifc
        # arch_cf['pred'] = CF_PRED[np.random.randint(len(CF_PRED))]
        for cf in CF_MODEL:
            arch_cf['cf'] = cf
            if arch_cf['cf'] == 'ui':
                arch_cf['emb']['u'] = 'mat'
                arch_cf['emb']['i'] = 'mat'
                arch_cf_list.append(arch_cf)
            elif arch_cf['cf'] == 'ur':
                arch_cf['emb']['u'] = 'mat'
                for iemb in CF_EMB:
                    arch_cf['emb']['i'] = iemb
                    arch_cf_list.append(arch_cf)
            elif arch_cf['cf'] == 'ri':
                for uemb in CF_EMB:
                    arch_cf['emb']['u'] = uemb
                    arch_cf_list.append(arch_cf)
                arch_cf['emb']['i'] = 'mat'
            else:
                for uemb in CF_EMB:
                    for iemb in CF_EMB:
                        arch_cf['emb']['u'] = uemb
                        arch_cf['emb']['i'] = iemb
                        arch_cf_list.append(arch_cf)
                
    return arch_cf_list

def sample_arch_cf_emb(cf_pred_mode): 
    '''
    根据arch_cf['pred']进行分类
    '''
    # CF_EMB_LIST = ['ui_mat_mat', 
    #                 'ur_mat_mat', 'ur_mat_mlp',
    #                 'ri_mat_mat', 'ri_mlp_mat',
    #                 'rr_mat_mat', 'rr_mat_mlp',
    #                 'rr_mlp_mlp', 'rr_mlp_mat',] 
    if cf_pred_mode not in CF_EMB_LIST:
        print("输入错误")
        return []
    arch_cf_list = []
    arch_cf = {'cf':'', 
                'emb':{'u':'', 'i':''},
                'ifc':'',
                'pred':''}
    arch_cf['cf'], arch_cf['emb']['u'], arch_cf['emb']['i'] = cf_pred_mode.split('_')
    
    for ifc in CF_IFC:
        arch_cf['ifc'] = ifc
        # arch_cf['pred'] = CF_PRED[np.random.randint(len(CF_PRED))]
        for pred in CF_PRED:
            arch_cf['pred'] = pred
            arch_cf_list.append(arch_cf)
                
    return arch_cf_list


def sample_arch_cf_test(num=0):
    arch_lib = [
        ['ur', 'mat', 'mlp', 'max', 'h'],
        ['ur', 'mat', 'mlp', 'max', 'h'],
        ['ur', 'mat', 'mlp', 'plus', 'mlp'],
        ['ur', 'mat', 'mlp', 'plus', 'mlp'],

        ['ur', 'mat', 'mlp', 'concat', 'h'],
        ['ur', 'mat', 'mlp', 'concat', 'i'],
        ['ur', 'mat', 'mlp', 'concat', 'i'],
        ['rr', 'mlp', 'mat', 'plus', 'mlp']
    ]
    arch_cf = dict()
    arch_cf['cf'] = arch_lib[num][0]
    arch_cf['emb'] = dict()
    arch_cf['emb']['u'] = arch_lib[num][1]
    arch_cf['emb']['i'] = arch_lib[num][2]
    arch_cf['ifc'] = arch_lib[num][3]
    arch_cf['pred'] = arch_lib[num][4]
    
    return arch_cf

class Controller_CF_Signal(nn.Module):
    def __init__(self, choice, triple=False):
        super(Controller_CF_Signal, self).__init__()
        self.triple = triple
        self.space = max(len(CF_MODEL), len(CF_EMB), len(CF_IFC), len(CF_PRED))
        self.num_action = 17
        self.choice = choice
        if choice == 'RNN':
            self.controller = nn.RNNCell(
                input_size=self.space, hidden_size=self.space)
        elif choice == 'PURE':
            self._arch_parameters = []
            for _ in range(self.num_action):
                alpha = torch.ones(
                    [1, self.space], dtype=torch.float, device='cuda') / self.space
                # alpha = alpha + torch.randn(self.space, device='cuda') * 1e-2
                self._arch_parameters.append(
                    Variable(alpha, requires_grad=True))

    def arch_parameters(self):
        return self._arch_parameters

    def forward(self):
        if self.choice == 'RNN':
            input0 = torch.ones([1, self.space]) / self.space / 10.0
            input0 = input0.cuda()
            h = torch.zeros([1, self.space]).cuda()
            inferences = []
            for i in range(self.num_action):
                if i == 0:
                    h = self.controller(input0, h)
                else:
                    h = self.controller(h, h)
                inferences.append(h)
            return inferences
        elif self.choice == 'PURE':
            return self._arch_parameters

    def compute_loss(self, rewards):
        inferences = self()
        inferences = torch.cat(
            inferences[0:-1], dim=0).repeat(len(self.archs), 1)

        self.archs = [k[0:-4] for k in self.archs]
        self.archs = torch.tensor(list(chain(*self.archs))).cuda()
        rewards = torch.reshape(torch.tensor(rewards), [-1, 1])
        rewards = torch.reshape(rewards.repeat(
            1, self.num_action-1), [-1, 1]).cuda()
        return torch.mean(rewards * F.cross_entropy(inferences, self.archs))

    def print_prob(self):
        inferences = self()
        for infer in inferences:
            infer = F.softmax(infer, dim=-1).cpu().detach().numpy()

    def sample_arch_cf(self, batch_size):
        self.archs = []
        arch_list, archs = [], []
        inferences = self()
        batch_count = 0
        while batch_count < batch_size:
            tmp = []
            arch = dict()
            for action_count, infer in enumerate(inferences):
                infer = torch.squeeze(infer)
                cf_part_num = int(action_count / 4)
                if cf_part_num not in arch:
                    arch[cf_part_num] = dict()
                if action_count == 16:
                    p = F.softmax(infer[:4], dim=-1).cpu().detach().numpy()
                    arch['weight'] = np.around(p / p.sum(), 2)
                    tmp.extend(list(arch['weight']))
                elif action_count % 4 == 0:
                    p = F.softmax(infer[:len(CF_EMB)],
                                  dim=-1).cpu().detach().numpy()
                    choice = np.random.choice(len(CF_EMB), p=p)
                    if 'emb' not in arch[cf_part_num]:
                        arch[cf_part_num]['emb'] = dict()
                    arch[cf_part_num]['emb']['u'] = CF_EMB[choice]
                    tmp.append(choice)
                elif action_count % 4 == 1:
                    p = F.softmax(infer[:len(CF_EMB)],
                                  dim=-1).cpu().detach().numpy()
                    choice = np.random.choice(len(CF_EMB), p=p)
                    arch[cf_part_num]['emb']['i'] = CF_EMB[choice]
                    tmp.append(choice)
                elif action_count % 4 == 2:
                    p = F.softmax(infer[:len(CF_IFC)],
                                  dim=-1).cpu().detach().numpy()
                    choice = np.random.choice(len(CF_IFC), p=p)
                    arch[cf_part_num]['ifc'] = CF_IFC[choice]
                    tmp.append(choice)
                elif action_count % 4 == 3:
                    p = F.softmax(infer[:len(CF_PRED)],
                                  dim=-1).cpu().detach().numpy()
                    choice = np.random.choice(len(CF_PRED), p=p)
                    arch[cf_part_num]['pred'] = CF_PRED[choice]
                    tmp.append(choice)
            re_arch_dict = dict()
            cf_list = ['ui', 'ur', 'ri', 'rr']
            for i in range(4):
                re_arch = dict()
                re_arch['cf'] = cf_list[i]
                re_arch['emb'] = arch[i]['emb']
                # try:
                re_arch['ifc'] = arch[i]['ifc']
                re_arch['pred'] = arch[i]['pred']
                re_arch_dict[re_arch['cf']] = re_arch
                re_arch_dict['weight'] = arch['weight']
            if arch not in archs:
                archs.append(arch)
                self.archs.append(tmp)
                arch_list.append(re_arch_dict)
                batch_count += 1
        return arch_list


if __name__ == '__main__':
    pass
