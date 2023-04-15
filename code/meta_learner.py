
import torch.nn as nn
import torch
import os
import numpy as np
from meta_models import FM, MLP
from sklearn.utils import shuffle
import logging
import sys
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn import metrics
from sklearn.model_selection import cross_val_score, ShuffleSplit, cross_validate
import random
from visdom import Visdom
from typing import List, Union
import time
import bottleneck


class VisShow(object):
    def __init__(self, server: str, port: int, envdir: str, subenv: str) -> None:
        self.vis = Visdom(server, port=port, env=f'{envdir}_{subenv}')

    def update(self, target: str, X: List[Union[int, float]], Y: List[Union[int, float]]) -> None:
        attrname = f'__{target}'
        if hasattr(self, attrname):
            self.vis.line(Y, X, win=getattr(self, attrname), update='append')
        else:
            setattr(self, attrname, self.vis.line(
                Y, X, opts={'title': target}))

class MetaLearner(object):
    def __init__(self, vis_name='', res_file=''):
        self.arch_performance = dict()
        self.best_arch = ''
        self.best_performance = 0.0
        if 'avg' not in res_file:
            self.dataset = self.get_dataset(res_file)
        else:
            self.dataset = self.get_dataset_new(res_file)
        self.arch_num = len(self.dataset)
        self.learning_rate = 0.01
        self.log_file_name = 'test'
        self.init_logger()
        self.vis = VisShow(server='127.0.0.1',
                          port=9998,
                          envdir='visdom',
                          subenv=vis_name)

    def init_logger(self):
        log_format = '%(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, filemode='w', format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(self.log_file_name)
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

    def get_arch_performance(self, x):
        encoding_dict = {
            'ui':0,
            'ur':1,
            'ri':2,
            'rr':3,
            'mat':0,
            'mlp':1,
            'max':0,
            'min':1,
            'mul':2,
            'plus':3,
            'concat':4,
            'h':0,
            'i':2,
            '0.01':0,
            '0.02':1,
            '0.05':2,
            '0.1':3,
            '2':0,
            '4':1,
            '8':2,
            '16':3
        }
        arch = x.split()[2].split(':')[0]
        encoding = [encoding_dict[k] for k in arch.split('_')]
        performance = x.split()[2].split(':')[1].strip()
        return encoding, performance

    def get_arch_performance_new(self, x):
        encoding_dict = {
            'ui':0,
            'ur':1,
            'ri':2,
            'rr':3,
            'mat':0,
            'mlp':1,
            'max':0,
            'min':1,
            'mul':2,
            'plus':3,
            'concat':4,
            'h':0,
            'i':2,
            '0.01':0,
            '0.02':1,
            '0.05':2,
            '0.1':3,
            '2':0,
            '4':1,
            '8':2,
            '16':3
        }
        arch = x.split()[2].split(':')[0]
        encoding = [encoding_dict[k] for k in arch.split('_')]
        performance = x.split()[2].split(':')[1].strip()
        return encoding, performance
    
    def get_dataset(self, log_file):
        if not os.path.exists(log_file):
            raise NotADirectoryError
        with open(log_file, 'r') as fr:
            lines = fr.readlines()
            arch_performance = [self.get_arch_performance(k) for k in lines[2:]]
            fullarch_performance = []
            for k in range(len(arch_performance)):
                arch = arch_performance[k][0]
                performance = float(arch_performance[k][1])
                fullarch_performance.append((arch, performance))
        return fullarch_performance


    def get_dataset_new(self, log_file):
        if not os.path.exists(log_file):
            raise NotADirectoryError
        with open(log_file, 'r') as fr:
            lines = fr.readlines()
            arch_performance = [self.get_arch_performance_new(k) for k in lines[2:]]
            fullarch_performance = []
            for k in range(len(arch_performance)):
                idx = k % 16
                lr = int(idx / 4)
                emb = idx % 4
                arch = arch_performance[k][0]
                performance = float(arch_performance[k][1])
                arch.append(lr)
                arch.append(emb)
                fullarch_performance.append((arch, performance))
        return fullarch_performance


    def convert_encoding_to_dict(self, arch_encoding):
        raise NotImplementedError

    def convert_dict_to_encoding(self, arch_dict):
        raise NotImplementedError

    def parse_raw_log(self, log_file):
        raise NotImplementedError

    def train_model(self, model: nn.Module):
        raise NotImplementedError

    def random_simulator(self):
        logging.info('Start')
        best_arch, best_performance = '', 100
        best_performances = []
        simu_dataset = shuffle(deepcopy(self.dataset))
        for meta_epoch in range(self.arch_num):
            arch, performance = simu_dataset[meta_epoch]
            if performance < best_performance:
                best_arch = arch
                best_performance = performance
            best_performances.append(best_performance)
            if best_performance <= 0.3657:
                return meta_epoch
                break
            logging.info('{},{}'.format(meta_epoch, best_performance))

    def get_top(self, meta_batch, meta_results, meta_label, k):
        meta_label = np.squeeze(meta_label)
        meta_results = np.squeeze(meta_results)
        idx = np.argsort(meta_label.detach().numpy(), kind='heapsort')[0:k]
        idx_remained = np.argsort(meta_results, kind='heapsort')[k:]
        return [k[idx] for k in meta_batch], meta_results[idx], idx_remained

    def online_simulator_oneshot(self, model: nn.Module):
        logging.info('Start Online Simulator')
        simu_dataset = shuffle(deepcopy(self.dataset))
        best_arch, best_performance = '', 100
        best_performances = []
        loss_fn = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        batch_size = 10
        for meta_epoch in range(self.arch_num//5):
            meta_batch = simu_dataset[0:batch_size]
            input_meta_batch_tensor = []
            for k in range(7):
                input_meta_batch_tensor.append(torch.LongTensor([i[0][k] for i in meta_batch]))
            label_meta_batch_tensor = torch.FloatTensor([i[1] for i in meta_batch])
            output_meta_batch_tensor = model(input_meta_batch_tensor)
            top_input, top_label, remained_idx = self.get_top(input_meta_batch_tensor, label_meta_batch_tensor, 5)
            model.train()
            optimizer.zero_grad()
            model.zero_grad()
            top_output = model(top_input)
            loss = loss_fn(torch.tensor(top_label), top_output.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            arch, performance, _ = self.get_top(top_input, top_label, 1)
            if performance < best_performance:
                best_arch = arch
                best_performance = performance
            if best_performance <= 0.3657:
                return meta_epoch*5
            best_performances.append(best_performance)
            simu_dataset = simu_dataset[batch_size:] + [meta_batch[k] for k in remained_idx]
            simu_dataset = shuffle(simu_dataset)

    def online_simulator(self, model: nn.Module):
        simu_dataset = shuffle(deepcopy(self.dataset))
        best_arch, best_performance = '', 100
        best_performances = []
        loss_fn = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        batch_size = 10
        all_input = []
        all_output = torch.FloatTensor([])
        all_label = torch.LongTensor([])
        for meta_epoch in range(self.arch_num//5):
            meta_batch = simu_dataset[0:batch_size]
            input_meta_batch_tensor = []
            for k in range(7):
                input_meta_batch_tensor.append(torch.LongTensor([i[0][k] for i in meta_batch]))
            label_meta_batch_tensor = torch.FloatTensor([i[1] for i in meta_batch])
            output_meta_batch_tensor = model(input_meta_batch_tensor)
            top_input, top_label, remained_idx = self.get_top(input_meta_batch_tensor, label_meta_batch_tensor, output_meta_batch_tensor, 5)
            model.train()
            optimizer.zero_grad()
            model.zero_grad()
            if all_label.shape[0] == 0:
                all_input = top_input
                all_label = top_label
            else:
                all_input = [torch.cat([all_input[k], top_input[k]]) for k in range(7)]
                all_label = torch.cat([all_label, top_label])
            init_loss = 1000
            while True:
                top_output = model(all_input)
                loss = loss_fn(torch.tensor(all_label), top_output.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if loss.detach().numpy() > init_loss:
                    break
                init_loss = loss.detach().numpy()
            performance = min(top_label.detach().numpy())
            if performance < best_performance:
                best_performance = performance
            if best_performance <= 0.3657:
                return meta_epoch*5
            best_performances.append(best_performance)
            simu_dataset = simu_dataset[batch_size:] + [meta_batch[k] for k in remained_idx]
            simu_dataset = shuffle(simu_dataset)
    

    def online_simulator_old(self, model: nn.Module):
        simu_dataset = shuffle(deepcopy(self.dataset))
        best_arch, best_performance = '', 100
        best_performances = []
        loss_fn = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        batch_size = 10
        all_input = []
        all_output = torch.FloatTensor([])
        all_label = torch.LongTensor([])
        for meta_epoch in range(self.arch_num//5):
            meta_batch = simu_dataset
            input_meta_batch_tensor = []
            for k in range(7):
                input_meta_batch_tensor.append(torch.LongTensor([i[0][k] for i in meta_batch]))
            label_meta_batch_tensor = torch.FloatTensor([i[1] for i in meta_batch])
            output_meta_batch_tensor = model(input_meta_batch_tensor)
            top_input, top_label, remained_idx = self.get_top(input_meta_batch_tensor, label_meta_batch_tensor, 5)
            model.train()
            optimizer.zero_grad()
            model.zero_grad()
            if all_label.shape[0] == 0:
                all_input = top_input
                all_label = top_label
            else:
                all_input = [torch.cat([all_input[k], top_input[k]]) for k in range(7)]
                all_label = torch.cat([all_label, top_label])

            init_loss = 1000
            while True:
                top_output = model(top_input)
                loss = loss_fn(torch.tensor(top_label), top_output.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if loss.detach().numpy() > init_loss:
                    break
                init_loss = loss.detach().numpy()
            top_output
            if performance < best_performance:
                best_performance = performance
            if best_performance <= 0.3657:
                return meta_epoch*5
            best_performances.append(best_performance)
            simu_dataset = simu_dataset[batch_size:] + [meta_batch[k] for k in remained_idx]
            simu_dataset = shuffle(simu_dataset)

    def offline_rf_simulator(self, model):
        simu_dataset = shuffle(deepcopy(self.dataset))
        X = np.array([k[0] for k in simu_dataset])
        Y = np.array([k[1] for k in simu_dataset])
        train_ratio = 0.01
        X_train = X[0:int(X.shape[0]*train_ratio)]
        Y_train = Y[0:int(X.shape[0]*train_ratio)]
        X_test = X[int(X.shape[0]*train_ratio):]
        Y_test = Y[int(X.shape[0]*train_ratio):]
        model.fit(X_train, Y_train)
        Y_predict = model.predict(X_test)
        print('Training data ratio: {}'.format(train_ratio))
        print('RMSE: {}'.format(metrics.mean_squared_error(Y_predict, Y_test)))
        print('Precision of pair-prediction: {}'.format(self.get_ranking_performance(Y_predict, Y_test)))
        print('Predicted rank of the best arch: {}'.format(self.get_toprank_performance(Y_predict, Y_test)))

    def online_rf_simulator(self, model):
        simu_dataset = shuffle(deepcopy(self.dataset))
        X = np.array([k[0] for k in simu_dataset])
        Y = np.array([k[1] for k in simu_dataset])
        K_selection = 30
        trained_idx = np.array([], dtype=int)
        remained_idx = np.array([], dtype=int)
        best_arch_performance = 2.0
        for meta_epoch in range(int(X.shape[0]/K_selection)):
            print('Meta epoch:{}'.format(meta_epoch))
            if meta_epoch == 0:
                X_train, Y_train = X[0:(meta_epoch+1)*K_selection], Y[0:(meta_epoch+1)*K_selection]
                X_test, Y_test = X[(meta_epoch+1)*K_selection:], Y[(meta_epoch+1)*K_selection:]
            else:
                train_idx = np.append(remained_idx[0:K_selection], trained_idx)
                X_train, Y_train = X[train_idx], Y[train_idx]
                test_idx = remained_idx[K_selection:]
                X_test, Y_test = X[test_idx], Y[test_idx]
            model.fit(X_train, Y_train)
            Y_predict = model.predict(X_test)
            if meta_epoch == 0:
                min_K_idx = np.argpartition(Y_predict, K_selection)[0:K_selection]
            else:
                if test_idx.shape[0] <= K_selection:
                    min_K_idx = test_idx
                else:
                    min_K_idx = test_idx[np.argpartition(Y_predict, K_selection)[0:K_selection]]
            trained_idx = np.append(min_K_idx, trained_idx)
            remained_idx = np.array(list(set(list(range(X.shape[0]))).difference(set(list(trained_idx)))))

            self.vis.update('RMSE', [meta_epoch*K_selection], [metrics.mean_squared_error(Y_predict, Y_test)])
            self.vis.update('Pair-prediction accuracy', [meta_epoch*K_selection], [self.get_ranking_performance(Y_predict, Y_test)])
            print('Training data ratio: {}'.format(X_train.shape[0]/X.shape[0]))
            print('RMSE: {}'.format(metrics.mean_squared_error(Y_predict, Y_test)))
            print('Precision of pair-prediction: {}'.format(self.get_ranking_performance(Y_predict, Y_test)))
            best_rank, best_rmse = self.get_toprank_performance(Y_predict, Y_test)
            print('Predicted rank of the best arch: {}/{}'.format(best_rank, Y_test.shape[0]))
            self.vis.update('Rank of the best arch', [meta_epoch*K_selection], [best_rank])
            self.vis.update('Best performance', [meta_epoch*K_selection], [best_rmse])

            if best_rank <= K_selection:
                print('Best arch found!')
                break
        return meta_epoch, K_selection


    def online_mlp_simulator(self, model, reinitialize=False, if_convergence=True, if_pairwise=False):
        simu_dataset = shuffle(deepcopy(self.dataset))
        X = np.array([k[0] for k in simu_dataset])
        Y = np.array([k[1] for k in simu_dataset])
        K_selection = 30
        trained_idx = np.array([], dtype=int)
        remained_idx = np.array([], dtype=int)
        loss_fn = torch.nn.L1Loss()
        loss_fn_pair = torch.nn.BCEWithLogitsLoss(size_average=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        best_arch_performance = 0.0
        for meta_epoch in range(int(X.shape[0]/K_selection)):
            print('Meta epoch:{}'.format(meta_epoch))
            if meta_epoch == 0:
                X_train, Y_train = X[0:(meta_epoch+1)*K_selection], Y[0:(meta_epoch+1)*K_selection]
                X_test, Y_test = X[(meta_epoch+1)*K_selection:], Y[(meta_epoch+1)*K_selection:]
            else:
                train_idx = np.append(remained_idx[0:K_selection], trained_idx)
                X_train, Y_train = X[train_idx], Y[train_idx]
                test_idx = remained_idx[K_selection:]
                X_test, Y_test = X[test_idx], Y[test_idx]
            
            X_train_tensor = [torch.LongTensor(X_train[:, k]) for k in range(7)]

            tried_arches = Y_train
            partition_len = int(len(tried_arches)/2)
            ranked_tried_arches_neg = shuffle(bottleneck.argpartition(tried_arches, partition_len)[:partition_len])
            ranked_tried_arches_pos = shuffle(bottleneck.argpartition(tried_arches, partition_len)[-partition_len:])
            X_train_pos = [torch.LongTensor(X_train[ranked_tried_arches_pos, k]) for k in range(7)]
            X_train_neg = [torch.LongTensor(X_train[ranked_tried_arches_neg, k]) for k in range(7)]

                
            former_loss = 10000
            while True:
                if not if_pairwise:
                    train_output = model(X_train_tensor)
                    loss = loss_fn(torch.FloatTensor(Y_train).unsqueeze(1), train_output)
                else:
                    train_output = model.forward_pair(X_train_pos, X_train_neg)
                    labels = torch.ones(train_output.size()[0], device=train_output.device)
                    labels = torch.reshape(labels, [-1, 1])
                    loss = loss_fn_pair(train_output, labels) / (train_output.size()[0])

                model.train()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if loss.detach().numpy() > former_loss:
                    break
                if not if_convergence:
                    break
                former_loss = loss.detach().numpy()

            X_test_tensor = [torch.LongTensor(X_test[:,k]) for k in range(7)]
            test_output = model(X_test_tensor)
            Y_predict = test_output.detach().numpy()[:, 0]

            if meta_epoch == 0:
                min_K_idx = np.argpartition(Y_predict, K_selection)[0:K_selection]
            else:
                if test_idx.shape[0] <= K_selection:
                    min_K_idx = test_idx
                else:
                    min_K_idx = test_idx[np.argpartition(Y_predict, K_selection)[0:K_selection]]
            trained_idx = np.append(min_K_idx, trained_idx)
            remained_idx = np.array(list(set(list(range(X.shape[0]))).difference(set(list(trained_idx)))))

            self.vis.update('RMSE', [meta_epoch*K_selection], [metrics.mean_squared_error(Y_predict, Y_test)])
            self.vis.update('Pair-prediction accuracy', [meta_epoch*K_selection], [self.get_ranking_performance(Y_predict, Y_test)])
            print('Training data ratio: {}'.format(X_train.shape[0]/X.shape[0]))
            print('RMSE: {}'.format(metrics.mean_squared_error(Y_predict, Y_test)))
            print('Precision of pair-prediction: {}'.format(self.get_ranking_performance(Y_predict, Y_test)))
            best_rank, best_rmse = self.get_toprank_performance(Y_predict, Y_test)
            print('Predicted rank of the best arch: {}/{}'.format(best_rank, Y_test.shape[0]))
            self.vis.update('Rank of the best arch', [meta_epoch*K_selection], [best_rank])
            if best_rmse < best_arch_performance:
                best_arch_performance = best_rmse
            self.vis.update('Best searched performance', [meta_epoch*K_selection], [best_arch_performance])

            if best_rank <= K_selection:
                print('Best arch found!')
                break
        return meta_epoch, K_selection


    def get_ranking_performance(self, prediction, label, sample_num=100000):
        precision = []
        for k in range(sample_num):
            i = random.randint(0, prediction.shape[0]-1)
            j = random.randint(0, prediction.shape[0]-1)
            while i == j:
                i = random.randint(0, prediction.shape[0]-1)
                j = random.randint(0, prediction.shape[0]-1)
            true_rank = label[i] > label[j]
            predicted_rank = prediction[i] > prediction[j]
            if true_rank == predicted_rank:
                precision.append(1)
            else:
                precision.append(0)
        return sum(precision) / len(precision)
    

    def get_toprank_performance(self, prediction, label, K_selection=10):
        best_one_index = list(label).index(min(label))
        best_one_predict = prediction[best_one_index]
        best_rank = 0
        for k in prediction:
            if k <= best_one_predict:
                best_rank += 1
        best_predicted_index = list(prediction).index(min(prediction))
        best_predicted_real_label = label[best_predicted_index]
        print('The predicted best one\'s real performance: {}'.format(best_predicted_real_label))
        smallest_prediction_idx = np.argpartition(prediction, K_selection)[0:K_selection]
        smallest_prediction_real_label = label[smallest_prediction_idx]
        print('The real performance among the top-K prediction: {}'.format(min(smallest_prediction_real_label)))
        return best_rank, best_predicted_real_label 

    


if __name__ == '__main__':

    fw = open('save/log_meta/RF10_ml1m.txt', 'w')
    for loop in range(10):
        rf_model = RF(oob_score=True, random_state=10)
        mlp_model = MLP(4, 0.001)
        current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        meta_learer = MetaLearner(vis_name='ml1m-random10-' + current_time, res_file='save/ml100kbpr.txt')
        meta_epoch_num, meta_batch_size = meta_learer.online_mlp_simulator(mlp_model, reinitialize=False, if_pairwise=True)
        fw.write('{}, {}\n'.format(meta_epoch_num, meta_batch_size))
