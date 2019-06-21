import numpy as np
import scipy
import scipy.sparse
import torch
import torch.utils.data
import pandas as pd

from mlperf_compliance import mlperf_log


class CFTrainDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, train_fname, data_summary_fname, nb_neg):
        data_summary = pd.read_csv(data_summary_fname, sep=',', header=0)
        self.nb_users = data_summary.ix[0]['users']
        self.nb_items = data_summary.ix[0]['items']
        self._load_train_matrix(train_fname)
        self.nb_neg = nb_neg

        mlperf_log.ncf_print(key=mlperf_log.INPUT_STEP_TRAIN_NEG_GEN, value=nb_neg)
        mlperf_log.ncf_print(key=mlperf_log.INPUT_HP_SAMPLE_TRAIN_REPLACEMENT)

    def _load_train_matrix(self, train_fname):
        def process_line(line):
            line = line.strip().split('\t')
            tmp = []
            tmp.extend(np.array(line[0:-1]).astype(int))
            tmp.extend([float(line[-1]) > 0])

            return tmp

        with open(train_fname, 'r') as file:
            data = list(map(process_line, file))
        # self.nb_users = max(data, key=lambda x: x[0])[0] + 1
        # self.nb_items = max(data, key=lambda x: x[1])[1] + 1

        length = len(data)

        self.data = list(filter(lambda x: x[-1], data))
        self.mat = scipy.sparse.dok_matrix(
                (self.nb_users, self.nb_items), dtype=np.float32)
        for i in range(length):
            user = self.data[i][0]
            item = self.data[i][1]
            self.mat[user, item] = 1.

    def __len__(self):
        return (self.nb_neg + 1) * len(self.data)

    def __getitem__(self, idx):
        if idx % (self.nb_neg + 1) == 0:
            idx = idx // (self.nb_neg + 1)
            return self.data[idx][0], self.data[idx][1], torch.LongTensor(self.data[idx][2:-1]), np.ones(1, dtype=np.float32)  # noqa: E501
        else:
            idx = idx // (self.nb_neg + 1)
            u = self.data[idx][0]
            j = torch.LongTensor(1).random_(0, int(self.nb_items)).item()
            while (u, j) in self.mat:
                j = torch.LongTensor(1).random_(0, int(self.nb_items)).item()
            return u, j, torch.LongTensor(self.data[idx][2:-1]), np.zeros(1, dtype=np.float32)


def load_test_ratings(fname):
    def process_line(line):
        tmp = map(int, line.strip().split('\t')[:-1])
        return list(tmp)
    ratings = map(process_line, open(fname, 'r'))
    return list(ratings)


def load_test_negs(fname):
    def process_line(line):
        tmp = map(int, line.strip().split('\t')[2:])
        return list(tmp)
    negs = map(process_line, open(fname, 'r'))
    return list(negs)
