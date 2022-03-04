import numpy as np
from torch.utils.data import Dataset
import copy
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler


class MixDatasets(Dataset):
    def __init__(self, dataset1, dataset2):
        super().__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, item):
        # assert item < len(self.dataset1) + len(self.dataset2)
        if item < len(self.dataset1):
            return (*self.dataset1[item], 0)
        else:
            return (*self.dataset2[item - len(self.dataset1)], 1)



class MutiSourceRandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, p=(0.5, 0.5), epoch_lenth=0):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.p = p      # 两组source采样概率
        self.index_dic = defaultdict(list)
        self.index_dic1 = defaultdict(list)
        for index, (pid) in enumerate(self.data_source.dataset1.labels):
            self.index_dic[pid].append(index)
        for index, (pid) in enumerate(self.data_source.dataset2.labels):
            self.index_dic1[pid].append(index + len(self.data_source.dataset1))
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        batch_idxs_dict1 = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            idxs1 = copy.deepcopy(self.index_dic1[pid])
            if len(idxs) < round(self.num_instances * self.p[0]):
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            if len(idxs1) < round(self.num_instances * self.p[1]):
                idxs1 = np.random.choice(idxs1, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            random.shuffle(idxs1)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == round(self.num_instances * self.p[0]):
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
            batch_idxs = []
            for idx in idxs1:
                batch_idxs.append(idx)
                if len(batch_idxs) == round(self.num_instances * self.p[1]):
                    batch_idxs_dict1[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                batch_idxs = batch_idxs_dict1[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0 or len(batch_idxs_dict1[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, ignored_ids=()):
        # super().__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        self.ignored_ids = ignored_ids
        for index, (_, pid) in enumerate(self.data_source):
            if pid in self.ignored_ids:
                continue
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class IterLoader():
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None) or (self.length == 0):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)

    def get_batches(self):
        for i in range(self.length):
            yield self.next()

    def __iter__(self):
        return self.get_batches()
