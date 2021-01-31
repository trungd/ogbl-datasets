#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from collections import defaultdict

import numpy as np
import torch

from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, args, ds, mode):
        self.triples = ds.train_triples
        self.nentity = ds.nentity
        self.nrelation = ds.nrelation
        self.args = args
        self.ds = ds

        self.negative_sample_size = args.negative_sample_size
        self.mode = mode
        self.entity_dict = ds.entity_dict

        train_count = defaultdict(lambda: 4)

        if self.args.new_edge_type:
            for i in range(int(self.args.new_edge_frac * len(self.triples['head']))):
                head_type = random.choice(list(self.entity_dict.keys()) or [None])
                tail_type = random.choice(list(self.entity_dict.keys()) or [None])
                head = random.randint(0, self.entity_dict[head_type][1] - self.entity_dict[head_type][0] - 1)
                tail = random.randint(0, self.entity_dict[tail_type][1] - self.entity_dict[tail_type][0] - 1)
                relation = self.nrelation - 1
                ds.train_triples['head'].append(head)
                ds.train_triples['tail'].append(tail)
                ds.train_triples['relation'].append(relation)
                ds.train_triples['head_type'].append(head_type)
                ds.train_triples['tail_type'].append(tail_type)

        for i in range(len(ds.train_triples['head'])):
            head, relation, tail = ds.train_triples['head'][i], ds.train_triples['relation'][i], \
                                   ds.train_triples['tail'][i]
            head_type, tail_type = ds.train_triples['head_type'][i], ds.train_triples['tail_type'][i]
            train_count[(ds.get_head(head, head_type), relation)] += 1
            train_count[(ds.get_tail(tail, tail_type), -relation - 1)] += 1

        self.count = train_count

    def __len__(self):
        if self.args.new_edge_type:
            return len(self.triples['head'])
        return len(self.triples['head'])

    def __getitem__(self, idx):
        ds = self.ds

        head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
        head_type, tail_type = self.triples['head_type'][idx], self.triples['tail_type'][idx]

        if self.args.no_reltype:
            relation = 0

        positive_sample = [ds.get_head(head, head_type), relation, ds.get_tail(tail, tail_type)]

        subsampling_weight = self.count[(ds.get_head(head, head_type), relation)] + self.count[
            (ds.get_tail(tail, tail_type), -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        if self.mode == 'head-batch':
            negative_sample = torch.randint(self.entity_dict[head_type][0], self.entity_dict[head_type][1],
                                            (self.negative_sample_size,))
        elif self.mode == 'tail-batch':
            negative_sample = torch.randint(self.entity_dict[tail_type][0], self.entity_dict[tail_type][1],
                                            (self.negative_sample_size,))
        else:
            raise
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode


class TestDataset(Dataset):
    def __init__(self, triples, args, ds, mode, random_sampling):
        self.triples = triples
        self.ds = ds
        self.nentity = ds.nentity
        self.nrelation = ds.nrelation
        self.args = args

        self.mode = mode
        self.random_sampling = random_sampling
        if random_sampling:
            self.neg_size = args.neg_size_eval_train
        self.entity_dict = ds.entity_dict

    def __len__(self):
        return len(self.triples['head'])

    def __getitem__(self, idx):
        ds = self.ds
        head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]

        if 'head_type' in self.triples:
            head_type, tail_type = self.triples['head_type'][idx], self.triples['tail_type'][idx]
        else:
            head_type = tail_type = None

        if self.args.no_reltype:
            relation = 0

        positive_sample = torch.LongTensor((ds.get_head(head, head_type), relation, ds.get_tail(tail, tail_type)))

        if self.mode == 'head-batch':
            if not self.random_sampling:
                negative_sample = torch.cat([
                    torch.LongTensor([ds.get_head(head, head_type)]),
                    torch.from_numpy(ds.get_head(self.triples['head_neg'][idx], head_type))])
            else:
                negative_sample = torch.cat([
                    torch.LongTensor([head + self.entity_dict[head_type][0]]),
                    torch.randint(self.entity_dict[head_type][0], self.entity_dict[head_type][1], size=(self.neg_size,))])
        elif self.mode == 'tail-batch':
            if not self.random_sampling:
                negative_sample = torch.cat([
                    torch.LongTensor([ds.get_tail(tail, tail_type)]),
                    torch.from_numpy(ds.get_tail(self.triples['tail_neg'][idx], tail_type))])
            else:
                negative_sample = torch.cat([
                    torch.LongTensor([tail + self.entity_dict[tail_type][0]]),
                    torch.randint(self.entity_dict[tail_type][0], self.entity_dict[tail_type][1], size=(self.neg_size,))])

        return positive_sample, negative_sample, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]

        return positive_sample, negative_sample, mode


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data