import os
from collections import defaultdict

from .base import BaseDataset, resample
from ogb.linkproppred import LinkPropPredDataset, Evaluator

import urllib
import numpy as np


class FB15K237(BaseDataset):
    nentity = None
    nrelation = None

    def __init__(self, args):
        super().__init__()

        self.data_root = os.path.join(args.dataset_root, 'fb15k237')
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root, exist_ok=True)
            for url in [
                    "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/FB15K237/train2id.txt",
                    "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/FB15K237/test2id.txt",
                    "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/FB15K237/valid2id.txt",
                    "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/FB15K237/relation2id.txt",
                    "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/FB15K237/entity2id.txt",
                ]:
                urllib.request.urlretrieve(url, filename=os.path.join(self.data_root, url.split('/')[-1]))

        for mode in ['train', 'valid', 'test']:
            with open(os.path.join(self.data_root, f'{mode}2id.txt')) as f:
                lines = f.read().split('\n')[1:]
                triples = [[int(x) for x in l.split(' ')] for l in lines if l]
                setattr(self, f"{mode}_triples", dict(
                    head=np.array([x[0] for x in triples]),
                    tail=np.array([x[1] for x in triples]),
                    relation=np.array([x[2] for x in triples]),
                    head_type=[None for _ in triples],
                    tail_type=[None for _ in triples]
                ))

        if args.resample:
            self.valid_triples, self.test_triples = resample(self.train_triples, self.valid_triples, self.test_triples)

        self.nentity = int(open(os.path.join(self.data_root, 'entity2id.txt')).readline())
        self.nrelation = int(open(os.path.join(self.data_root, 'relation2id.txt')).readline())

        if args.new_edge_type:
            self.nrelation += 1

        self.entity_dict = defaultdict(lambda: (0, self.nentity))

    def get_head(self, head, head_type):
        return head

    def get_tail(self, tail, tail_type):
        return tail