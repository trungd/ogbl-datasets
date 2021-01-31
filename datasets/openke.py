import os
from collections import defaultdict

from .base import BaseDataset
from ogb.linkproppred import LinkPropPredDataset, Evaluator

import urllib
import numpy as np


class OpenKE(BaseDataset):
    nentity = None
    nrelation = None

    def __init__(self, args):
        super().__init__()

        self.data_root = os.path.join(args.dataset_root, self.dataset_name.lower())
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root, exist_ok=True)
            for url in [
                    "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/%s/train2id.txt" % self.dataset_name,
                    "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/%s/test2id.txt" % self.dataset_name,
                    "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/%s/valid2id.txt" % self.dataset_name,
                    "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/%s/relation2id.txt" % self.dataset_name,
                    "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/%s/entity2id.txt" % self.dataset_name,
                ]:
                urllib.request.urlretrieve(url, filename=os.path.join(self.data_root, url.split('/')[-1]))

        for mode in ['train', 'valid', 'test']:
            with open(os.path.join(self.data_root, f'{mode}2id.txt')) as f:
                lines = f.read().split('\n')[1:]
                triples = [[int(x) for x in (l.split('\t') if '\t' in l else l.split(' '))] for l in lines if l]
                setattr(self, f"{mode}_triples", dict(
                    head=np.array([x[0] for x in triples]),
                    tail=np.array([x[1] for x in triples]),
                    relation=np.array([x[2] for x in triples]),
                    head_type=[None for _ in triples],
                    tail_type=[None for _ in triples]
                ))

        self.nentity = int(open(os.path.join(self.data_root, 'entity2id.txt')).readline())
        self.nrelation = int(open(os.path.join(self.data_root, 'relation2id.txt')).readline())

        if args.resample:
            self.load_negative_samples('resampled', True)
        else:
            self.load_negative_samples('uniform', False)

        if args.new_edge_type:
            self.nrelation += 1

        self.entity_dict = defaultdict(lambda: (0, self.nentity))

    @property
    def dataset_name(self):
        raise NotImplemented

    def get_head(self, head, head_type):
        return head

    def get_tail(self, tail, tail_type):
        return tail