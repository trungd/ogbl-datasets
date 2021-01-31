from collections import defaultdict

from .base import Basefset
from ogb.linkproppred import LinkPropPredDataset, Evaluator


class BioKG(BaseDataset):
    nentity = None
    nrelation = None

    def __init__(self, args):
        super().__init__()

        dataset = self.dataset = LinkPropPredDataset(name=args.dataset, root=args.dataset_root)
        split_edge = dataset.get_edge_split()
        self.train_triples, self.valid_triples, self.test_triples = split_edge["train"], split_edge["valid"], split_edge["test"]

        self.nentity = sum(self.dataset[0]['num_nodes_dict'].values())
        self.nrelation = int(max(self.train_triples['relation'])) + 1

        if args.new_edge_type:
            self.nrelation += 1

        entity_dict = self.entity_dict = defaultdict(lambda: (0, 0))
        if args.dataset == 'ogbl-biokg' or args.dataset == 'ogbl-wikikg2':
            cur_idx = 0
            for key in dataset[0]['num_nodes_dict']:
                entity_dict[key] = (cur_idx, cur_idx + dataset[0]['num_nodes_dict'][key])
                cur_idx += dataset[0]['num_nodes_dict'][key]

    def get_head(self, head, head_type):
        return head + self.entity_dict[head_type][0]

    def get_tail(self, tail, tail_type):
        return tail + self.entity_dict[tail_type][0]