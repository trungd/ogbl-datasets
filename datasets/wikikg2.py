from collections import defaultdict

from .base import BaseDataset
from ogb.linkproppred import LinkPropPredDataset, Evaluator


class WikiKG2(BaseDataset):
    nentity = None
    nrelation = None

    def __init__(self, args):
        super().__init__()

        dataset = self.dataset = LinkPropPredDataset(name='ogbl-wikikg2', root=args.dataset_root)
        split_edge = dataset.get_edge_split()
        self.train_triples, self.valid_triples, self.test_triples = split_edge["train"], split_edge["valid"], split_edge["test"]

        for triples in [self.train_triples, self.valid_triples, self.test_triples]:
            for i in triples:
                triples[i]['head_type'] = triples[i]['tail_type'] = None

        self.nentity = dataset.graph['num_nodes']
        self.nrelation = int(max(dataset.graph['edge_reltype'])[0]) + 1

        if args.new_edge_type:
            self.nrelation += 1

        self.entity_dict = defaultdict(lambda: (0, len(self.nentity)))

    def get_head(self, head, head_type):
        return head

    def get_tail(self, tail, tail_type):
        return tail