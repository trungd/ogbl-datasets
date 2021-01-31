from tqdm import tqdm
import os
import numpy as np
import random
import torch


class BaseDataset:
    data_root = None
    train_triples = valid_triples = test_triples = None

    def __init__(self):
        super().__init__()

    @property
    def nentity(self):
        raise NotImplemented

    @property
    def nrelation(self):
        raise NotImplemented

    def load_negative_samples(self, split='default', resample=True, sample_size=500):
        '''
        Resample the test set negatives.
        Note that the negatives will generally be ordered -
        it would be easy to shuffle them, but this seems ok for the evaluators we have.
        '''

        train, valid, test = self.train_triples, self.valid_triples, self.test_triples

        if os.path.exists(os.path.join(self.data_root, split, 'test.pt')):
            print("Load valid/test from file...")
            self.test_triples = torch.load(os.path.join(self.data_root, split, 'test.pt'))
            self.valid_triples = torch.load(os.path.join(self.data_root, split, 'valid.pt'))
            return

        print("Generating valid/test set...")
        if resample:
            fract = 0.5  # fraction to take which are tails of the relation
            extra = 1.1  # extra sample so that we can exclude triples in the graph

            heads = dict()
            tails = dict()
            ht = {'head': heads, 'tail': tails}
            at = {'head': train['head'].tolist(), 'tail': train['tail'].tolist()}
            all = set()

            # return N samples of field f for relation r
            def sample(N, f, r, v, ex=extra):
                Nr = int(N * fract)
                Nt = int(N * ex)
                if Nr < len(ht[f][r]):
                    sl = random.sample(ht[f][r], Nr)
                else:
                    sl = ht[f][r]
                sl = sl + random.sample(at[f], Nt - len(sl))
                # remove those for which (x,r,v) or (v,r,x) is in the graph
                if f == 'head':
                    sl = [x for x in sl if (x, r, v) not in all]
                else:
                    sl = [x for x in sl if (v, r, x) not in all]
                # remove duplicates
                sl = list(set(sl))
                if len(sl) < N:
                    # print('error: too few negs', len(sl), 'for item', f, r, v, 'try again')
                    sl = sample(N, f, r, v, ex * ex * N / (len(sl) + 1))
                return sl[:N]

            def triples(l):
                for t in l:
                    for i in range(t['head'].shape[0]):
                        yield (t['head'][i], t['relation'][i], t['tail'][i])

            for h, r, t in triples([train, valid, test]):
                heads.setdefault(r, []).append(h)
                tails.setdefault(r, []).append(t)
                all.add((h, r, t))

            # also valid

            for s in [valid, test]:
                if 'head_neg' not in s:
                    s['head_neg'] = np.zeros([s['head'].shape[0], sample_size], dtype=np.int)
                    s['tail_neg'] = np.zeros([s['head'].shape[0], sample_size], dtype=np.int)
                for i in tqdm(range(s['head'].shape[0])):
                    # r = s['relation'][i]
                    # nh, nt = len(heads.setdefault(r, [])), len(tails.setdefault(r, []))
                    # if i % 100 == 1:
                    #     print(i, r, nh, nt)
                    hnl, tnl = len(s['head_neg'][i]), len(s['tail_neg'][i])
                    s['head_neg'][i] = sample(hnl, 'head', s['relation'][i], s['tail'][i])
                    s['tail_neg'][i] = sample(tnl, 'tail', s['relation'][i], s['head'][i])
        else:
            for s in [valid, test]:
                s['head_neg'] = np.random.randint(0, self.nentity, [s['head'].shape[0], sample_size], dtype=np.int)
                s['tail_neg'] = np.random.randint(0, self.nentity, [s['head'].shape[0], sample_size], dtype=np.int)

        self.valid_triples = valid
        self.test_triples = test
        os.makedirs(os.path.join(self.data_root, split), exist_ok=True)
        torch.save(test, os.path.join(self.data_root, split, 'test.pt'))
        torch.save(test, os.path.join(self.data_root, split, 'valid.pt'))