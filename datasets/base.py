from tqdm import tqdm


class BaseDataset:
    def __init__(self):
        super().__init__()

    @property
    def nentity(self):
        raise NotImplemented

    @property
    def nrelation(self):
        raise NotImplemented


def resample(train, valid, test):
    '''
    Resample the test set negatives.
    Note that the negatives will generally be ordered -
    it would be easy to shuffle them, but this seems ok for the evaluators we have.
    '''

    import pandas as pd
    import shutil, os, string
    import os.path as osp
    from ogb.utils.url import decide_download, download_url, extract_zip
    from ogb.io.read_graph_raw import read_csv_graph_raw, read_csv_heterograph_raw, read_binary_graph_raw, \
        read_binary_heterograph_raw
    import torch
    import numpy as np
    import random

    newsplit = 'byrel'

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
        print("Generating valid/test set...")
        if 'head_neg' not in s:
            s['head_neg'] = np.zeros([s['head'].shape[0], 100], dtype=np.int)
            s['tail_neg'] = np.zeros([s['head'].shape[0], 100], dtype=np.int)
        for i in tqdm(range(s['head'].shape[0])):
            # r = s['relation'][i]
            # nh, nt = len(heads.setdefault(r, [])), len(tails.setdefault(r, []))
            # if i % 100 == 1:
            #     print(i, r, nh, nt)
            hnl, tnl = len(s['head_neg'][i]), len(s['tail_neg'][i])
            s['head_neg'][i] = sample(hnl, 'head', s['relation'][i], s['tail'][i])
            s['tail_neg'][i] = sample(tnl, 'tail', s['relation'][i], s['head'][i])

    return valid, test
    # os.system('mkdir -p ' + data_in + '/split/' + newsplit)
    # torch.save(test, data_in + '/split/' + newsplit + '/test.pt')