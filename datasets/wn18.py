import os
from collections import defaultdict

from .base import BaseDataset
from ogb.linkproppred import LinkPropPredDataset, Evaluator

import urllib
import numpy as np

from .openke import OpenKE


class WN18(OpenKE):
    @property
    def dataset_name(self):
        return 'WN18'