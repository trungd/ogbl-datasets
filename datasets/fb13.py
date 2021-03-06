import os
from collections import defaultdict

from .base import BaseDataset
from ogb.linkproppred import LinkPropPredDataset, Evaluator

import urllib
import numpy as np

from .openke import OpenKE


class FB13(OpenKE):
    @property
    def dataset_name(self):
        return 'FB13'