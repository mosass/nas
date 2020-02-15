import numpy as np
import copy as copy
import tensorflow as tf
from nasbench import api
from nas import constant as C

class NAS(object):
    times: [] = [0.0]
    best_specs: []

    def __init__(self, file_path="dataset/nasbench_only108.tfrecord", lazy=True):
        super().__init__()
        self.nasbench_file_path = file_path
        if(lazy == False):
            self._load_data()

    def _load_data(self):
        if(not hasattr(self, 'nasbench')):
            self.nasbench = api.NASBench(self.nasbench_file_path)

    def generate_random_spac(self):
        self._load_data()
        while True:
            matrix = np.random.choice(C.ALLOWED_EDGES, size=(C.NUM_VERTICES, C.NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(C.ALLOWED_OPS, size=(C.NUM_VERTICES)).tolist()
            ops[0] = C.INPUT
            ops[-1] = C.OUTPUT
            spec = api.ModelSpec(matrix=matrix, ops=ops)
            if self.nasbench.is_valid(spec):
                return spec

    def generate_random_spacs(self, size):
        return tuple(self.generate_random_spac() for i in range(1, size))

    def eval_query(self, spec):
        data = self.nasbench.query(spec)
        time_spent, _ = nasbench.get_budget_counters()
        self.times.append(time_spent)
        self.best_specs.append(self.compare_spec(self.best_specs[-1], data))
        return data

    ''' return the best spec'''
    def compare_spec(self, data, data_new):
        raise NotImplementedError
