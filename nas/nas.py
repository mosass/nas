import numpy as np
import copy as copy
import tensorflow as tf
from nasbench import api
from nas import constant as C

class NAS(object):
  times = [0.0]
  best_specs = []

  def __init__(self, file_path="dataset/nasbench_only108.tfrecord", lazy=True):
    super().__init__()
    self.nasbench_file_path = file_path
    if(lazy == False):
      self._load_data()

  def _load_data(self):
    if(not hasattr(self, 'nasbench')):
      self.nasbench = api.NASBench(self.nasbench_file_path)

  def reset_budget(self):
    self._load_data()
    self.nasbench.reset_budget_counters()
  
  def create_spec(self, matrix, ops):
    spec = api.ModelSpec(matrix=matrix, ops=ops)
    if self.nasbench.is_valid(spec):
      return spec
    else:
      return False


  def generate_random_spec(self):
    self._load_data()
    while True:
      matrix = np.random.choice(C.ALLOWED_EDGES, size=(C.NUM_VERTICES, C.NUM_VERTICES))
      matrix = np.triu(matrix, 1)
      ops = np.random.choice(C.ALLOWED_OPS, size=(C.NUM_VERTICES)).tolist()
      ops[0] = C.INPUT
      ops[-1] = C.OUTPUT
      spec = self.create_spec(matrix=matrix, ops=ops)
      if spec != False:
        return spec

  def generate_random_specs(self, size):
    return (self.generate_random_spec() for i in range(1, size))

  def eval_query(self, spec):
    data = self.nasbench.query(spec)
    time_spent, _ = self.nasbench.get_budget_counters()
    self.times.append(time_spent)

    indv = (spec, data)

    if self.compare_indv(indv, self.best_specs[-1]) >= 0:
      self.best_specs.append((spec, data))
    else:
      self.best_specs.append(self.best_specs[-1])

    return indv

  '''
  The return value is negative if indv1 < indv2, zero if indv1 == indv2
  and strictly positive if indv1 > indv1.
  '''
  def compare_indv(self, indv1, indv2):
    return (indv1[1]['validation_accuracy'] > indv2[1]['validation_accuracy']) - (indv1[1]['validation_accuracy'] < indv2[1]['validation_accuracy'])
