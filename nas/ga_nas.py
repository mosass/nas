import numpy as np
import copy as copy
import tensorflow as tf
from nasbench import api
from nas import constant as C
from nas import NAS
import random


class GANAS(NAS):
  MAX_ATTEMPS = 10

  config = {
    'population_size': 4,
    'mutation_rate': 0.5,
    'crossover_rate': 0.5,
  }

  population = []
  offspring_specs = []

  def __init__(self, file_path=None, lazy=True):
    super().__init__(file_path=file_path, lazy=lazy)

  def __init__(self):
    super().__init__()

  def initialization(self):
    self.reset_budget()
    specs = self.generate_random_specs(self.population_size)
    for spec in specs:
      self.population.append((spec, self.fitness(spec)))

  def fitness(self, spec):
    data = self.eval_query(spec)
    return data

  def selection(self):
    sorted(self.population, cmp=self.compare_indv)
    n = len(self.population)
    r = map(float, range(n,0,-1))
    s = float(sum(r))
    prop = [(p/s) for p in r]
    return np.random.choice(self.population, 2, replace=False, p=prop)

  def crossover(self):
    for _ in range(self.MAX_ATTEMPS):
      parents = self.selection()
      offspring = self.mate_spec(parents)
      if offspring != False:
        self.offspring_specs.append(parents)
        break
  
  def mate_spec(self, parents):
    p1_matrix = np.array(copy.deepcopy(parents[0].original_matrix))
    p2_matrix = np.array(copy.deepcopy(parents[1].original_matrix))
    p1_ops = copy.deepcopy(parents[0].original_ops)
    p2_ops = copy.deepcopy(parents[1].original_ops)

    l1 = len(p1_ops)
    l2 = len(p2_ops)
    s1 = l1/2
    s2 = l2/2

    c1_ops = p1_ops[:s1]+p2_ops[s2:]
    c2_ops = p2_ops[:s2]+p1_ops[s1:]
    
    c1_matrix = np.zeros([])

  def mutation(self):
    self.offspring_specs = map(self.mutate_spec, self.offspring_specs)
  
  def mutate_spec(self, old_spec):
    for _ in range(self.MAX_ATTEMPS):
      new_matrix = copy.deepcopy(old_spec.original_matrix)
      new_ops = copy.deepcopy(old_spec.original_ops)

      # In expectation, V edges flipped (note that most end up being pruned).
      edge_mutation_prob = self.config['mutation_rate'] / C.NUM_VERTICES
      for src in range(0, C.NUM_VERTICES - 1):
        for dst in range(src + 1, C.NUM_VERTICES):
          if random.random() < edge_mutation_prob:
            new_matrix[src, dst] = 1 - new_matrix[src, dst]

      # In expectation, one op is resampled.
      op_mutation_prob = self.config['mutation_rate'] / C.OP_SPOTS
      for ind in range(1, C.NUM_VERTICES - 1):
        if random.random() < op_mutation_prob:
          available = [o for o in C.ALLOWED_OPS if o != new_ops[ind]]
          new_ops[ind] = random.choice(available)

      new_spec = self.create_spec(matrix=new_matrix, ops=new_ops)
      if new_spec != False:
        return new_spec
