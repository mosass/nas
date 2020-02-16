import numpy as np
import copy as copy
import tensorflow as tf
from nasbench import api
import constant as C
from nas import NAS
import random
from util import helper


class GANAS(NAS):
  MAX_ATTEMPS = 10

  config = {
    'population_size': 4,
    'mutation_rate': 1.0,
    'crossover_rate': 0.5,
  }

  population = []
  parent_specs = []
  offspring_specs = []

  def initialization(self):
    self.reset_budget()
    self.population = []
    specs = self.generate_random_specs(self.config['population_size'])
    for spec in specs:
      self.population.append(self.fitness(spec))

  def fitness(self, spec):
    data = self.eval_query(spec)
    return data

  def selection(self):
    sorted(self.population, cmp=self.compare_indv)
    n = len(self.population)
    r = map(float, range(n,0,-1))
    s = float(sum(r))
    prop = [(p/s) for p in r]
    ind = np.random.choice(range(n), 2, replace=False, p=prop)
    return tuple(self.population[i] for i in ind)

  def crossover(self):
    for _ in range(self.MAX_ATTEMPS):
      parents = self.selection()
      self.parent_specs.append(parents)
      offsprings = self.mate_spec(parents)
      if offsprings != False:
        self.offspring_specs += offsprings
        break
  
  def mate_spec(self, parents):
    p1_matrix = np.array(copy.deepcopy(parents[0][0].original_matrix))
    p2_matrix = np.array(copy.deepcopy(parents[1][0].original_matrix))
    p1_ops = copy.deepcopy(parents[0][0].original_ops)
    p2_ops = copy.deepcopy(parents[1][0].original_ops)

    pl1 = len(p1_ops)
    pl2 = len(p2_ops)
    s1 = pl1/2
    s2 = pl2/2

    c1_ops = p1_ops[:s1]+p2_ops[s2:]
    c2_ops = p2_ops[:s2]+p1_ops[s1:]

    cl1 = len(c1_ops)
    cl2 = len(c2_ops)
    
    c1_matrix = np.zeros([cl1, cl1])
    c2_matrix = np.zeros([cl2, cl2])

    c1_matrix[:s1,:s1] = p1_matrix[:s1, :s1]
    if cl1 < pl2: 
      c1_matrix[:,s1:] = p2_matrix[:cl1, s2:]
    else:
      c1_matrix[cl1 - pl2:,s1:] = p2_matrix[:, s2:]

    c1_matrix = np.triu(c1_matrix, 1)

    c2_matrix[:s2,:s2] = p2_matrix[:s2, :s2]
    if cl2 < pl1: 
      c2_matrix[:,s2:] = p1_matrix[:cl2, s1:]
    else:
      c2_matrix[cl2 - pl1:,s2:] = p1_matrix[:, s1:]

    c2_matrix = np.triu(c2_matrix, 1)

    c1_spec = self.create_spec(matrix=c1_matrix, ops=c1_ops)
    c2_spec = self.create_spec(matrix=c2_matrix, ops=c2_ops)

    if c1_spec != False and c2_spec != False:
      return [c1_spec, c2_spec]

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
