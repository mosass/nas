import numpy as np
import copy as copy
import tensorflow as tf
from nasbench import api
from nas import constant as C
from nas import nas

class GANAS(nas.NAS):

    population_size: int = 50
    mutation_rate: float = 1.0

    population = []

    def __init__(self, file_path=None, lazy=True):
        super().__init__(file_path=file_path, lazy=lazy)

    def __init__(self):
        super().__init__()
    
    def config(self, population_size=50, mutation_rate=1.0):
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def initialization(self):
        self.nasbench.reset_budget_counters()
        return self.generate_random_spacs(self.population_size)

    def fitness(self, spec):
        data = self.eval_query(spec)
        time_spent, _ = nasbench.get_budget_counters()
        times.append(time_spent)
        population.append((data['validation_accuracy'], spec))
        return self.nasbench.query(spec)