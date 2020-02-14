import numpy as np
import copy as copy
import tensorflow as tf
from nasbench import api
from nas import constant as C
from nas import nas

class GANAS(nas.NAS):

    population = []

    def __init__(self, file_path=None, lazy_load=True):
        super().__init__(file_path=file_path, lazy_load=lazy_load)

    def __init__(self):
        super().__init__()
        