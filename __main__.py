from nas import ga_nas
from nasbench import api
from util import helper
import numpy as np
from nas import constant as C

def main():
    # ganas = ga_nas.GANAS()
    # ganas.config(population_size=1)
    # specs = ganas.initialization()

    # for s in specs:
    #     helper.print_spec(s)
    #     ganas.nasbench.evaluate(s, 'output/eval/t01')

    nasbench = api.NASBench('dataset/nasbench_only108.tfrecord')
    nasbench.config['use_tpu'] = False
    nasbench.config['train_data_files'] = [
        'dataset/cifar10/train_1.tfrecords',
        'dataset/cifar10/train_2.tfrecords',
        'dataset/cifar10/train_3.tfrecords',
        'dataset/cifar10/train_4.tfrecords'
    ]
    nasbench.config['valid_data_file'] = 'dataset/cifar10/validation.tfrecords'
    nasbench.config['test_data_file'] = 'dataset/cifar10/test.tfrecords'
    nasbench.config['sample_data_file'] = 'dataset/cifar10/sample.tfrecords'

    while True:
        matrix = np.random.choice(C.ALLOWED_EDGES, size=(C.NUM_VERTICES, C.NUM_VERTICES))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(C.ALLOWED_OPS, size=(C.NUM_VERTICES)).tolist()
        ops[0] = C.INPUT
        ops[-1] = C.OUTPUT
        spec = api.ModelSpec(matrix=matrix, ops=ops)
        if nasbench.is_valid(spec):
            helper.print_spec(spec)
            nasbench.evaluate(spec, 'output/t01')
            break



if __name__ == "__main__":
    main()