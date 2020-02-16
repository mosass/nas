import nas.ga_nas as ga_nas
import nasbench.api as api
import util.helper as helper
import numpy as np
import nas.constant as C

def main(nasbench):
    ganas = ga_nas.GANAS(nasbench)
    ganas.initialization()

    print('--------initialization---------')
    for s in ganas.population:
        helper.print_spec(s[0])
        helper.print_cell(s[1], include=['validation_accuracy'])
    print('-------------------------------')
    print('--------test-selection--------------')
    for s in ganas.selection():
        helper.print_cell(s[1], include=['validation_accuracy'])
    print('-------------------------------')

    ganas.crossover()
    print('--------crossover-parents--------------')
    for s in ganas.parent_specs:
        helper.print_spec(s[0][0])
        helper.print_cell(s[0][1], include=['validation_accuracy'])
        helper.print_spec(s[1][0])
        helper.print_cell(s[1][1], include=['validation_accuracy'])
    print('-------------------------------')
    print('--------crossover-offsprings--------------')
    for s in ganas.offspring_specs:
        helper.print_spec(s)
    print('-------------------------------')
    ganas.mutation()
    print('--------mutation-offsprings--------------')
    for s in ganas.offspring_specs:
        helper.print_spec(s)
    print('-------------------------------')


if __name__ == "__main__":
    nasbench = api.NASBench('dataset/nasbench_only108.tfrecord')
    main(nasbench)