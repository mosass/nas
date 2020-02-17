import nas.ga_nas as ga_nas
import nasbench.api as api
import util.helper as helper
import numpy as np
import nas.constant as C
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main(nasbench):
    ganas = ga_nas.GANAS(nasbench)
    ganas.initialization()
    logger.info('--------initialization---------')

    for _ in range(5):
        for s in ganas.population:
            helper.print_spec(s[0])
            helper.print_cell(s[1], include=['validation_accuracy'])
        logger.info('-------------------------------')
        ganas.crossover()
        logger.info('--------crossover-parents--------------')
        for s in ganas.parent_specs:
            helper.print_spec(s[0][0])
            helper.print_cell(s[0][1], include=['validation_accuracy'])
            helper.print_spec(s[1][0])
            helper.print_cell(s[1][1], include=['validation_accuracy'])
        logger.info('-------------------------------')
        logger.info('--------crossover-offsprings--------------')
        for s in ganas.offspring_specs:
            helper.print_spec(s)
        logger.info('-------------------------------')
        # ganas.mutation()
        # logger.info('--------mutation-offsprings--------------')
        # for s in ganas.offspring_specs:
        #     helper.print_spec(s)
        #     helper.print_cell(nasbench.query(s))
        # logger.info('-------------------------------')

        ganas.evolve()


if __name__ == "__main__":
    nasbench = api.NASBench('dataset/nasbench_only108.tfrecord')
    main(nasbench)