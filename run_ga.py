import nas.ga_nas as ga_nas
import nasbench.api as api
import util.helper as helper
import numpy as np
import nas.constant as C
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_ga(nasbench):
    ganas = ga_nas.GANAS(nasbench)
    ganas.initialization()
    logger.info('--------initialization---------')
    logger.info('generation[%d] : %f : %s ', 0, ganas.times[-1], ganas.best_specs[-1][1]['validation_accuracy'])
    for gen in range(100):
        logger.debug('--------generation : %d---------', gen+1)
        ganas.crossover()
        ganas.mutation()
        ganas.evolve()

        logger.info('generation[%d] : %f : %s ', gen+1, ganas.times[-1], ganas.best_specs[-1][1]['validation_accuracy'])


if __name__ == "__main__":
    nasbench = api.NASBench('dataset/nasbench_only108.tfrecord')
    for epoch in range(5):
        run_ga(nasbench)