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
    gen = 0
    logger.info('generation[%d] : %f : %s ', gen, ganas.times[-1], ganas.best_specs[-1][1]['validation_accuracy'])
    while ganas.times[-1] < 5000000:
        gen += 1
        logger.debug('--------generation : %d---------', gen)
        ganas.crossover()
        ganas.mutation()
        ganas.evolve()

        if gen % 100 == 0:
            logger.info('generation[%d] : %f : %s ', gen+1, ganas.times[-1], ganas.best_specs[-1][1]['validation_accuracy'])
    
    return ganas


if __name__ == "__main__":
    nasbench = api.NASBench('dataset/nasbench_only108.tfrecord')
    for epoch in range(3):
        ganas = run_ga(nasbench)
        logger.info('epoch[%d] : %f : %s ', epoch, ganas.times[-1], ganas.best_specs[-1][1]['validation_accuracy'])