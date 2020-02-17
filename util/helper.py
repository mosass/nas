from nasbench.lib import model_spec
import logging
logger = logging.getLogger(__name__)

def print_cell(cell, include=[]):
    for k, v in cell.items():
        if len(include) > 0 and k not in include:
            continue
        logger.info('%s: %s', k, str(v))

def print_spec(spec, original=False):
    logger.info('%s: %s', 'Matrix', spec.original_matrix if original else spec.matrix)
    logger.info('%s: %s', 'Ops', spec.original_ops if original else spec.ops)
