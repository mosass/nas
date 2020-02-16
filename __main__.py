from nas import ga_nas
from nasbench import api
from util import helper
import numpy as np
from nas import constant as C

def main():
    ganas = ga_nas.GANAS()
    ganas.config['population_size'] = 50
    specs = ganas.initialization()

    for s in specs:
        helper.print_spec(s)
        ganas.nasbench.evaluate(s, 'output/eval/t01')


if __name__ == "__main__":
    main()