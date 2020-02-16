from nasbench.lib import model_spec

def print_cell(cell, include=[]):
    for k, v in cell.items():
        if len(include) > 0 and k not in include:
            continue
        print('%s: %s' % (k, str(v)))

def print_spec(spec, original=False):
    print('%s: %s' % ('Matrix', spec.original_matrix if original else spec.matrix))
    print('%s: %s' % ('Ops', spec.original_ops if original else spec.ops))
