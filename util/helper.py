from nasbench.lib import model_spec

def print_cell(cell):
    for k, v in cell.items():
        print('%s: %s' % (k, str(v)))

def print_spec(spec: model_spec.ModelSpec):
    print('%s: %s' % ('Matrix', spec.matrix))
    print('%s: %s' % ('Ops', spec.ops))

def print_graph(spec):
    pass