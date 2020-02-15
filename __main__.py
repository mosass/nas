from nas import ga_nas
from nasbench import api
from util import helper

def main():
    ganas = ga_nas.GANAS()
    specs = ganas.initialization()

    for s in specs:
        helper.print_spec(s)

if __name__ == "__main__":
    main()