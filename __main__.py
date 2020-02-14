from nas import ga_nas
from nasbench import api

def main():
    ganas = ga_nas.GANAS()

    for i in range(1, 10):
        g = ganas.generate_random_spac().visualize()
        g.render("output/dot"+str(i), view=False)

if __name__ == "__main__":
    main()