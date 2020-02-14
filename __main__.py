from ga_nas import islga
from nasbench import api

def main():
    nas = islga.ISLGA('nasbench_only108.tfrecord')

    for i in range(1, 100):
        g = nas.generate_random_spac().visualize()
        g.render("output/dot"+str(i), view=True)

if __name__ == "__main__":
    main()