import random

def generate_random_euclidean__instances(count: int, cnodes: int, squaresize: int):
    # use tsplib format for easy use of tsplib95 imports

    print("Generating instances...")
    for i in range(count):

        # create file and compulsories
        name = "rnd" + str(i) + "_" + str(cnodes) + ".tsp"
        f = open("./instances/" + name, "w+")
        f.write("NAME : " + name + "\n")
        f.write("TYPE : TSP" + "\n")
        f.write("DIMENSION : " + str(cnodes) + "\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D" + "\n")
        f.write("NODE_COORD_SECTION" + "\n")

        # generate coordinates
        for inode in range(cnodes):
            line = str(inode) + " " + str(random.randint(0, squaresize)) + " " + str(random.randint(0, squaresize))
            f.write(line + "\n")

        f.close()