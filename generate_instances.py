import random

def generate_optimal_solutions(instancefolder: str):

    from pathlib import Path
    import tsplib95
    import numpy as np
    import pandas as pd
    from python_tsp.exact import solve_tsp_dynamic_programming

    print('--- Computing optimal solutions... ---')

    # iterate over instances
    entries = Path(instancefolder)
    for entry in entries.iterdir():
        
        if entry.suffix != '.tsp':
            continue

        # load problem and get distances
        print(entry.absolute())
        problem = tsplib95.load(entry.absolute())
        cnodes = len(list(problem.get_nodes()))
        distances = [ [problem.get_weight(a, b) for b in range(cnodes)] for a in range(cnodes) ]

        # compute optimal solution
        distance_matrix = np.array(distances)
        permutation, distance = solve_tsp_dynamic_programming(distance_matrix)

        # save optimal solution
        print(str(permutation))
        print(distance)
        optpath = Path(instancefolder + entry.name).with_suffix('.opt.tour')
        fopt = optpath.open('w+')
        fopt.write('NAME : ' + entry.name + '.tsp.optbc.tour\n')
        fopt.write('TYPE : TOUR\n')
        fopt.write('DIMENSION : '+ str(problem.dimension) + '\n')
        fopt.write('TOUR_SECTION''\n')
        for node in permutation:
            fopt.write(str(node) + '\n')
        fopt.write('-1')
        fopt.close()

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