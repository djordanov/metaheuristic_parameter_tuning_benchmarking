import random

def generate_optimal_solutions(instancefolder: str):

    from pathlib import Path
    import tsplib95
    import numpy as np
    import pandas as pd
    from python_tsp.exact import solve_tsp_dynamic_programming

    print('--- Computing optimal solutions... ---')
    df: pd.DataFrame = pd.DataFrame({'name' : [], 'permutation' : [], 'distance' : []})

    # iterate over instances
    entries = Path(instancefolder)
    for entry in entries.iterdir():
        
        # load problem and get distances
        problem = tsplib95.load(instancefolder + entry.name)
        cnodes = len(list(problem.get_nodes()))
        distances = [ [problem.get_weight(a, b) for b in range(cnodes)] for a in range(cnodes) ]

        # compute optimal solution
        distance_matrix = np.array(distances)
        permutation, distance = solve_tsp_dynamic_programming(distance_matrix)

        # save optimal solution
        print(entry.name)
        print(str(permutation))
        print(distance)
        df = df.append({'name' : entry.name, 'permutation' : str(permutation), 'distance' : str(distance)}, ignore_index = True)
    df.to_csv(instancefolder + 'optimal_solutions.txt', sep = '\t')

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