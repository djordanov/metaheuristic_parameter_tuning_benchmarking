import random
from pathlib import Path
import tsplib95

import numpy as np
import pandas as pd
from python_tsp.exact import solve_tsp_dynamic_programming

import os
from rpy2 import robjects

CONCORDE_PATH = Path('myproject/')

def generate_optimal_solutions(instancefolder: str):

    print('--- Computing optimal solutions... ---')

    # solve with concorde
    robjects.r('library("TSP")')
    robjects.r('concorde_path(\"' + str(CONCORDE_PATH.absolute()) + '\")')

    # iterate over instances
    entries = Path(instancefolder)
    for entry in entries.iterdir():
        
        if entry.suffix != '.tsp':
            continue

        # solve with concorde
        robjects.r('problem = read_TSPLIB(\"' + str(entry.absolute()) +  '\")')
        robjects.r('tour = solve_TSP(problem, method = "concorde")')
        tour = list(robjects.r('as.integer(tour)'))
        
        # save optimal solution
        optpath = Path(instancefolder + '/' + entry.name).with_suffix('.opt.tour')
        fopt = optpath.open('w+')
        fopt.write('NAME : ' + entry.name + '.tsp.optbc.tour\n')
        fopt.write('TYPE : TOUR\n')
        fopt.write('DIMENSION : '+ str(len(tour)) + '\n')
        fopt.write('TOUR_SECTION''\n')
        for node in tour:
            fopt.write(str(node) + '\n')
        fopt.write('-1')
        fopt.close()

def generate_random_euclidean_instances(dir: Path, count: int, cnodes: int, squaresize: int):
    # use tsplib format

    # create directory if not exists
    dir.mkdir(parents = True, exist_ok = True)

    # create problems...
    for i in range(count):

        # create file and compulsories
        name = "rnd" + str(i) + "_" + str(cnodes) + ".tsp"
        f = open(str(dir.absolute()) + '/' + name, "w")
        f.write("NAME : " + name + "\n")
        f.write("TYPE : TSP" + "\n")
        f.write("DIMENSION : " + str(cnodes) + "\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D" + "\n")
        f.write("NODE_COORD_SECTION" + "\n")

        # generate coordinates
        for inode in range(1, cnodes + 1):
            line = str(inode) + " " + str(random.randint(0, squaresize)) + " " + str(random.randint(0, squaresize))
            f.write(line + "\n")
        f.close()

generate_random_euclidean_instances(Path('myproject/instances/50nodes/'), count = 50, cnodes = 50, squaresize = 1000)
generate_optimal_solutions('myproject/instances/50nodes')
generate_random_euclidean_instances(Path('myproject/instances/50nodes/test'), count = 50, cnodes = 50, squaresize = 1000)
generate_optimal_solutions('myproject/instances/50nodes/test')
os.system('ls myproject/instances/50nodes/ *.tsp | sort > myproject/instances/50nodes/trainInstancesFile')

generate_random_euclidean_instances(Path('myproject/instances/20nodes'), count = 50, cnodes = 20, squaresize = 1000)
generate_optimal_solutions('myproject/instances/20nodes')
generate_random_euclidean_instances(Path('myproject/instances/20nodes/test'), count = 50, cnodes = 20, squaresize = 1000)
generate_optimal_solutions('myproject/instances/20nodes/test')
os.system('ls myproject/instances/20nodes/ *.tsp | sort > myproject/instances/50nodes/trainInstancesFile')

generate_random_euclidean_instances(Path('myproject/instances/100nodes'), count = 50, cnodes = 100, squaresize = 1000)
generate_optimal_solutions('myproject/instances/100nodes')
generate_random_euclidean_instances(Path('myproject/instances/100nodes/test'), count = 50, cnodes = 100, squaresize = 1000)
generate_optimal_solutions('myproject/instances/100nodes/test')
os.system('ls myproject/instances/100nodes/ *.tsp | sort > myproject/instances/50nodes/trainInstancesFile')