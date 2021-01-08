import random
from pathlib import Path
import tsplib95

import numpy as np
import pandas as pd

from concorde.tsp import TSPSolver # wrapper around concorde
import os

def generate_optimal_solutions(instancefolder: str):

    # iterate over instances
    entries = Path(instancefolder)
    for entry in entries.iterdir():
        
        if entry.suffix != '.tsp':
            continue

        # solve with concorde
        solver = TSPSolver.from_tspfile(str(entry.absolute()))
        solution = solver.solve()
        tour = solution.tour + 1 # nodes start at 1
        
        # save optimal solution
        optpath = Path(instancefolder + '/' + entry.name).with_suffix('.opt.tour')
        fopt = optpath.open('w+')
        fopt.write('NAME : ' + entry.name + '.tsp.opt.tour\n')
        fopt.write('TYPE : TOUR\n')
        fopt.write('DIMENSION : '+ str(len(tour)) + '\n')
        fopt.write('TOUR_SECTION''\n')
        for node in tour:
            fopt.write(str(node) + '\n')
        fopt.write('-1')
        fopt.close()

def generate_random_euclidean_instances(dir: str, count: int, cnodes: int, squaresize: int):
    # use tsplib format

    # create directory if not exists
    Path(dir).mkdir(parents = True, exist_ok = True)

    # create problems...
    for i in range(count):

        # create file and compulsories
        name = "rnd" + str(i) + "_" + str(cnodes) + ".tsp"
        f = open(dir + name, "w")
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

dir = 'myproject/instances/20nodes/'
generate_random_euclidean_instances(dir, count = 50, cnodes = 20, squaresize = 1000)
generate_optimal_solutions(dir)
generate_random_euclidean_instances(dir + 'test/', count = 50, cnodes = 20, squaresize = 1000)
generate_optimal_solutions(dir + 'test/')
os.system('ls ' + dir + '*.tsp | sort > ' + dir + 'trainInstancesFile')