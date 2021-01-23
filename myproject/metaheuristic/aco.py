from pathlib import Path
import random
import time
from operator import attrgetter

import numpy as np
import pandas as pd

import tsplib95
from myproject.metaheuristic.commons import Solution, Convdata, iterimprov_2opt

def constructAntSolution(problem: tsplib95.models.StandardProblem, weights: np.array) -> Solution:

    # start tour
    tour = [random.randint(1, problem.dimension)]

    # finish tour
    while len(tour) < problem.dimension:

        # helper variables
        current_node = tour[-1]
        weights_slice = weights[current_node-1].copy()
        for node in tour:
            weights_slice[node - 1] = 0
        sum_weights = sum(weights_slice)

        if (sum(weights_slice) == 0): 
            tour.append(random.choice( list( set(problem.get_nodes()) - set(tour)) ))
            continue

        probabilities = weights_slice / sum_weights

        # select move
        probability = random.random()

        probs_cumsum = 0
        for node in problem.get_nodes():
            probs_cumsum += probabilities[node - 1]
            if probs_cumsum > probability:
                tour.append(node)              
                break

    return Solution(problem.trace_tours([tour])[0], tour)

def updatePheromones(pheromone_matrix: list, evaporation: float, Q: float, ants: list) -> list:

    # pheromone evaporation
    pheromone_matrix *= evaporation

    # pheromone adding
    for ant in ants:
        # add pheromones for last/ first edge
        firstNode = ant.tour[0]
        lastNode = ant.tour[-1]
        pheromone_matrix[firstNode - 1][lastNode - 1] = pheromone_matrix[firstNode - 1][lastNode - 1] + (Q / ant.qual)

        # add pheromones for all the other edges
        for a in range(len(ant.tour) - 1):
            nodeFrom = ant.tour[a]
            nodeTo = ant.tour[a+1]
            pheromone_matrix[nodeFrom - 1][nodeTo - 1] = pheromone_matrix[nodeFrom - 1][nodeTo - 1] + (Q / ant.qual)

    return pheromone_matrix

def aco(instance: Path, 
        cfg: dict,
        terminate: dict, 
        fname_convdata: Path = None) -> dict:
    starttime = time.perf_counter()
    convdata = [] if fname_convdata != None else None

    # problem: tsplib95.models.StandardProblem
    problem: tsplib95.models.StandardProblem = tsplib95.load(instance.absolute())
    optimaltour: tsplib95.models.StandardProblem = tsplib95.load(instance.with_suffix('.opt.tour').absolute())
    optimal_quality: int = problem.trace_tours(optimaltour.tours)[0]

    # setup...
    best = Solution(problem.trace_canonical_tour(), problem) # default
    evals = 0
    ants = [] # initialize here because its needed in termination condition

    # distance matrix
    distance_matrix = np.empty((problem.dimension, problem.dimension))
    for edge in problem.get_edges():
        distance_matrix[edge[0] - 1][edge[1] - 1] = float(problem.get_weight(edge[0], edge[1]))
    np.fill_diagonal(distance_matrix, np.inf) # probably unnecessary, since these edges get excluded anyways, but it removes warning
    distance_matrix[ distance_matrix == 0 ] = 0.1**10 # some distances are zero, which messes up computation

    # pheromone matrix
    pheromone_matrix = np.full((problem.dimension, problem.dimension), cfg['initial_pheromone'])

    while not ('evals' in terminate and evals >= terminate['evals'] \
        or 'qualdev' in terminate and best.qual < optimal_quality * (1 + terminate['qualdev']) \
        or 'time' in terminate and time.perf_counter() - starttime > terminate['time'] \
        or 'iterations' in terminate and evals / cfg['antcount'] > terminate['iterations']): 
    
        # construct ant solutions
        weights = (1/distance_matrix)**cfg['alpha'] * pheromone_matrix**cfg['beta']
        ants = [constructAntSolution(problem, weights) for _ in range(cfg['antcount'])]
        evals += cfg['antcount']

        # update best reached quality (could be done a little bit more speed friendly)
        itbest = min(ants, key = attrgetter('qual'))
        if best == None or itbest.qual < best.qual:
            best = itbest

        # update pheromones...
        pheromone_matrix = updatePheromones(pheromone_matrix, cfg['evaporation'], cfg['Q'], ants)

        # save state if convergence is looked for 
        if convdata != None:
            qualdev = (best.qual - optimal_quality) / optimal_quality
            convdata.append(Convdata(instance.name, qualdev, evals, time.perf_counter() - starttime))

        
    if convdata != None:
        print_headers: bool = False if fname_convdata.exists() else True
        pd.DataFrame(convdata).to_csv(fname_convdata.absolute(),  index = None, mode = 'a', header = print_headers)

    return {'qualdev': (best.qual - optimal_quality) / optimal_quality, 'evals': evals, 'time': time.perf_counter() - starttime}
