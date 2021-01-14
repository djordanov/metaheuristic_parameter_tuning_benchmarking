from pathlib import Path
import random
import time
from operator import attrgetter

import numpy as np
import pandas as pd

import tsplib95
from myproject.metaheuristic.commons import Solution, Convdata, iterimprov_2opt

def constructAntSolution(problem: tsplib95.models.StandardProblem,
            distance_matrix: list, pheromone_matrix: list, alpha: float, beta: float) -> Solution:

    # start tour
    tour = [random.randint(1, problem.dimension)]

    # finish tours
    while len(tour) < problem.dimension:

        # helper variables
        current_node = tour[-1]
        edges_distances = distance_matrix[current_node]
        edges_pheromones = pheromone_matrix[current_node]
        possible_moves = set(problem.get_nodes()) - set(tour)
        
        # move probabilities
        unnormed_probabilities = [0] # nodes start at 1, so node 0 has a probability of ß
        for node in problem.get_nodes():
            if node in possible_moves:
                probability = max((1/edges_distances[node])**alpha * edges_pheromones[node]**beta, 0.00001)
                unnormed_probabilities.append(probability)
            else:
                unnormed_probabilities.append(0)
        sum_unnormed = sum(unnormed_probabilities)
        move_probabilities = [prob / sum_unnormed for prob in unnormed_probabilities]

        # select move
        probability = random.random()

        probs_cumsum = 0
        for node in problem.get_nodes():
            probs_cumsum += move_probabilities[node]
            if probs_cumsum > probability:
                tour.append(node)              
                break

    return Solution(tour, problem.trace_tours([tour])[0])

def updatePheromones(pheromone_matrix: list, evaporation: float, Q: float, ants: list) -> list:

    # pheromone evaporation
    new_pheromones = [[cell * evaporation for cell in row] for row in pheromone_matrix]

    # pheromone adding
    for ant in ants:
        # add pheromones for last/ first edge
        firstNode = ant.tour[0]
        lastNode = ant.tour[-1]
        new_pheromones[firstNode][lastNode] = new_pheromones[firstNode][lastNode] + (Q / ant.qual)

        # add pheromones for all the other edges
        for a in range(len(ant.tour) - 1):
            nodeFrom = ant.tour[a]
            nodeTo = ant.tour[a+1]
            new_pheromones[nodeFrom][nodeTo] = new_pheromones[nodeFrom][nodeTo] + (Q / ant.qual)

    return new_pheromones

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
    best = Solution(problem.get_nodes(), problem.trace_canonical_tour()) # default
    evals = 0
    ants = [] # initialize here because its needed in termination condition

    # distance matrix
    distance_matrix = [[np.inf] * (problem.dimension + 1) for _ in range(problem.dimension + 1)]
    for edge in problem.get_edges():
        distance_matrix[edge[0]][edge[1]] = problem.get_weight(edge[0], edge[1])

    # pheromone matrix
    pheromone_matrix = [ [cfg['initial_pheromone']] * len(distance_matrix)] * len(distance_matrix)
    pheromone_matrix[0] = [-np.inf] * len(distance_matrix) # nodes start with 1, so 0-x edges need to be de-facto removed
    for i in range(len(pheromone_matrix)):
        pheromone_matrix[i][0] = -np.inf

    while not ('evals' in terminate and evals >= terminate['evals'] \
        or 'qualdev' in terminate and best.qual < optimal_quality * (1 + terminate['qualdev']) \
        or 'time' in terminate and time.perf_counter() - starttime > terminate['time'] \
        or 'noimprovement' in terminate and len(set([ant.qual for ant in ants])) == 1): # comparing qualities is easier than comparing tours, so...
    
        # construct ant solutions
        ants = [constructAntSolution(problem, distance_matrix, pheromone_matrix, \
                    cfg['alpha'], cfg['beta']) for _ in range(cfg['antcount'])]
        evals += cfg['antcount']

        # local search...
        if cfg['localsearch']:
            maxevals = np.inf if 'evals' not in terminate else terminate['evals'] - evals
            minqual = -np.inf if 'qualdev' not in terminate else optimal_quality * (1 + terminate['qualdev'])
            for i in range(cfg['antcount']):
                ants[i], newevals = iterimprov_2opt(problem, ants[i], minqual = minqual, maxevals = maxevals, mode = cfg['localsearch'])
                evals += newevals

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
