from pathlib import Path
import random
import time
import numpy as np
import pandas as pd

import tsplib95
from myproject.metaheuristic.commons import iterimprov_2opt

def constructAntSolution(problem: tsplib95.models.StandardProblem,
            distance_matrix: list, pheromone_matrix: list, alpha: float, beta: float) -> list:

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
                probability = (1/edges_distances[node])**alpha * edges_pheromones[node]**beta
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

    return tour

def updatePheromones(pheromone_matrix: list, evaporation: float, Q: float, antSols: list, antQuals: list) -> list:

    # pheromone evaporation
    new_pheromones = [[cell * evaporation for cell in row] for row in pheromone_matrix]

    # pheromone adding
    for i in range(len(antSols)):
        solution = antSols[i]
        quality = antQuals[i]

        # add pheromones for last/ first edge
        firstNode = solution[0]
        lastNode = solution[-1]
        new_pheromones[firstNode][lastNode] = new_pheromones[firstNode][lastNode] + (Q / quality)

        # add pheromones for all the other edges
        for a in range(len(solution) - 1):
            nodeFrom = solution[a]
            nodeTo = solution[a+1]
            new_pheromones[nodeFrom][nodeTo] = new_pheromones[nodeFrom][nodeTo] + (Q / quality)

    return new_pheromones

def aco(instance: str, 
        cfg: dict,
        terminate: dict, 
        fconvergence: Path = None) -> dict:
    starttime = time.perf_counter()
    convergence = {'qualdev': [], 'evals': [], 'time': []} if fconvergence != None else None

    # problem: tsplib95.models.StandardProblem
    problem: tsplib95.models.StandardProblem = tsplib95.load(instance)
    optimaltour: tsplib95.models.StandardProblem = tsplib95.load(Path(instance).with_suffix('.opt.tour').absolute())
    optimal_quality: int = problem.trace_tours(optimaltour.tours)[0]

    # setup...
    best_quality = np.inf
    evals = 0

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
        or 'qualdev' in terminate and best_quality < optimal_quality * (1 + terminate['qualdev']) \
        or 'time' in terminate and time.perf_counter() - starttime > terminate['time']):
    
        # construct ant solutions...
        antSols = [constructAntSolution(problem, distance_matrix, pheromone_matrix, \
                    cfg['alpha'], cfg['beta']) for _ in range(cfg['antcount'])]
        antQuals = problem.trace_tours(antSols)
        evals += cfg['antcount']

        # local search...
        if cfg['localsearch'] != None:
            for i in range(cfg['antcount']):
                maxevals = np.inf if 'evals' not in terminate else terminate['evals'] - evals
                minqual = -np.inf if 'qualdev' not in terminate else optimal_quality * (1 + terminate['qualdev'])
                antSols[i], antQuals[i], newevals = iterimprov_2opt(problem, antSols[i], \
                                                                    antQuals[i], cfg['localsearch'], \
                                                                    minqual = minqual, maxevals = maxevals)
                evals += newevals

        # update best reached quality (could be done a little bit more speed friendly)
        best_new_quality = min(antQuals)
        if best_quality == None or best_new_quality < best_quality:
            best_quality = best_new_quality

        # update pheromones...
        pheromone_matrix = updatePheromones(pheromone_matrix, cfg['evaporation'], cfg['Q'], antSols, antQuals)

        # save state if convergence is looked for 
        if convergence != None:
            convergence['qualdev'].append((best_quality - optimal_quality) / optimal_quality)
            convergence['evals'].append(evals)
            convergence['time'].append(time.perf_counter() - starttime)
        
    if convergence != None:
        print_headers: bool = False if fconvergence.exists() else True
        pd.DataFrame(convergence).to_csv(fconvergence.absolute(),  index = None, mode = 'a', header = print_headers)

    return {'qualdev': (best_quality - optimal_quality) / optimal_quality, 'evals': evals, 'time': time.perf_counter() - starttime}
