from pathlib import Path
import random
import time
import pandas as pd

import tsplib95

def constructAntSolution(distance_matrix: list, pheromone_matrix: list, alpha: float, beta: float) -> list:

    dimension = len(distance_matrix)

    # start tour
    tour = [random.randint(0, dimension-1)]

    # finish tours
    while len(tour) < dimension:

        # helper variables
        current_node = tour[-1]
        edges_distances = distance_matrix[current_node]
        edges_pheromones = pheromone_matrix[current_node]
        possible_moves = set(range(dimension)) - set(tour)
        
        # move probabilities
        unnormed_probabilities = []
        for node in range(dimension):
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
        for node in range(dimension):
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
        initial_pheromone: float, 
        antcount: int, 
        alpha: float,
        beta: float,
        Q: float,
        evaporation: float,
        terminate: dict, 
        fconvergence: Path = None) -> dict:
    starttime = time.perf_counter()
    convergence = {'qualdev': [], 'evals': [], 'time': []} if fconvergence != None else None

    # problem: tsplib95.models.StandardProblem
    problem: tsplib95.models.StandardProblem = tsplib95.load(instance)
    optimaltour: tsplib95.models.StandardProblem = tsplib95.load(Path(instance).with_suffix('.opt.tour').absolute())
    optimal_quality: int = problem.trace_tours(optimaltour.tours)[0]

    # setup
    distance_matrix = [ [problem.get_weight(a, b) for b in range(problem.dimension)] for a in range(problem.dimension) ]    
    pheromone_matrix = [ [initial_pheromone] * problem.dimension] * problem.dimension
    best_quality = None
    evals = 0

    while not ('evals' in terminate and evals >= terminate['evals'] \
        or 'qualdev' in terminate and best_quality < optimal_quality * (1 + terminate['qualdev']) \
        or 'time' in terminate and time.perf_counter() - starttime > terminate['time']):
    
        # construct ant solutions...
        antSols = [constructAntSolution(distance_matrix, pheromone_matrix, alpha, beta) for _ in range(antcount)]
        antQuals = problem.trace_tours(antSols)
        evals += antcount

        # local search...
        # lets skip that for now...

        # set best (could be done more speed friendly)
        best_new_quality = min(antQuals)
        if best_quality == None or best_new_quality < best_quality:
            best_quality = best_new_quality

        # update pheromones...
        pheromone_matrix = updatePheromones(pheromone_matrix, evaporation, Q, antSols, antQuals)

        # save state if convergence is looked for 
        if convergence != None:
            convergence['qualdev'].append((best_quality - optimal_quality) / optimal_quality)
            convergence['evals'].append(evals)
            convergence['time'].append(time.perf_counter() - starttime)
        
    if convergence != None:
        print_headers: bool = False if fconvergence.exists() else True
        pd.DataFrame(convergence).to_csv(fconvergence.absolute(),  index = None, mode = 'a', header = print_headers)

    return {'qualdev': (best_quality - optimal_quality) / optimal_quality, 'evals': evals, 'time': time.perf_counter() - starttime}
