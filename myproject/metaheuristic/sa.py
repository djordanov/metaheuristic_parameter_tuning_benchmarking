import logging

from pathlib import Path
import math
import random
import time
import numpy as np
import pandas as pd

import tsplib95

# logging.basicConfig(level=logging.DEBUG) # logging

def random_n2opt(solution: list) -> list:
    combinations = [(a, b) for a in range(len(solution)) for b in range(a, len(solution)) if abs(a-b) > 1]
    permutation = random.choice(combinations)
    return n2opt(solution, permutation[0], permutation[1])

def n2opt(solution: list, i: int, j: int) -> list:
     return solution[:i+1] + solution[i+1:j+1][::-1] + solution[j+1:]

def accept(current_quality: float, neighbor_quality: float, current_temperature: float) -> bool:
    if neighbor_quality <= current_quality:
        return True
            
    probability = math.e**((current_quality - neighbor_quality)/current_temperature)
    if random.random() < probability:
        return True

    return False

def sa( instance: str, 
        cfg: dict, 
        terminate: dict,
        fconvergence: Path = None) -> dict:
    starttime = time.perf_counter()

    convergence = {'qualdev': [], 'evals': [], 'time': []} if fconvergence != None else None

    # problem: tsplib95.models.StandardProblem
    problem: tsplib95.models.StandardProblem = tsplib95.load(instance)
    optimaltour: tsplib95.models.StandardProblem = tsplib95.load(Path(instance).with_suffix('.opt.tour').absolute())
    optimal_quality: int = problem.trace_tours(optimaltour.tours)[0]

    # setup
    current_solution = list(range(problem.dimension))
    random.shuffle(current_solution)
    current_quality = problem.trace_tours([current_solution])[0]
    evals = 1

    current_temperature = cfg['initial_temperature']
    best_quality = current_quality
    count_accepted = 0
    count_temperatures_wo_improvement = 0

    while not ('evals' in terminate and evals >= terminate['evals'] \
                or 'qualdev' in terminate and best_quality < optimal_quality * (1 + terminate['qualdev']) \
                or 'time' in terminate and time.perf_counter() - starttime > terminate['time'] \
                or 'noimprovement' in terminate \
                    and count_temperatures_wo_improvement > terminate['noimprovement']['temperatures'] \
                    and count_accepted / evals < terminate['noimprovement']['accportion']):
        
        count_temperatures_wo_improvement += 1
        for _ in range(cfg['repetitions']):

            # get neighbor
            neighbor_solution = random_n2opt(current_solution)
            neighbor_quality = problem.trace_tours([neighbor_solution])[0]
            evals += 1

            # accept neighbor if 
            if accept(current_quality, neighbor_quality, current_temperature):
                current_solution = neighbor_solution
                current_quality = neighbor_quality
                count_accepted += 1

                if current_quality < best_quality:
                    best_quality = current_quality
                    count_temperatures_wo_improvement = 0    
            
            if convergence != None:
                convergence['qualdev'].append((best_quality - optimal_quality) / optimal_quality)
                convergence['evals'].append(evals)
                convergence['time'].append(time.perf_counter() - starttime)

        # cool down
        current_temperature *= cfg['cooling_factor']
        current_temperature = max(current_temperature, 0.00001) # avoid rounding errors
    if convergence != None:
        print_headers: bool = False if fconvergence.exists() else True
        pd.DataFrame(convergence).to_csv(fconvergence.absolute(),  index = None, mode = 'a', header = print_headers)

    return {'qualdev': (best_quality - optimal_quality) / optimal_quality, 'evals': evals, 'time': time.perf_counter() - starttime}