import typing
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
        initial_temperature: float, 
        repetitions: int, 
        cooling_factor: float, 
        terminate: dict,
        ftrajectory: Path = None) -> dict:
    starttime = time.perf_counter()

    trajectory = {'qualdev': [], 'evals': [], 'time': []} if ftrajectory != None else None

    # problem: tsplib95.models.StandardProblem
    problem: tsplib95.models.StandardProblem = tsplib95.load(instance)
    optimaltour: tsplib95.models.StandardProblem = tsplib95.load(Path(instance).with_suffix('.opt.tour').absolute())
    optimal_quality: int = problem.trace_tours(optimaltour.tours)[0]

    # setup
    current_solution = list(range(problem.dimension))
    random.shuffle(current_solution)
    current_quality = problem.trace_tours([current_solution])[0]
    evals = 1

    current_temperature = initial_temperature
    best_quality = current_quality
    count_accepted = 0
    count_temperatures_wo_improvement = 0

    while True:
        count_temperatures_wo_improvement += 1

        for _ in range(repetitions):

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
            
            if trajectory != None:
                trajectory['qualdev'].append((best_quality - optimal_quality) / optimal_quality)
                trajectory['evals'].append(evals)
                trajectory['time'].append(time.perf_counter() - starttime)

            # check termination condition
            if 'evals' in terminate and evals >= terminate['evals'] \
                or 'qualdev' in terminate and best_quality < optimal_quality * (1 + terminate['qualdev']) \
                or 'time' in terminate and time.perf_counter() - starttime > terminate['time'] \
                or 'temperature' in terminate and current_temperature > terminate['temperature'] \
                or 'noimprovement' in terminate \
                    and count_temperatures_wo_improvement > terminate['noimprovement']['temperatures'] \
                    and count_accepted / evals < terminate['noimprovement']['accportion']:
                
                if trajectory != None:
                    pd.DataFrame(trajectory).to_csv(ftrajectory.absolute(), mode = 'a', index = None)

                return {'qualdev': (best_quality - optimal_quality) / optimal_quality, 'evals': evals, 'time': time.perf_counter() - starttime}

        # cool down
        current_temperature *= cooling_factor
        current_temperature = max(current_temperature, 0.00001) # avoid rounding errors