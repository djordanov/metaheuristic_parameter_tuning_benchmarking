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

def sa( instance: str, 
        initial_temperature: float, 
        repetitions: int, 
        cooling_factor: float, 
        terminate: dict) -> dict:

    logging.debug('--- Starting simulated annealing ---')
    logging.debug('Initial Temperature: ' + str(initial_temperature))
    logging.debug('Repetitions: ' + str(repetitions))
    logging.debug('Cooling Factor: ' + str(cooling_factor))

    # problem: tsplib95.models.StandardProblem
    problem: tsplib95.models.StandardProblem = tsplib95.load(instance)
    optimaltour: tsplib95.models.StandardProblem = tsplib95.load(Path(instance).with_suffix('.opt.tour').absolute())
    optimal_quality: int = problem.trace_tours(optimaltour.tours)[0]

    # setup
    starttime = time.time()
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
            neighbor_solution = random_n2opt(current_solution)
            neighbor_quality = problem.trace_tours([neighbor_solution])[0]
            evals += 1

            logging.debug('------')
            logging.debug('Current Solution: ' + str(current_solution))
            logging.debug('Current Quality: ' + str(current_quality))
            logging.debug('Current Temperature: ' + str(current_temperature))
            logging.debug('------')

            # downhill moves
            if neighbor_quality < best_quality:
                best_quality = neighbor_quality
                count_temperatures_wo_improvement = 0
                count_accepted += 1
            
            if neighbor_quality <= current_quality:
                current_solution = neighbor_solution
                current_quality = neighbor_quality
                count_accepted += 1
            
            else: # uphill moves
                probability = math.e**((current_quality - neighbor_quality)/current_temperature)
                if random.random() < probability:
                    current_solution = neighbor_solution
                    current_quality = neighbor_quality
                    count_accepted += 1
            
            # check termination condition
            if 'evals' in terminate and evals >= terminate['evals'] \
                or 'quality' in terminate and best_quality < optimal_quality * (1 + terminate['quality']) \
                or 'time' in terminate and time.time() - starttime > terminate['time'] \
                or 'temperature' in terminate and current_temperature > terminate['temperature'] \
                or 'noimprovement' in terminate \
                    and count_temperatures_wo_improvement > terminate['noimprovement']['temperatures'] \
                    and count_accepted / evals < terminate['noimprovement']['accportion']:
                return {'qualdev': (best_quality - optimal_quality) / optimal_quality, 'evals': evals, 'time': time.time() - starttime}

        # cool down
        current_temperature *= cooling_factor
        current_temperature = max(current_temperature, 0.00001) # avoid rounding errors