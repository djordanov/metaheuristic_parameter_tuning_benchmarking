import typing

import math
import random
import time
import numpy as np
import pandas as pd

import tsplib95

# Quality evaluation wrapper for counting quality evaluations
class Eval:
    evals_count: int = 0
    problem: tsplib95.models.StandardProblem = None

    def __init__(self, problem):
        self.problem = problem

    def eval(self, tour: list) -> int:
        self.evals_count += 1
        return self.problem.trace_tours([tour])[0]

def random_n2opt(solution: list) -> list:
    combinations = [(a, b) for a in range(len(solution)) for b in range(a, len(solution)) if abs(a-b) > 1]
    permutation = random.choice(combinations)
    return n2opt(solution, permutation[0], permutation[1])

def n2opt(solution: list, i: int, j: int) -> list:
     return solution[:i+1] + solution[i+1:j+1][::-1] + solution[j+1:]

def sa( eval: Eval, \
        initial_solution: list, 
        initial_temperature: int, 
        repetitions: int, 
        cooling_factor: float, 
        terminate: dict) -> dict:

    # setup
    current_solution = initial_solution
    current_quality = eval.eval(current_solution)
    current_temperature = initial_temperature
    best_quality = current_quality

    while True:
        for _ in range(repetitions):
            neighbor_solution = random_n2opt(current_solution)
            neighbor_quality = eval.eval(neighbor_solution)

            # downhill moves
            if neighbor_quality < best_quality:
                best_quality = neighbor_quality
            
            if neighbor_quality <= current_quality:
                current_solution = neighbor_solution
                current_quality = neighbor_quality
            
            else: # uphill moves
                probability = math.e**((current_quality - neighbor_quality)/current_temperature)
                if random.random() < probability:
                    current_solution = neighbor_solution
                    current_quality = neighbor_quality
            
            # check termination condition
            if 'evals' in terminate and eval.evals_count >= terminate['evals']:
                return {'quality': best_quality, 'countevals': eval.evals_count, 'timestamp': time.time()}
            if 'quality' in terminate and eval.best_quality <= terminate['quality']:
                return {'quality': best_quality, 'countevals': eval.evals_count, 'timestamp': time.time()}
            if 'time' in terminate and eval.time.time() >= terminate['time']:
                return {'quality': best_quality, 'countevals': eval.evals_count, 'timestamp': time.time()}
            if 'temperature' in terminate and current_temperature >= terminate['temperature']:
                return {'quality': best_quality, 'countevals': eval.evals_count, 'timestamp': time.time()}

        # cool down
        current_temperature *= cooling_factor