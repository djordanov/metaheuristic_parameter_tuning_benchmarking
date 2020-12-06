import typing

import math
import random
import numpy as np
import pandas as pd

import tsplib95

def generate_random_euclidean__instances(count: int, cnodes: int, squaresize: int):
    # use tsplib format for easy use of tsplib95 imports

    print("Generating instances...")
    for i in range(count):

        # create file and compulsories
        name = "rnd" + str(i) + "_" + str(cnodes) + ".tsp"
        f = open("./instances/" + name, "w+")
        f.write("NAME : " + name + "\n")
        f.write("TYPE : TSP" + "\n")
        f.write("DIMENSION : " + str(cnodes) + "\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D" + "\n")
        f.write("NODE_COORD_SECTION" + "\n")

        # generate coordinates
        for inode in range(cnodes):
            line = str(inode) + " " + str(random.randint(0, squaresize)) + " " + str(random.randint(0, squaresize))
            f.write(line + "\n")

        f.close()

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

def sa(eval: Eval, initial_solution: list, initial_temperature, repetitions, cooling_factor, max_evals: int) -> tuple:
    print("Starting Simulated Annealing")
    current_solution = initial_solution
    current_quality = eval.eval(current_solution)
    current_temperature = initial_temperature

    while eval.evals_count < max_evals:
        for _ in range(repetitions):
            neighbor_solution = random_n2opt(current_solution)
            neighbor_quality = eval.eval(neighbor_solution)
            
            if neighbor_quality <= current_quality:
                current_solution = neighbor_solution
                current_quality = neighbor_quality
                continue
            
            probability = math.e**((current_quality - neighbor_quality)/current_temperature)
            if random.random() < probability:
                current_solution = neighbor_solution
                current_quality = neighbor_quality
                continue
        
        current_temperature *= cooling_factor

    return (current_solution, current_quality)