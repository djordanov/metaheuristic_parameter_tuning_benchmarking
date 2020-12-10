import typing

import random
import numpy as np
import pandas as pd

import time
import tsplib95

import metaheuristics

# test run simulated annealing on first instance
runname: str = str(time.time())

problem: tsplib95.models.StandardProblem = tsplib95.load('./instances/rnd0_20.tsp')

eval: metaheuristics.Eval = metaheuristics.Eval(problem)
initial_solution = list(range(problem.dimension))
random.shuffle(initial_solution)
initial_temperature: float = 0.03 * problem.trace_canonical_tour() # should be changed to something better and more robust
repetitions: int = problem.dimension * (problem.dimension - 1) 
cooling_factor: float = 0.95 # by default between 0.8 and 0.95
max_evals: int = 450 * repetitions # I have no idea

result = metaheuristics.sa(eval, initial_solution, initial_temperature, repetitions, cooling_factor, max_evals)
