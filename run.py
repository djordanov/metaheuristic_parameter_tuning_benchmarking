# pylint: disable=no-member

from pathlib import Path
import random
import time
import tsplib95
from metaheuristics import Eval, sa

import rpy2.robjects.packages as rpackages
import numpy as np
import pandas as pd

class Results:

    # static information
    runid: str = None

    # actual run results
    tuning_budget: list = []
    instances: list = []
    qualities: list = []
    evals: list = []
    time: list = []

    def __init__(self, runid):
        self.runid = runid

    def save(self):
        df: pd.DataFrame = pd.DataFrame({
                'tuning_budget': self.tuning_budget,
                'instances': self.instances, 
                'qualities': self.qualities, 
                'evals': self.evals, 
                'time': self.time
        })
        df.to_csv(self.runid, index = False)

def wotuning(runid: str, instancefolder: str, iterations: int):
    # run simulated annealing on all training- and test problems...
    results = Results(runid)
    entries = Path(instancefolder)

    for _ in range(iterations):
        for entry in entries.iterdir():

            if entry.suffix != '.tsp':
                    continue

            problem = tsplib95.load(entry)
            eval = Eval(problem)
            optimaltour = tsplib95.load(entry.with_suffix('.opt.tour').absolute())
            optimal_quality = problem.trace_tours(optimaltour.tours)[0]

            # compute default parameter values
            initial_solution = optimaltour.tours[0]
            random.shuffle(initial_solution)
            distances = [ [problem.get_weight(a, b) for b in range(problem.dimension)] for a in range(problem.dimension) ]
            initial_temperature = np.array(distances).flatten().std()
            repetitions = problem.dimension * (problem.dimension - 1)
            cooling_factor = 0.95

            starttime = time.time()
            result = sa(eval = eval, 
                        initial_solution = initial_solution,
                        initial_temperature = initial_temperature,
                        repetitions = repetitions,
                        cooling_factor = cooling_factor,
                        terminate = {'quality': optimal_quality * 1.05})
            quality_deviation = (result['quality'] - optimal_quality) / optimal_quality
            
            results.tuning_budget.append(0)
            results.instances.append(entry.name)
            results.qualities.append(quality_deviation)
            results.evals.append(result['evals'])
            results.time.append(starttime - result['time'])

    results.save()

def irace(): # doesn't work for some reason
    irace = rpackages.importr('irace')
    parameters = irace.readParameters('./tuning/sa-parameters.txt')
    scenario = irace.readScenario('./tuning/sa-scenario.txt')
    irace.checkIraceScenario(scenario = scenario, parameters = parameters)
    irace_results = irace.irace(scenario = scenario, parameters = parameters)
    print("hi")