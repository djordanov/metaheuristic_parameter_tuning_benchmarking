# pylint: disable=no-member

from pathlib import Path
import random
import time
import tsplib95
from metaheuristics import Eval, sa

from rpy2 import robjects
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

def runirace(budgets: list): # doesn't work for some reason
    elites = pd.DataFrame({})

    robjects.r('library("irace")')
    robjects.r('parameters = readParameters("tuning/sa-parameters.txt")')
    robjects.r('scenario = readScenario(filename = "tuning/sa-scenario.txt")')

    for budget in budgets:
        robjects.r('scenario$maxExperiments = ' + str(budget))
        robjects.r('checkIraceScenario(scenario = scenario, parameters = parameters)')
        robjects.r('results = irace(scenario = scenario, parameters = parameters)')

        # get parameter values of best configuration
        colnames = list(robjects.r('names(results[1,])'))
        values = np.array(robjects.r('results[1,]')).flatten()
        best = {'budget': budget}

        for i in range(len(colnames)):
            if not colnames[i].startswith('.'):
                best[colnames[i]] = values[i]
        elites = elites.append(best, ignore_index = True)

    elites.to_csv('data/runirace_elites.csv')