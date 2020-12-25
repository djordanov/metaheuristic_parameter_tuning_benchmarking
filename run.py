# pylint: disable=no-member

from pathlib import Path
import random
import time
import tsplib95
from metaheuristics import sa

from rpy2 import robjects
import numpy as np
import pandas as pd

class Results:

    # static information
    runname: str = None

    # actual run results
    tuning_budget: list = []
    instances: list = []
    qualities: list = []
    evals: list = []
    time: list = []

    def __init__(self, runid):
        self.runname = runid

    def save(self):

        df: pd.DataFrame = pd.DataFrame({
                'tuning_budget': self.tuning_budget,
                'instances': self.instances, 
                'qualdev': self.qualities, 
                'evals': self.evals, 
                'time': self.time
        })
        path = Path('data/' + self.runname)
        if path.exists():
            old: pd.DataFrame = pd.read_csv('data/' + self.runname)
            df = old.append(df)
        df.to_csv('data/' + self.runname, index = False)

def sa_test_config(name: str, instancefolder: str, iterations: int, budget_tuned: int, terminate: dict, config: dict = None):
    # run simulated annealing on all training- and test problems...
    results = Results(name)
    entries = Path(instancefolder)

    for _ in range(iterations):
        for entry in entries.iterdir():

            if entry.suffix != '.tsp':
                    continue
            problem = tsplib95.load(entry)

            if config == None:
                # compute default parameter values
                distances = [ [problem.get_weight(a, b) for b in range(problem.dimension)] for a in range(problem.dimension) ]
                initial_temperature = np.array(distances).flatten().std()
                repetitions = problem.dimension * (problem.dimension - 1)
                cooling_factor = 0.95
                config = {'initial_temperature': initial_temperature, 'repetitions': repetitions, 'cooling_factor': cooling_factor}

            starttime = time.time()
            result = sa(instance = entry.absolute(), 
                        initial_temperature = config['initial_temperature'],
                        repetitions = config['repetitions'],
                        cooling_factor = config['cooling_factor'],
                        terminate = terminate)
            
            results.tuning_budget.append(0)
            results.instances.append(entry.name)
            results.qualities.append(result['qualdev'])
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

# generate 0tuning data
name = '0tuning-evals2success_rate-qualdev0.05+noimprov'
terminate = {'qualdev': 0.05, 'noimprovement': {'temperatures': 5, 'accportion': 0.02}}
sa_test_config(name, 'instances/20nodes/test', iterations = 1, budget_tuned = 0, terminate = terminate)