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
        df.to_csv('data/' + self.runname, mode = 'a', index = False)

def sa_run_test(instance: Path,  
                terminate: dict = None, 
                config: dict = None, 
                ftrajectory = None):
    problem = tsplib95.load(instance)

    if config == None:
        # compute default parameter values
        distances = [ [problem.get_weight(a, b) for b in range(problem.dimension)] for a in range(problem.dimension) ]
        initial_temperature = np.array(distances).flatten().std()
        repetitions = problem.dimension * (problem.dimension - 1)
        cooling_factor = 0.95
        config = {'initial_temperature': initial_temperature, 'repetitions': repetitions, 'cooling_factor': cooling_factor}
    if terminate == None:
        # default termination condition
        terminate = {'noimprovement': {'temperatures': 5, 'accportion': 0.02}}

    result = sa(instance = instance.absolute(), 
                initial_temperature = config['initial_temperature'],
                repetitions = config['repetitions'],
                cooling_factor = config['cooling_factor'],
                terminate = terminate,
                ftrajectory = ftrajectory)
    return result
    
    

def sa_test_config( name: str, 
                    instancefolder: str, 
                    iterations: int, 
                    budget_tuned: int, 
                    terminate: dict = None, 
                    config: dict = None, 
                    ftrajectory = None):
    # run simulated annealing on all training- and test problems...
    results = Results(name)
    entries = Path(instancefolder)

    for _ in range(iterations):
        for entry in entries.iterdir():

            if entry.suffix != '.tsp':
                    continue
            
            result = sa_run_test(entry, terminate = terminate, config = config, ftrajectory = ftrajectory)
            results.tuning_budget.append(0)
            results.instances.append(entry.name)
            results.qualities.append(result['qualdev'])
            results.evals.append(result['evals'])
            results.time.append(result['time'])

    results.save()

def configsIrace(budget: int) -> dict: # doesn't work for some reason

    robjects.r('library("irace")')
    robjects.r('parameters = readParameters("tuning/sa-parameters.txt")')
    robjects.r('scenario = readScenario(filename = "tuning/sa-scenario.txt")')

    robjects.r('scenario$maxExperiments = ' + str(budget))
    robjects.r('checkIraceScenario(scenario = scenario, parameters = parameters)')
    robjects.r('results = irace(scenario = scenario, parameters = parameters)')

    # get parameter values of best configuration
    colnames = list(robjects.r('names(results[1,])'))
    values = np.array(robjects.r('results[1,]')).flatten()
    elite = {'budget': budget}

    for i in range(len(colnames)):
        if not colnames[i].startswith('.'):
            elite[colnames[i]] = values[i]

    return elite

# generate trajectory data
sa_test_config('trajectoryrun', 'instances/20nodes/test', 1, 0, terminate = None, config = None, ftrajectory = Path('def-Traj-SA-1It'))

# generate 0tuning data
# name = '0tuning_fixed-cost1000'
# terminate = {'evals': 1000}
# sa_test_config(name, 'instances/20nodes/test', iterations = 50, budget_tuned = 0, terminate = terminate)

# generate tuned configs
# elites = pd.DataFrame()
# for budget in range(300, 311, 10): # TODO fix saving results
#     elite = configsIrace(budget) 
#     elites = elites.append(elite, ignore_index = True)
# elites.to_csv('iraceElites.csv', index = False)