# pylint: disable=no-member

from pathlib import Path
import random
import time
import tsplib95
from metaheuristics import sa
import wrapper_irace_sa

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

def iraceTune(budget: int, terminate: dict, optimize: str) -> dict:

    robjects.r('library("irace")')
    robjects.r('parameters = readParameters("tuning/sa-parameters.txt")')
    robjects.r('scenario = readScenario(filename = "tuning/sa-scenario.txt")')

    # set tuning budget
    robjects.r('scenario$maxExperiments = ' + str(budget))
    
    # set optimize parameter
    robjects.r('parameters$domain$optimize = \"' + optimize + '\"')

    # set termination condition
    for key in terminate:
        if key == 'noimprovement':
            robjects.r('parameters$domain$term_noimprovement = \"True\"')
            robjects.r('parameters$domain$term_noimpr_temp_val = ' + str(terminate['noimprovement']['temperatures']))
            robjects.r('parameters$domain$term_noimpr_accp_val = ' + str(terminate['noimprovement']['accportion']))
        else:
            boolparam = 'term_' + key
            valparam = boolparam + '_val'
            robjects.r('parameters$domain$' + boolparam + ' = \"True\"')
            robjects.r('parameters$domain$' + valparam + ' = ' + str(terminate[key]))

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

# metaheuristic default trajectory data
# sa_test_config('trajectoryrun', 'instances/20nodes/test', 5, 0, terminate = None, config = None, ftrajectory = Path('def-Traj-SA (2)'))

# metaheuristic tuned for different budget trajectory data
# tunebudget = 10000
# cfg500 = iraceTune(tunebudget)

# generate irace default trajectory data

# generate 0tuning data
# name = '0tuning_fixed-cost1000'
# terminate = {'evals': 1000}
# sa_test_config(name, 'instances/20nodes/test', iterations = 50, budget_tuned = 0, terminate = terminate)

# generate tuned configs
# elites = pd.DataFrame()
# for budget in range(300, 311, 10): # TODO fix saving results
#     elite = iraceTune(budget) 
#     elites = elites.append(elite, ignore_index = True)
# elites.to_csv('iraceElites.csv', index = False)

# saved irace configurations to usable parameter configurations and termination criteria
df: pd.DataFrame = pd.read_csv('data/iraceElites.csv')
for i in range(0, len(df)):
    row: pd.Series = df.iloc[i]     # select first row
    budget = row['budget']
    row = row.drop('budget')

    # build parameter - value list analogous to the one coming from irace
    candparams = []
    for i in range(0, len(row)):
        candparams.append(row.index[i])
        candparams.append(row.values[i])

    cfg, terminate, optimize = wrapper_irace_sa.create_config_terminate_optimize(candparams)
    print((cfg, terminate, optimize))