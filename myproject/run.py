# pylint: disable=no-member

import os
from pathlib import Path
import random
import time

from rpy2 import robjects
import numpy as np
import pandas as pd
import tsplib95

from myproject.metaheuristic.sa import sa
import myproject.wrapper.irace_sa as irace_sa

class Results:

    # actual run results
    tuning_budget: list = []
    instances: list = []
    qualities: list = []
    evals: list = []
    time: list = []

    def save(self, fpath: Path):
        df: pd.DataFrame = pd.DataFrame({
                'tuning_budget': self.tuning_budget,
                'instances': self.instances, 
                'qualdev': self.qualities, 
                'evals': self.evals, 
                'time': self.time
        })
        mode = 'a' if fpath.exists() else 'w+'
        df.to_csv(fpath.absolute(), mode = mode, index = False)

def sa_run_test(instance: Path,  
                terminate: dict = None, 
                config: dict = None, 
                fconvergence = None):
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
                fconvergence = fconvergence)
    return result
    
    

def sa_test_config( fname: str, 
                    instancefolder: str, 
                    iterations: int, 
                    budget_tuned: int, 
                    terminate: dict = None, 
                    config: dict = None, 
                    fconvergence = None):
    # run simulated annealing on all training- and test problems...
    results = Results()
    entries = Path(instancefolder)

    for _ in range(iterations):
        for entry in entries.iterdir():

            if entry.suffix != '.tsp':
                    continue
            
            result = sa_run_test(entry, terminate = terminate, config = config, fconvergence = fconvergence)
            results.tuning_budget.append(0)
            results.instances.append(entry.name)
            results.qualities.append(result['qualdev'])
            results.evals.append(result['evals'])
            results.time.append(result['time'])

    results.save(Path('myproject/data/' + fname))

def iraceTune(budget: int, terminate: dict, optimize: str) -> dict:

    robjects.r('library("irace")')
    robjects.r('parameters = readParameters("myproject/tuning/sa-parameters.txt")')
    robjects.r('scenario = readScenario(filename = "myproject/tuning/sa-scenario.txt")')

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

# default convergence
sa_test_config('cfgdefaultt0', 'myproject/instances/20nodes/test', iterations = 5, budget_tuned = 0, 
                terminate = None, config = None, fconvergence = Path('myproject/data/saconv-cfgdefaultt0'))

# tune
# evals = 2000
# budget = 10000
# optimize = 'qualdev'
# name = 'e' + str(evals) + 't' + str(budget) + 'o' + str(optimize)
# elite = iraceTune(budget = budget, terminate = {'evals': evals}, optimize = optimize)
# os.rename(r'irace.Rdata', r'data/irace.Rdata.' + name)
# print(elite)

# run tuned cfg
# config, terminate, optimize = irace_sa.dict2params(elite) 
# sa_test_config(fname = name, 
#                instancefolder = 'myproject/instances/20nodes/test', iterations = 50, 
#                budget_tuned = budget, terminate = None, config = config,
#                fconvergence = Path('myproject/data/test-' + name)) 

# generate tuned configs
# elites = pd.DataFrame()
# for budget in range(300, 311, 10): # TODO fix saving results
#     elite = iraceTune(budget) 
#     elites = elites.append(elite, ignore_index = True)
# elites.to_csv('iraceElites.csv', index = False)

# saved irace configurations to usable parameter configurations and termination criteria
# df: pd.DataFrame = pd.read_csv('data/iraceElites.csv')
# for i in range(0, len(df)):
#     row: pd.Series = df.iloc[i]     # select first row
#     budget = row['budget']
#     row = row.drop('budget')

#     # build parameter - value list analogous to the one coming from irace
#     candparams = []
#     for i in range(0, len(row)):
#         candparams.append(row.index[i])
#         candparams.append(row.values[i])

#     cfg, terminate, optimize = wrapper_irace_sa.create_config_terminate_optimize(candparams)
#     print((cfg, terminate, optimize))