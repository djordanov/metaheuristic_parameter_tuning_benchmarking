# pylint: disable=no-member

import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import tsplib95

from myproject.metaheuristic.sa import sa
from myproject.metaheuristic.aco import aco
import myproject.wrapper_irace as wrapper_irace

from collections import namedtuple

DEF_CFG_ACO = {'initial_pheromone': 5, 'antcount': 5, 'alpha': 0.5, 'beta': 0.5, 'Q': 1000, 'evaporation': 0.5, 'localsearch': 'first'}
DEF_TERM_SA = {'noimprovement': {'temperatures': 5, 'accportion': 0.02}}
DEF_TERM_ACO = {'evals': 5000}

Result = namedtuple('Result', 'tuning_budget instance quality evals time')

def def_cfg_sa(problem: tsplib95.models.StandardProblem) -> dict:
    distances = [problem.get_weight(edge[0], edge[1]) for edge in problem.get_edges()]
    initial_temperature = np.array(distances).std() # TODO is that correct?
    repetitions = problem.dimension * (problem.dimension - 1)
    cooling_factor = 0.95
    return {'initial_temperature': initial_temperature, 'repetitions': repetitions, 'cooling_factor': cooling_factor}

def mhrun(instance: Path,  
            algorithm: str,
            terminate: dict = None, 
            config: dict = None, 
            fconvergence = None):

    problem = tsplib95.load(instance)

    if algorithm == 'SA':
        # use default configuration and termination values if none are given
        cfg = def_cfg_sa(problem) if config == None else config
        terminate = DEF_TERM_SA if terminate == None else terminate
        return sa(instance = instance.absolute(), cfg = cfg, terminate = terminate, fconvergence = fconvergence)

    if algorithm == 'ACO':
        # use default configuration and termination values if none are given
        cfg = DEF_CFG_ACO if config == None else config
        terminate = DEF_TERM_ACO if terminate == None else terminate
        return aco(instance = instance.absolute(), cfg = cfg, terminate = terminate, fconvergence = fconvergence)

    else: 
        print('Error: Algorithm ' + '"' + algorithm + '" not found!')
    
def mhruns( fname: str, 
                iterations: int, 
                budget_tuned: int, 
                instancefolder: str, 
                algorithm: str,
                terminate: dict = None, 
                config: dict = None, 
                fconvergence = None):
    # run metaheuristic on all problems in folder...
    # results = Results()
    results = []
    entries = Path(instancefolder)

    for _ in range(iterations):
        for entry in entries.iterdir():

            if entry.suffix != '.tsp':
                    continue
            
            result = mhrun(entry, algorithm = algorithm, terminate = terminate, config = config, fconvergence = fconvergence)
            results.append(Result(budget_tuned, entry.name, result['qualdev'], result['evals'], result['time']))

    fpath = Path('myproject/data/' + fname)
    df: pd.DataFrame = pd.DataFrame(results)
    mode = 'a' if fpath.exists() else 'w+'
    df.to_csv(fpath.absolute(), mode = mode, index = False)

# test metaheuristics
random.seed(1)
result = mhrun(Path('myproject/instances/20nodes/rnd0_20.tsp'), algorithm = 'SA', terminate = {'evals': 100})
print(result)
mhruns('test', instancefolder = 'myproject/instances/20nodes/test', algorithm = 'SA', iterations = 1, budget_tuned = 0, terminate = {'evals': 100})

# default convergence
# sa_test_config('cfgdefaultt0', 'myproject/instances/20nodes/test', iterations = 5, budget_tuned = 0, 
#             terminate = None, config = None, fconvergence = Path('myproject/data/saconv-cfgdefaultt0'))

# tune
# budget = 300
# train_instances_dir = 'myproject/instances/20nodes'
# train_instances_file = 'myproject/instances/20nodes/trainInstancesFile'
# terminate = {'qualdev': 0.05}
# optimize = 'evals'
# name = 'qd' + str(0.05) + 't' + str(budget) + 'o' + str(optimize)
# elite = wrapper_irace.tune(budget = budget, algorithm = 'ACO', terminate = terminate, optimize = optimize,
#                             train_instances_dir = train_instances_dir, train_instances_file = train_instances_file)
# os.rename(r'irace.Rdata', r'myproject/data/irace.Rdata.' + name)
# print(elite)

# run tuned cfg
# config, terminate, optimize = wrapper_irace.dict2params(elite) 
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

#     cfg, terminate, optimize = wrapper_wrapper_irace.create_config_terminate_optimize(candparams)
#     print((cfg, terminate, optimize))