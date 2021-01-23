# pylint: disable=no-member

from pathlib import Path
import math

import numpy as np
import pandas as pd
import tsplib95

from myproject.metaheuristic.sa import sa
from myproject.metaheuristic.aco import aco
from myproject.metaheuristic.ga import ga
import myproject.tuning_wrapper as tuning_wrapper

from collections import namedtuple

DEF_TERM_SA = {'noimprovement': {'temperatures': 5, 'accportion': 0.02}}
DEF_CFG_GA = {'popsize': 200, 'mut_rate': 0.01, 'b': 1.9}
DEF_TERM_GA = {'noimprovement': {'iterations': 5000}, 'evals': 50000}

Result = namedtuple('Result', 'tuning_budget instance quality evals time')

def def_cfg_sa(problem: tsplib95.models.StandardProblem) -> dict:
    distances = [problem.get_weight(edge[0], edge[1]) for edge in problem.get_edges()]
    initial_temperature = -np.array(distances).std() / math.log(0.47)
    repetitions = problem.dimension * (problem.dimension - 1)
    return {'initial_temperature': initial_temperature, 'repetitions': repetitions, 'cooling_factor': 0.95}

def def_cfg_aco(problem: tsplib95.models.StandardProblem) -> dict:
    antcount = problem.dimension
    return {'antcount': antcount, 'alpha': 1, 'beta': 2, 'evaporation': 0.98, 'pbest': 0.05}

def def_term_aco(problem: tsplib95.models.StandardProblem) -> dict:
    return {'evals': 2500 * problem.dimension}

def mhrun(instance: Path,  
            algorithm: str,
            terminate: dict = None, 
            config: dict = None, 
            fname_convdata = None):

    problem = tsplib95.load(instance)

    if algorithm == 'SA':
        # use default configuration and termination values if none are given
        cfg = def_cfg_sa(problem) if config == None else config
        terminate = DEF_TERM_SA if terminate == None else terminate
        return sa(instance = instance, cfg = cfg, terminate = terminate, fname_convdata = fname_convdata)

    if algorithm == 'ACO':
        # use default configuration and termination values if none are given
        cfg = def_cfg_aco(problem) if config == None else config
        terminate = def_term_aco(problem) if terminate == None else terminate
        return aco(instance = instance, cfg = cfg, terminate = terminate, fname_convdata = fname_convdata)

    if algorithm == 'GA':
        # use default configuration and termination values if none are given
        cfg = DEF_CFG_GA if config == None else config
        terminate = DEF_TERM_GA if terminate == None else terminate
        return ga(instance = instance, cfg = cfg, terminate = terminate, fname_convdata = fname_convdata)

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
            
            result = mhrun(entry, algorithm = algorithm, terminate = terminate, config = config, fname_convdata = fconvergence)
            results.append(Result(budget_tuned, entry.name, result['qualdev'], result['evals'], result['time']))

    fpath = Path('myproject/data/' + fname)
    df: pd.DataFrame = pd.DataFrame(results)
    mode = 'a' if fpath.exists() else 'w+'
    df.to_csv(fpath.absolute(), mode = mode, index = False)

# test metaheuristics
import random
random.seed(1)
# result = mhrun(Path('myproject/instances/20nodes/rnd0_20.tsp'), algorithm = 'SA')
# print(result)
# result = mhrun(Path('myproject/instances/20nodes/rnd0_20.tsp'), algorithm = 'ACO')
# print(result)
# result = mhrun(Path('myproject/instances/20nodes/rnd0_20.tsp'), algorithm = 'GA')
# print(result)
# mhruns('test', instancefolder = 'myproject/instances/20nodes/test', algorithm = 'SA', iterations = 1, budget_tuned = 0, terminate = {'evals': 100})

# tune
# budget = 300
# train_instances_dir = 'myproject/instances/20nodes'
# train_instances_file = 'myproject/instances/20nodes/trainInstancesFile'
# terminate = DEF_TERM_SA 
# optimize = 'qualdev'
# name = 'qd' + str(0.05) + 't' + str(budget) + 'o' + str(optimize)
# elite = tuning_wrapper.irace(budget = budget, algorithm = 'SA', terminate = terminate, optimize = optimize, train_instances_dir = train_instances_dir)
# os.rename(r'irace.Rdata', r'myproject/data/irace.Rdata.' + name)
# print(elite)
# tuning_wrapper.smac(budget = budget, algorithm = 'SA', terminate = terminate, optimize = optimize, train_instances_dir = train_instances_dir)

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