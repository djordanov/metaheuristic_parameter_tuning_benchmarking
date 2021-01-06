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
                cfg = config,
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

# test sa
# sa_test_config('cfgdefaultt0', 'myproject/instances/20nodes/test', iterations = 5, budget_tuned = 0, 
#                 terminate = {'evals': 100})

# test aco
# terminate = {'evals': 10000}
# aco('myproject/instances/20nodes/rnd0_20.tsp', 1, 5, 0.5, 0.5, 50, 0.8, terminate, Path('acoconvergence'))

# default convergence
# sa_test_config('cfgdefaultt0', 'myproject/instances/20nodes/test', iterations = 5, budget_tuned = 0, 
#             terminate = None, config = None, fconvergence = Path('myproject/data/saconv-cfgdefaultt0'))

# tune
evals = 500
budget = 300
train_instances_dir = 'myproject/instances/20nodes'
train_instances_file = 'myproject/instances/20nodes/trainInstancesFile'
optimize = 'qualdev'
name = 'e' + str(evals) + 't' + str(budget) + 'o' + str(optimize)
elite = wrapper_irace.tune(budget = budget, algorithm = 'ACO', terminate = {'evals': evals}, optimize = optimize,
                            train_instances_dir = train_instances_dir, train_instances_file = train_instances_file)
os.rename(r'irace.Rdata', r'myproject/data/irace.Rdata.' + name)
print(elite)

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