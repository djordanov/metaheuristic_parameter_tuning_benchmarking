# pylint: disable=no-member
import sys
from pathlib import Path
import random
import math
import datetime

import numpy as np
import pandas as pd
import tsplib95

from myproject.metaheuristic.sa import sa
from myproject.metaheuristic.aco import aco
from myproject.metaheuristic.ga import ga
import myproject.tuning_wrapper as tuning_wrapper

from collections import namedtuple

BASE_TERM = {'qualdev': 0, 'evals': 100000}
DEF_CFG_SA_50N = {'initial_temperature': (-2 * 253.83) / math.log(0.47), 'repetitions': 2450, 'cooling_factor': 0.95}
DEF_CFG_GA = {'popsize': 200, 'mut_rate': 0.01, 'rank_weight': 1.9}
DEF_CFG_ACO_50N = {'antcount': 50, 'alpha': 1, 'beta': 2, 'evaporation': 0.98, 'pbest': 0.05}

CFG_SA_TSMAC_1000E_OQD = {'initial_temperature': 26.556465295873004, 'repetitions': 9597, 'cooling_factor': 0.9012752030874001} # TODO REDO
CFG_SA_TIRACE_1000E_OQD = {'initial_temperature': 27.2484, 'repetitions': 8026, 'cooling_factor': 0.1716}

CFG_GA_TIRACE_1000E_OQD = {'popsize': 1, 'mut_rate': 0.851, 'rank_weight': 0.2627}
CFG_GA_TIRACE_10000E_OQD = {'popsize': 1, 'mut_rate': 0.6601, 'rank_weight': 16.519}
CFG_GA_TSMAC_1000E_OQD = {'popsize': 1, 'mut_rate': 0.9354300849532875, 'rank_weight': 4.2800906310470275}
CFG_GA_TSMAC_10000E_OQD = {'popsize': 1, 'mut_rate': 0.9938586259040849, 'rank_weight': 8.208254826424472}

CFG_ACO_TIRACE_1000E_OQD = {'antcount': 439, 'alpha': 3.543, 'beta': 14.8282, 'evaporation': 0.4428, 'pbest': 84.5142}
CFG_ACO_TIRACE_10000E_OQD = {'antcount': 132, 'alpha': 0.6219, 'beta': 5.4251, 'evaporation': 0.7806, 'pbest': 62.2747}
CFG_ACO_TSMAC_1000E_OQD = {'antcount': 67, 'alpha': 0.9553886797155056, 'beta': 13.368063653676876, \
                           'evaporation': 0.058241608029438105, 'pbest': 0.781074288363884}
CFG_ACO_TSMAC_10000E_OQD = {'antcount': 345, 'alpha': 3.2050014761743935, 'beta': 17.686316543093692, \
                            'evaporation': 0.9263612615269491, 'pbest': 0.11156702049905944 }

Result = namedtuple('Result', 'tuning_budget instance quality evals time')

def calc_mean_std(instancefolder: str):
    entries = Path(instancefolder)
    stds = []
    for entry in entries.iterdir():

        if entry.suffix != '.tsp':
                continue

        problem = tsplib95.load(entry.absolute())
        distances = [problem.get_weight(edge[0], edge[1]) for edge in problem.get_edges()]
        stds.append(np.array(distances).std())
    return np.array(stds).mean()

def def_cfg_sa(problem: tsplib95.models.StandardProblem) -> dict:
    distances = [problem.get_weight(edge[0], edge[1]) for edge in problem.get_edges()]
    initial_temperature = -np.array(distances).std() / math.log(0.47)
    repetitions = problem.dimension * (problem.dimension - 1)
    return {'initial_temperature': initial_temperature, 'repetitions': repetitions, 'cooling_factor': 0.95}

def def_cfg_aco(problem: tsplib95.models.StandardProblem) -> dict:
    antcount = problem.dimension
    return {'antcount': antcount, 'alpha': 1, 'beta': 2, 'evaporation': 0.98, 'pbest': 0.05}

def mhrun(instance: Path,  
            algorithm: str,
            terminate: dict = None, 
            config: dict = None, 
            fname_convdata = None):

    if algorithm == 'SA':
        # use default configuration and termination values if none are given
        cfg = DEF_CFG_SA_50N if config == None else config
        terminate = BASE_TERM if terminate == None else terminate
        return sa(instance = instance, cfg = cfg, terminate = terminate, fname_convdata = fname_convdata)

    if algorithm == 'ACO':
        # use default configuration and termination values if none are given
        cfg = DEF_CFG_ACO_50N if config == None else config
        terminate = BASE_TERM if terminate == None else terminate
        return aco(instance = instance, cfg = cfg, terminate = terminate, fname_convdata = fname_convdata)

    if algorithm == 'GA':
        # use default configuration and termination values if none are given
        cfg = DEF_CFG_GA if config == None else config
        terminate = BASE_TERM if terminate == None else terminate
        return ga(instance = instance, cfg = cfg, terminate = terminate, fname_convdata = fname_convdata)

    else: 
        print('Error: Algorithm ' + '"' + algorithm + '" not found!')
    
def mhruns(budget_tuned: int, 
                instancefolder: str, 
                algorithm: str,
                terminate: dict = None, 
                config: dict = None):
    
    seed = random.randint(1, 10000)
    random.seed(seed)
    results = []
    entries = Path(instancefolder)
    fname = '-'.join([algorithm, str(budget_tuned), str(config), str(terminate), str(seed)]) + '.csv'

    for entry in entries.iterdir():

        if entry.suffix != '.tsp':
                continue
        
        result = mhrun(entry, algorithm = algorithm, terminate = terminate, config = config, \
                        fname_convdata = Path('myproject/data/conv/' +  fname))
        results.append(Result(budget_tuned, entry.name, result['qualdev'], result['evals'], result['time']))

    fpath = Path('myproject/data/results/' + fname)
    df: pd.DataFrame = pd.DataFrame(results)
    mode = 'a' if fpath.exists() else 'w+'
    df.to_csv(fpath.absolute(), mode = mode, index = False)

# set standard output
sys.stdout = open('myproject/' + str(datetime.datetime.now()) + '.log', 'w')

# mhruns(instancefolder = 'myproject/instances/50nodes', algorithm = 'SA', budget_tuned = 0)
# mhruns(instancefolder = 'myproject/instances/50nodes', algorithm = 'ACO', budget_tuned = 0)
# mhruns(instancefolder = 'myproject/instances/50nodes', algorithm = 'GA', budget_tuned = 0)

# scenario optimize qualdev few evals
tuning_wrapper.smac(budget = 5000, algorithm = 'SA', optimize = 'qualdev',
                        terminate = {'qualdev': 0, 'evals': 1000}, train_instances_dir = 'myproject/instances/50nodes')
tuning_wrapper.smac(budget = 5000, algorithm = 'GA', optimize = 'qualdev',
                        terminate = {'qualdev': 0, 'evals': 1000}, train_instances_dir = 'myproject/instances/50nodes')
tuning_wrapper.smac(budget = 5000, algorithm = 'ACO', optimize = 'qualdev',
                        terminate = {'qualdev': 0, 'evals': 1000}, train_instances_dir = 'myproject/instances/50nodes')
tuning_wrapper.irace(budget = 5000, algorithm = 'SA', initial_parameters = DEF_CFG_SA_50N, optimize = 'qualdev',
                        terminate = {'qualdev': 0, 'evals': 1000}, train_instances_dir = 'myproject/instances/50nodes')
tuning_wrapper.irace(budget = 5000, algorithm = 'GA', initial_parameters = DEF_CFG_GA, optimize = 'qualdev',
                        terminate = {'qualdev': 0, 'evals': 1000}, train_instances_dir = 'myproject/instances/50nodes')
tuning_wrapper.irace(budget = 5000, algorithm = 'ACO', initial_parameters = DEF_CFG_ACO_50N, optimize = 'qualdev',
                        terminate = {'qualdev': 0, 'evals': 1000}, train_instances_dir = 'myproject/instances/50nodes')