# pylint: disable=no-member

from pathlib import Path
import random
import math

import numpy as np
import pandas as pd
import tsplib95

from myproject.metaheuristic.sa import sa
from myproject.metaheuristic.aco import aco
from myproject.metaheuristic.ga import ga
import myproject.tuning_wrapper as tuning_wrapper

from collections import namedtuple

BASE_TERM = {'qualdev': 0, 'evals': 100000}
DEF_CFG_SA_50N = {'initial_temperature': 336, 'repetitions': 870, 'cooling_factor': 0.95}
DEF_CFG_GA = {'popsize': 200, 'mut_rate': 0.01, 'rank_weight': 1.9}
DEF_CFG_ACO_50N = {'antcount': 50, 'alpha': 1, 'beta': 2, 'evaporation': 0.98, 'pbest': 0.05}

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
    
    seed = random.seed(10000)
    results = []
    entries = Path(instancefolder)
    fname = '-'.join([algorithm, str(budget_tuned), str(config), str(terminate), str(seed)]) + '.csv'

    for entry in entries.iterdir():

        if entry.suffix != '.tsp':
                continue
        
        result = mhrun(entry, algorithm = algorithm, terminate = terminate, config = config, fname_convdata = 'myproject/data/' + 'conv-' + fname)
        results.append(Result(budget_tuned, entry.name, result['qualdev'], result['evals'], result['time']))

    fpath = Path('myproject/data/' + 'res' + fname)
    df: pd.DataFrame = pd.DataFrame(results)
    mode = 'a' if fpath.exists() else 'w+'
    df.to_csv(fpath.absolute(), mode = mode, index = False)

mhruns(instancefolder = 'myproject/instances/50nodes', algorithm = 'SA', budget_tuned = 0)
mhruns(instancefolder = 'myproject/instances/50nodes', algorithm = 'ACO', budget_tuned = 0)
mhruns(instancefolder = 'myproject/instances/50nodes', algorithm = 'GA', budget_tuned = 0)