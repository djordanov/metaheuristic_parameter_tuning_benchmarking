# pylint: disable=no-member
import sys
from pathlib import Path
import random
import datetime
import itertools

import pandas as pd
from rpy2 import robjects

from myproject.metaheuristic.sa import sa
from myproject.metaheuristic.aco import aco
from myproject.metaheuristic.ga import ga
import myproject.tuning_wrapper as tuning_wrapper

from myproject.helpers import BASE_TERM, DEF_CFG_SA_50N, DEF_CFG_GA, DEF_CFG_ACO_50N
from myproject.helpers import cmhrun_fname, ctun_fname
from myproject.helpers import incumbents_smac, config_to_cand_params_smac, from_cand_params

from collections import namedtuple

Result = namedtuple('Result', 'tuning_budget instance quality evals time')

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
    mhrun_fname = cmhrun_fname(algorithm, config, terminate, optimize)

    for entry in entries.iterdir():

        if entry.suffix != '.tsp':
                continue
        
        result = mhrun(entry, algorithm = algorithm, terminate = terminate, config = config, \
                        fname_convdata = Path('myproject/data/conv/' + mhrun_fname + '.csv'))
        results.append(Result(budget_tuned, entry.name, result['qualdev'], result['evals'], result['time']))

    fpath = Path('myproject/data/results/' + mhrun_fname + '.csv')
    df: pd.DataFrame = pd.DataFrame(results)
    mode = 'a' if fpath.exists() else 'w+'
    df.to_csv(fpath.absolute(), mode = mode, index = False)

# set standard output
# sys.stdout = open('myproject/data/' + str(datetime.datetime.now()) + '.log', 'w')

### Run tuners ###

metaheuristics = [ ]
tuners = ['smac']

default_configs = {
    'GA': DEF_CFG_GA
}
termination_conditions = [{'qualdev': 0, 'evals': 100000}]

for constellation in itertools.product(tuners, metaheuristics, termination_conditions):
    print(constellation)
    if constellation[0] == 'irace':
        tuning_wrapper.irace(budget = 5000, algorithm = constellation[1], initial_parameters = default_configs[constellation[1]], 
                        optimize = 'qualdev', terminate = constellation[2], train_instances_dir = 'myproject/instances/50nodes')
    if constellation[0] == 'smac':
        tuning_wrapper.smac(budget = 5000, algorithm = constellation[1], optimize = 'qualdev',
                        terminate = constellation[2], train_instances_dir = 'myproject/instances/50nodes')

### Postprocessing SMAC ###
### Test incumbents to enable later tuning trajectory (and also tuning result) comparisons ###

# smac ...
tuning_budgets = [5000]
metaheuristics = ['SA', 'ACO']
termination_conditions = [{'qualdev': 0, 'evals': 1000}, {'qualdev': 0, 'evals': 10000}, {'qualdev': 0, 'evals': 100000}]
optimizes = ['qualdev']

for constellation in itertools.product(tuning_budgets, metaheuristics, termination_conditions, optimizes):
    ftun_smac_run = ctun_fname(constellation[0], constellation[1], constellation[2], constellation[3])

    # test incumbents
    incumbents = incumbents_smac(ftun_smac_run)
    for configuration in incumbents['Configuration']:
        cand_params = config_to_cand_params_smac(configuration)
        algorithm, config, terminate, optimize = from_cand_params(cand_params)
        mhrun_fname = cmhrun_fname(algorithm, config, terminate, optimize)
        if not Path('myproject/data/results/' + mhrun_fname + '.csv').exists():
            print(Path('myproject/data/results/' + mhrun_fname).absolute())
            print('Metaheuristic run with ' + str((algorithm, constellation[0], config, terminate, optimize)))
            mhruns(constellation[0], 'myproject/instances/50nodes/test', algorithm, terminate, config)