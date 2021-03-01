# pylint: disable=no-member
import datetime
import sys
from pathlib import Path
import random
import itertools

import pandas as pd

from myproject.metaheuristic.sa import sa
from myproject.metaheuristic.aco import aco
from myproject.metaheuristic.ga import ga
import myproject.tuning_wrapper as tuning_wrapper

from myproject.helpers import BASE_TERM, DEF_CFG_SA_50N, DEF_CFG_GA, DEF_CFG_ACO_50N, DEF_CFGS
from myproject.helpers import cmhrun_fname, ctun_fname
from myproject.helpers import incumbents_smac, config_to_cand_params_smac, from_cand_params, tun_fname_irace_to_cand_params

from collections import namedtuple

Result = namedtuple('Result', 'tuning_budget instance qualdev evals time')

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
                config: dict = None,
                optimize: str = 'qualdev'):
    
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

### Run tuners Tune evals###

metaheuristics = ['GA']
tuners = ['irace', 'smac']
optimize = 'evals'
tuning_budget = 1000

# set termination condition to time out after 100000 evals 
# and terminate when the same qualdev is reached as with default parameters and 100000 evals
termination_conditions = {}
for metaheuristic in metaheuristics:
    mhrun_fname = cmhrun_fname(metaheuristic, DEF_CFGS[metaheuristic], BASE_TERM, 'qualdev')
    mhrun_results = pd.read_csv('myproject/data/results/' + mhrun_fname + '.csv')
    default_qualdev = round(mhrun_results.mean()['qualdev'], 15)
    termination_conditions[metaheuristic] = {'qualdev': default_qualdev, 'evals': 100000}

# actually run tuners
for constellation in itertools.product(tuners, metaheuristics, termination_conditions):
    for terminate in termination_conditions:
        print(str(constellation) + str(terminate))

        if constellation[0] == 'irace':
            tuning_wrapper.irace(budget = tuning_budget, algorithm = constellation[1], initial_parameters = DEF_CFGS[constellation[1]], 
                            optimize = optimize, terminate = terminate, train_instances_dir = 'myproject/instances/50nodes')
        if constellation[0] == 'smac':
            tuning_wrapper.smac(budget = tuning_budget, algorithm = constellation[1], optimize = optimize,
                            terminate = terminate, train_instances_dir = 'myproject/instances/50nodes')

### Postprocessing SMAC ###
### Test smac incumbents to enable later tuning trajectory (and also tuning result) comparisons ###

# # smac ...

for terminate in termination_conditions:
    tun_fname = ctun_fname(tuning_budget, 'GA', terminate, optimize)

    # test incumbents
    incumbents = incumbents_smac(tun_fname)
    for configuration in incumbents['Configuration']:
        cand_params = config_to_cand_params_smac(configuration)
        algorithm, config, term, optimize = from_cand_params(cand_params)
        mhrun_fname = cmhrun_fname(algorithm, config, terminate, optimize)
        print('Metaheuristic run with ' + str((algorithm, tuning_budget, config, terminate, optimize)))
        if not Path('myproject/data/results/' + mhrun_fname + '.csv').exists():
            print(Path('myproject/data/results/' + mhrun_fname).absolute())
            mhruns(tuning_budget, 'myproject/instances/50nodes/test', algorithm, terminate, config)

    # create convergence file smac for last incumbent with base termination condition
    
    # smac
    string_config_smac = incumbents['Configuration'][len(incumbents) - 1]
    cand_params = config_to_cand_params_smac(string_config_smac)
    algorithm, config, terminate, optimize = from_cand_params(cand_params)

    mhrun_fname = cmhrun_fname(algorithm, config, BASE_TERM, optimize)
    if not Path('myproject/data/results/' + mhrun_fname + '.csv').exists():
        print(Path('myproject/data/results/' + mhrun_fname).absolute())
        print('Metaheuristic run with ' + str((algorithm, tuning_budget, config, BASE_TERM, optimize)))
        mhruns(tuning_budget, 'myproject/instances/50nodes/test', algorithm, BASE_TERM, config)

    # irace
    cand_params = tun_fname_irace_to_cand_params(tun_fname)
    algorithm, config, terminate, optimize = from_cand_params(cand_params)

    mhrun_fname = cmhrun_fname(algorithm, config, BASE_TERM, optimize)
    if not Path('myproject/data/results/' + mhrun_fname + '.csv').exists():
        print(Path('myproject/data/results/' + mhrun_fname).absolute())
        print('Metaheuristic run with ' + str((algorithm, tuning_budget, config, BASE_TERM, optimize)))
        mhruns(tuning_budget, 'myproject/instances/50nodes/test', algorithm, BASE_TERM, config)