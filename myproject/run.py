# pylint: disable=no-member
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
from myproject.helpers import incumbents_smac, config_to_cand_params_smac, from_cand_params
from myproject.helpers import final_elites_irace

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

metaheuristics = []
tuners = []

termination_conditions = []

for constellation in itertools.product(tuners, metaheuristics, termination_conditions):
    print(constellation)
    if constellation[0] == 'irace':
        tuning_wrapper.irace(budget = 5000, algorithm = constellation[1], initial_parameters = DEF_CFGS[constellation[1]], 
                        optimize = 'qualdev', terminate = constellation[2], train_instances_dir = 'myproject/instances/50nodes')
    if constellation[0] == 'smac':
        tuning_wrapper.smac(budget = 5000, algorithm = constellation[1], optimize = 'qualdev',
                        terminate = constellation[2], train_instances_dir = 'myproject/instances/50nodes')

### Postprocessing SMAC ###
### Test incumbents to enable later tuning trajectory (and also tuning result) comparisons ###

# smac ...
tuning_budgets = [5000]
metaheuristics = ['SA', 'ACO', 'GA']
termination_conditions = [
    {'qualdev': 0, 'evals': 1000}, 
    {'qualdev': 0, 'evals': 10000}, 
    {'qualdev': 0, 'evals': 100000}
]
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

    # create convergence Files for last incumbent with base termination condition
    string_config_smac = incumbents['Configuration'][len(incumbents) - 1]
    cand_params = config_to_cand_params_smac(string_config_smac)
    algorithm, config, terminate, optimize = from_cand_params(cand_params)

    mhrun_fname = cmhrun_fname(algorithm, config, BASE_TERM, optimize)
    if not Path('myproject/data/results/' + mhrun_fname + '.csv').exists():
        print(Path('myproject/data/results/' + mhrun_fname).absolute())
        print('Metaheuristic run with ' + str((algorithm, constellation[0], config, terminate, optimize)))
        mhruns(constellation[0], 'myproject/instances/50nodes/test', algorithm, terminate, config)

### Postprocessing irace ###

# create convergence files with final elite and base termination condition

for elite in final_elites_irace:
    mhrun_fname = cmhrun_fname(elite[0], elite[1], elite[2], elite[3])
    if not Path('myproject/data/results/' + mhrun_fname + '.csv').exists():
        print(Path('myproject/data/results/' + mhrun_fname).absolute())
        print('Metaheuristic run with ' + str((elite[0], elite[1], elite[2], elite[3])))
        mhruns(5000, 'myproject/instances/50nodes/test', elite[0], elite[2], elite[1])

# create convergence files for base cfgs and termination conditions
for terminate in termination_conditions:
    for metaheuristic in metaheuristics:
        mhrun_fname = cmhrun_fname(metaheuristic, DEF_CFGS[metaheuristic], terminate, 'qualdev')
        if not Path('myproject/data/results/' + mhrun_fname + '.csv').exists():
            print(metaheuristic)
            print(terminate)
            mhruns(0, 'myproject/instances/50nodes/test', metaheuristic, terminate, DEF_CFGS[metaheuristic])