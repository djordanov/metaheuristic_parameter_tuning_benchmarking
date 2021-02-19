import math
from pathlib import Path
import sys 
import datetime

import numpy as np
import pandas as pd
from rpy2 import robjects
import tsplib95

# default termination condition and default parameter configurations

BASE_TERM = {'qualdev': 0, 'evals': 100000}
DEF_CFG_SA_50N = {'initial_temperature': (-2 * 253.83) / math.log(0.47), 'repetitions': 2450, 'cooling_factor': 0.95}
DEF_CFG_GA = {'popsize': 200, 'mut_rate': 0.01, 'rank_weight': 1.9}
DEF_CFG_ACO_50N = {'antcount': 50, 'alpha': 1, 'beta': 2, 'evaporation': 0.98, 'pbest': 0.05}

DEF_CFGS = {
    'SA': DEF_CFG_SA_50N,
    'GA': DEF_CFG_GA,
    'ACO': DEF_CFG_ACO_50N
}

# valid parameters for tuning configuration transformations
VALID_PARAMETERS = [
    'algorithm',
    'initial_temperature',
    'repetitions',
    'cooling_factor',
    'initial_pheromone',
    'antcount',             
    'alpha',                
    'beta',                 
    'pbest',                    
    'evaporation',
    'popsize',
    'mut_rate',
    'rank_weight',
    'term_evals',
    'term_evals_val',
    'term_qualdev',
    'term_qualdev_val',
    'optimize'
]

# configurations were copied manually :)

final_elites_irace = [
    ('ACO', {'antcount': 439, 'alpha': 3.543, 'beta': 14.8282, 'pbest': 84.5142, 'evaporation': 0.4428}, BASE_TERM, 'qualdev'),
    ('ACO', {'antcount': 132, 'alpha': 0.6219, 'beta': 5.4251, 'pbest': 62.2747, 'evaporation': 0.7806}, BASE_TERM, 'qualdev'),
    ('ACO', {'antcount': 47, 'alpha': 0.8079, 'beta': 6.6904, 'pbest': 0.7138, 'evaporation': 0.9638}, BASE_TERM, 'qualdev'),
    ('GA', {'popsize': 1, 'mut_rate': 0.8581, 'rank_weight': 0.2627}, BASE_TERM, 'qualdev'),
    ('GA', {'popsize': 1, 'mut_rate': 0.6601, 'rank_weight': 16.519}, BASE_TERM, 'qualdev'),
    ('GA', {'popsize': 281, 'mut_rate': 0.5634, 'rank_weight': 0.9341}, BASE_TERM, 'qualdev'),
    ('SA', {'initial_temperature': 22.1271, 'repetitions': 7902, 'cooling_factor': 0.8808}, BASE_TERM, 'qualdev'),
    ('SA', {'initial_temperature': 258.6476, 'repetitions': 857, 'cooling_factor': 0.6707}, BASE_TERM, 'qualdev'),
    ('SA', {'initial_temperature': 235.7388, 'repetitions': 3831, 'cooling_factor': 0.8593}, BASE_TERM, 'qualdev')
]

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

# file and folder names
def ctun_fname(tuning_budget: int, algorithm: str, terminate: dict, optimize: str) -> str:
    return '-'.join([str(tuning_budget), algorithm, str(sorted(terminate.items())), optimize])

def cmhrun_fname(algorithm: str, config: dict, terminate: dict, optimize: str):
    return '-'.join([algorithm, str(sorted(terminate.items())), str(sorted(config.items())), optimize])

# get configurations from tuning output files
def config_to_cand_params_smac(configuration: str) -> list:
    config = configuration.replace(' ', '').replace("'", "")
    cand_params = [x.split('=') for x in config.split(',')]
    return list(np.array(cand_params).flatten())

def tun_fname_to_cand_params_irace(tun_fname: str) -> list:
    tun_fname = 'test' + tun_fname + '.Rdata'
    robjects.r("library('irace')")
    robjects.r("config = getFinalElites(iraceResults, n = 1)")

    attributes = list(robjects.r("colnames(config)"))[1:-1]
    values = [ x[0] for x in  list(robjects.r("config"))[1:-1] ]
    cand_params = list(zip(attributes, values))
    return list(np.array(cand_params).flatten())

def from_cand_params(cand_params: list) -> tuple:
    dparams = params2dict(cand_params)
    algorithm = dparams.pop('algorithm')
    config, terminate, optimize = separate_cfg_term_opt(dparams)
    return algorithm, config, terminate, optimize

def params2dict(cand_params: list) -> dict:
    params_as_dict = {}
    
    # turn given parameters into dictionary form
    while len(cand_params) > 1:
        # Get and remove first and second elements.
        param = cand_params.pop(0)
        if param[:2] == '--': # irace
            param = param[2:]
        elif param[:1] == '-': # smac
            param = param[1:]
        value = cand_params.pop(0)

        if param not in VALID_PARAMETERS:
            print(param)
            msg = 'Unknown parameter \"%s\"' % (str(param)) + '\n Valid Parameters: ' + str(VALID_PARAMETERS)
            now = datetime.datetime.now()
            print(str(now) + " error: " + msg)
            sys.exit(1)

        params_as_dict[param] = value

    # transform parameters
    if params_as_dict['algorithm'] == 'SA':
        params_as_dict['initial_temperature'] = float(params_as_dict['initial_temperature'])
        params_as_dict['repetitions'] = int(float(params_as_dict['repetitions']))
        params_as_dict['cooling_factor'] = float(params_as_dict['cooling_factor'])
    elif params_as_dict['algorithm'] == 'ACO':
        params_as_dict['antcount'] = int(float(params_as_dict['antcount']))
        params_as_dict['alpha'] = float(params_as_dict['alpha'])
        params_as_dict['beta'] = float(params_as_dict['beta'])
        params_as_dict['pbest'] = float(params_as_dict['pbest'])
        params_as_dict['evaporation'] = float(params_as_dict['evaporation'])
    elif params_as_dict['algorithm'] == 'GA':
        params_as_dict['popsize'] = int(float(params_as_dict['popsize']))
        params_as_dict['mut_rate'] = float(params_as_dict['mut_rate'])
        params_as_dict['rank_weight'] = float(params_as_dict['rank_weight'])

    return params_as_dict

def separate_cfg_term_opt(params: dict) -> tuple:    
    optimize = params.pop('optimize')
    cfg = params.copy()
    terminate = {}

    for key in params:
        if key.startswith('term_'):
            if key == 'term_evals' and cfg[key] == 'True':
                terminate['evals'] = int(cfg.pop('term_evals_val'))
                cfg.pop('term_evals')
            if key == 'term_qualdev' and cfg[key] == 'True':
                terminate['qualdev'] = float(cfg.pop('term_qualdev_val')) 
                cfg.pop('term_qualdev')
            if key == 'term_time' and cfg[key] == 'True':
                terminate['time'] = int(cfg.pop('term_time_val'))
                cfg.pop('term_time')
            if key == 'term_noimprovement' and cfg[key] == 'True':
                terminate['noimprovement'] = {}
                terminate['noimprovement']['temperatures'] = float(cfg.pop('term_noimpr_temp_val'))
                terminate['noimprovement']['accportion'] = float(cfg.pop('term_noimpr_accp_val'))
                cfg.pop('term_noimprovement')             
    return cfg, terminate, optimize


### build tuning trajectory dataframe ###
    
def incumbents_smac(fname: str) -> pd.DataFrame:
    folder = Path('myproject/data/smac/' + fname + '/NoScenarioFile')
    subfolder = list(filter(lambda file: 'state-run' in file.name, folder.iterdir()))[0]
    fname_run_results = list(filter(lambda file: 'runs_and_results-it' in file.name, subfolder.iterdir()))[0]
    run_results = pd.read_csv(fname_run_results.absolute())
    configurations = run_results['Run History Configuration ID']

    # the incumbent is always the configuration with the most runs, i. e. with the most occurences
    changing_points = []
    incumbent = None
    dcs = pd.get_dummies(configurations).cumsum()
    for i in range(0, len(dcs)):
        runs = i
        rinc = dcs.loc[i].idxmax()
        if rinc != incumbent:
            incumbent = rinc
            changing_points.append((runs, incumbent))
    incumbents = pd.DataFrame(data = changing_points, columns = ['runs', 'incumbent'])

    # join Configurations
    ptraj = list(filter(lambda file: 'detailed-traj-run' in file.name, folder.iterdir()))[0]
    traj = pd.read_csv(ptraj.absolute(), sep = ',', skipinitialspace = True, quotechar = '"')
    incumbents['Configuration'] = \
        [ traj[traj['Incumbent ID'] == incumbent]['Full Configuration'].iloc[0] for incumbent in incumbents.incumbent ]
        
    return incumbents

def elite_results_smac(tuning_budget: int, algorithm: str, terminate: dict, optimize: str) -> pd.DataFrame:
    tun_fname = ctun_fname(tuning_budget, algorithm, terminate, optimize)

    # load incumbents with configurations
    incumbents = incumbents_smac(tun_fname)
    incumbent_qualities = []

    # join configuration qualities
    for i in incumbents.index:
        string_configuration_smac = incumbents.iloc[i]['Configuration']
        cand_params = config_to_cand_params_smac(string_configuration_smac)
        algorithm, config, terminate, optimize = from_cand_params(cand_params)

        mhrun_fname = cmhrun_fname(algorithm, config, terminate, optimize)
        mhrun_results = pd.read_csv('myproject/data/results/' + mhrun_fname + '.csv')
        incumbent_qualities.append(pd.to_numeric(mhrun_results[optimize], errors = 'coerce').mean())

    elite_results = incumbents[['runs']].copy()
    elite_results[optimize] = incumbent_qualities
    return elite_results

def elite_results_irace(tuning_budget: int, algorithm: str, terminate: dict, optimize: str) -> pd.DataFrame:
    fname = 'myproject/data/irace/' + ctun_fname(tuning_budget, algorithm, terminate, optimize)
    fiterations = fname + '-iterations.csv'
    ftest_experiments = fname + '-test-experiments.csv'
    iterations = pd.read_csv(fiterations)
    elite_qualities = pd.read_csv(ftest_experiments).mean()
    
    elite_results = pd.DataFrame({
        'runs': iterations['experiments'].cumsum(),
        optimize: [elite_qualities[str(elite)] for elite in iterations.elite]
    })

    return elite_results

def tun_traj(tuner: str, tuning_budget: int, algorithm: str, terminate: dict, optimize: str):
    elite_results = elite_results_irace(tuning_budget, algorithm, terminate, optimize) if tuner == 'irace' \
        else elite_results_smac(tuning_budget, algorithm, terminate, optimize)
    
    # get default result
    mhrun_fname = cmhrun_fname(algorithm, DEF_CFGS[algorithm], terminate, optimize)
    mhrun_results = pd.read_csv('myproject/data/results/' + mhrun_fname + '.csv')
    default_result = mhrun_results.mean()[optimize]

    # build tuning trajectory (number of runs - result in comparison to default result) ...
    nruns = list(range(tuning_budget + 1))
    results = np.full((tuning_budget + 1,), np.nan)
    results[0] = default_result
    for runs in elite_results['runs']:
        results[runs] = elite_results.loc[elite_results['runs'] == runs][optimize]

    # create dataframe and pad in result of runs without incumbent
    traj = pd.DataFrame({'runs': nruns, optimize: results})
    traj[optimize] = traj[optimize].pad()

    # move to tuning result as a proportion of default result
    traj[optimize] = traj[optimize] / default_result
    return traj