#!/home/damian/anaconda3/envs/ma-code/bin/python

###############################################################################
# This script is the command that is executed every run.
# Check the examples in examples/
#
# This script is run in the execution directory (execDir, --exec-dir).
#
# PARAMETERS:
# argv[1] is the candidate configuration number
# argv[2] is the instance ID
# argv[3] is the seed
# argv[4] is the instance name
# The rest (argv[5:]) are parameters to the run
#
# RETURN VALUE:
# This script should print one numerical value: the cost that must be minimized.
# Exit with 0 if no error, with 1 in case of error
###############################################################################

import datetime
import sys
import os

from pathlib import Path
from rpy2 import robjects
import numpy as np 

import tsplib95

from myproject.metaheuristic.sa import sa
from myproject.metaheuristic.aco import aco

SMAC_EXECUTABLE = 'smac-v2.10.03-master-778/smac'

VALID_PARAMETERS = [
    'algorithm',
    'initial_temperature',
    'repetitions',
    'cooling_factor',
    'initial_pheromone',
    'antcount',             
    'alpha',                
    'beta',                 
    'Q',                    
    'localsearch',
    'evaporation',
    'term_evals',
    'term_evals_val',
    'term_qualdev',
    'term_qualdev_val',
    'term_time',
    'term_time_val',
    'term_noimprovement',
    'term_noimpr_temp_val',
    'term_noimpr_accp_val',
    'optimize'
]

# Useful function to print errors.
def target_runner_error(msg):
    now = datetime.datetime.now()
    print(str(now) + " error: " + msg)
    sys.exit(1)

def dict2params(asdict):
    cand_params = []
    for i in range(0, len(asdict)):
        cand_params.append(asdict.index[i])
        cand_params.append(asdict.values[i])
    
    return params2dict(cand_params)

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
            target_runner_error('Unknown parameter \"%s\"' % (param) + '\n Valid Parameters: ' + str(VALID_PARAMETERS))

        params_as_dict[param] = value

    # transform parameters
    if params_as_dict['algorithm'] == 'SA':
        params_as_dict['initial_temperature'] = float(params_as_dict['initial_temperature'])
        params_as_dict['repetitions'] = int(params_as_dict['repetitions'])
        params_as_dict['cooling_factor'] = float(params_as_dict['cooling_factor'])
    elif params_as_dict['algorithm'] == 'ACO':
        params_as_dict['initial_pheromone'] = float(params_as_dict['initial_pheromone'])
        params_as_dict['antcount'] = int(params_as_dict['antcount'])
        params_as_dict['alpha'] = float(params_as_dict['alpha'])
        params_as_dict['beta'] = float(params_as_dict['beta'])
        params_as_dict['Q'] = float(params_as_dict['Q'])
        params_as_dict['evaporation'] = float(params_as_dict['evaporation'])

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

def smac_add_fixed_params(pcs_file: str, algorithm: str, optimize: str, terminate: dict):

    # set algorithm, optimize in the parameter space file...
    fparams = {'algorithm': algorithm, 'optimize': optimize}
    f = open(pcs_file, 'r')
    lines = []
    for line in f.readlines():
        fparam = line.split(' ')[0]

        # add termination conditions later
        if fparam.startswith('term_'):
            continue

        # non-termination-condition fixed parameters
        if fparam in fparams.keys():
            line = '{fparam} categorical {{{value}}}[{value}]\n'.format(fparam = fparam, value = fparams[fparam])

        lines.append(line)
    f.close()

    # set termination condition lines
    for key in terminate:
        if key == 'noimprovement':
            temperatures = terminate['noimprovement']['temperatures']
            accportion = terminate['noimprovement']['accportion']
            lines.append('term_noimprovement categorical {True}[True]\n')
            lines.append('term_noimpr_temp_val categorical {{{value}}}[{value}]\n'.format(value = temperatures))
            lines.append('term_noimpr_accp_val categorical {{{value}}}[{value}]\n'.format(value = accportion))
        else:
            lines.append('term_{param} categorical {{True}}[True]\n'.format(param = key))
            lines.append('term_{param}_val categorical {{{value}}}[{value}]\n'.format(param = key, value = terminate[key]))

    f = open(pcs_file, 'w')
    f.writelines(lines)
    f.close()

def smac(budget: int, 
            algorithm: str,
            terminate: dict,
            optimize: str,
            train_instances_dir: str) -> None:
    test_instances_dir = train_instances_dir + '/test'

    pcs_file = 'myproject/tuning-settings/{}-parameters-smac.pcs'.format(algorithm.lower())
    smac_add_fixed_params(pcs_file, algorithm, optimize, terminate)
    
    os.system('''%s --instances %s --instance-suffix tsp --test-instances %s --numberOfRunsLimit %i \
            --runObj QUALITY --pcs-file %s --algo-deterministic False \
            --algo "python3 ./myproject/tuning_wrapper.py"''' \
            % (SMAC_EXECUTABLE, train_instances_dir, test_instances_dir, budget, pcs_file))

def irace(budget: int, 
            algorithm: str,
            terminate: dict, 
            optimize: str,
            train_instances_dir: str) -> dict:

    robjects.r('library("irace")')

    # define scenario
    robjects.r('scenario = list()')
    robjects.r('scenario$trainInstancesDir = \"' + train_instances_dir + '\"')
    robjects.r('scenario$trainInstancesFile = \"' + train_instances_dir + '/trainInstancesFile\"')
    robjects.r('scenario$maxExperiments = ' + str(budget))
    robjects.r('scenario$targetRunner = "/home/damian/Desktop/MA/macode/myproject/tuning_wrapper.py"')

    if algorithm == 'SA':
        robjects.r('parameters = readParameters("myproject/tuning-settings/sa-parameters.txt")')
    elif algorithm == 'ACO':
        robjects.r('parameters = readParameters("myproject/tuning-settings/aco-parameters.txt")')

    # set optimize parameter
    robjects.r('parameters$domain$optimize = \"' + optimize + '\"')

    # set termination conditions
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

if __name__=='__main__':
    if len(sys.argv) < 5:
        print("\nUsage: ./target-runner.py <configuration_id> <instance_id> <seed> <instance_path_name> <list of parameters>\n")
        sys.exit(1)

    # Get the parameters as command line arguments.
    tuner = None
    instance = None 
    lparams = None

    # case differentiation between smac and irace
    if (Path(sys.argv[4]).exists()):
        # called by irace 
        tuner = 'irace'
        instance = sys.argv[4]
        lparams = sys.argv[5:]
    elif (Path(sys.argv[1]).exists()):
        # called by smac
        tuner = 'SMAC'
        instance = sys.argv[1]
        lparams = sys.argv[6:]

    # Tuned parameters
    dparams = params2dict(lparams)
    algorithm = dparams.pop('algorithm')
    cfg, terminate, optimize = separate_cfg_term_opt(dparams)
                  
    # Run runner
    result = None
    if algorithm == 'SA':
        result = sa(instance = instance, cfg = cfg, terminate = terminate, fname_convdata = None)
    elif algorithm == 'ACO':
        result = aco(instance = instance, cfg = cfg, terminate = terminate, fname_convdata = None)

    if tuner == 'irace':
        print(result[optimize])
    elif tuner == 'SMAC':
        print('Result of this algorithm run: %s, %f, %i, %f, %i, %s' % ('SUCCESS', result['time'], result['evals'], result['qualdev'], 1, 0) )
    
    sys.exit(0)