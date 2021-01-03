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
import os.path
import re
import subprocess
import sys

import numpy as np 
import pandas as pd 

import random
import tsplib95

import myproject.metaheuristic.sa

from myproject.metaheuristic.sa import sa

VALID_PARAMETERS = [
    'initial_temperature',
    'repetitions',
    'cooling_factor',
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

def create_config_terminate_optimize(cand_params: list) -> tuple:
    params_as_dict = {}
    
    # turn given parameters into dictionary form
    while len(cand_params) > 1:
        # Get and remove first and second elements.
        param = cand_params.pop(0)
        if param[:2] == '--':
            param = param[2:]
        value = cand_params.pop(0)

        if param not in VALID_PARAMETERS:
            target_runner_error('Unknown parameter %s' % (param))

        params_as_dict[param] = value
    
    # transform parameters
    params_as_dict['initial_temperature'] = float(params_as_dict['initial_temperature'])
    params_as_dict['repetitions'] = int(params_as_dict['repetitions'])
    params_as_dict['cooling_factor'] = float(params_as_dict['cooling_factor'])
    
    # separate out termination criteria
    return separate_cfg_terminate(params_as_dict)

def separate_cfg_terminate(params: dict) -> tuple:    
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

if __name__=='__main__':
    if len(sys.argv) < 5:
        print("\nUsage: ./target-runner.py <configuration_id> <instance_id> <seed> <instance_path_name> <list of parameters>\n")
        sys.exit(1)

    # Get the parameters as command line arguments.
    configuration_id = sys.argv[1]
    instance_id = sys.argv[2]
    seed = sys.argv[3]
    instance = sys.argv[4]
    cand_params = sys.argv[5:]

    # load problem
    problem: tsplib95.models.StandardProblem = tsplib95.load(instance)

    # Tuned parameters
    initial_temperature = None
    repetitions = None
    cooling_factor = None
    cfg, terminate, optimize = create_config_terminate_optimize(cand_params)
                  
    # Run runner
    result = sa(instance = instance, 
                        initial_temperature = cfg['initial_temperature'],
                        repetitions = cfg['repetitions'],
                        cooling_factor = cfg['cooling_factor'],
                        terminate = terminate,
                        fconvergence = None)
    print(result[optimize])
    
    sys.exit(0)