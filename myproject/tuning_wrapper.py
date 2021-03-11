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

from myproject.helpers import ctun_fname, params2dict, separate_cfg_term_opt
import sys
import os

from pathlib import Path
from rpy2 import robjects
import pandas as pd

from myproject.metaheuristic.sa import sa
from myproject.metaheuristic.aco import aco
from myproject.metaheuristic.ga import ga

SMAC_EXECUTABLE = 'smac-v2.10.03-master-778/smac'

def dict2params(asdict):
    cand_params = []
    for i in range(0, len(asdict)):
        cand_params.append(asdict.index[i])
        cand_params.append(asdict.values[i])
    
    return params2dict(cand_params)

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
            train_instances_dir: str,
            outdir_suffix: str = '') -> None:

    outdir = 'myproject/data/smac/' + ctun_fname(budget, algorithm, terminate, optimize) + outdir_suffix
    if not Path(outdir).exists():
        Path(outdir).mkdir()

    pcs_file = 'myproject/tuning-settings/smac-{}-parameters.pcs'.format(algorithm.lower())
    smac_add_fixed_params(pcs_file, algorithm, optimize, terminate)
    call = '''%s --instances %s --instance-suffix tsp \
                 --validation false --num-seeds-per-test-instance 1 \
            --numberOfRunsLimit %i --runObj QUALITY --pcs-file %s \
            --algo-deterministic False --outdir "%s"\
            --algo "python3 ./myproject/tuning_wrapper.py"''' \
            % (SMAC_EXECUTABLE, train_instances_dir, budget, pcs_file, outdir)
    
    os.system(call)

def irace(budget: int, 
            algorithm: str,
            initial_parameters: dict,
            terminate: dict, 
            optimize: str,
            train_instances_dir: str):

    robjects.r('library("irace")')

    # define scenario
    robjects.r('scenario = list()')
    robjects.r('scenario$trainInstancesDir = \"' + train_instances_dir + '\"')
    robjects.r('scenario$trainInstancesFile = \"' + train_instances_dir + '/trainInstancesFile\"')
    robjects.r('scenario$maxExperiments = ' + str(budget))
    robjects.r('scenario$targetRunner = "/home/damian/Desktop/MA/macode/myproject/tuning_wrapper.py"')

    logfile = 'myproject/data/irace/' + ctun_fname(budget, algorithm, terminate, optimize) + '.Rdata'
    robjects.r('scenario$logFile = "%s"' % logfile)

    # parallelization
    robjects.r('scenario$parallel = 8')

    # disable scientific notation - conversion can cause clashess
    robjects.r('options(scipen=999)')

    # set validation with test instances
    robjects.r("scenario$testNbElites = 1")
    robjects.r("scenario$testIterationElites = 1")
    test_instances_dir = train_instances_dir + '/test'
    test_instances_file = test_instances_dir + '/testInstancesFile'
    robjects.r('scenario$testInstancesDir = \"' + test_instances_dir + '\"')
    robjects.r('scenario$testInstancesFile = \"' + test_instances_file + '\"')

    # parameter domains and termination condition
    fparameters = 'myproject/tuning-settings/irace-' + algorithm.lower() + '-parameters.txt'
    robjects.r('parameters = readParameters("%s")' % fparameters)
 
    # set termination conditions
    for key in terminate:
        boolparam = 'term_' + key
        valparam = boolparam + '_val'
        robjects.r('parameters$domain$' + boolparam + ' = \"True\"')
        robjects.r('parameters$domain$' + valparam + ' = ' + str(terminate[key]))
        initial_parameters[boolparam] = 'True' # part of technical requirement to use starting configuration
        initial_parameters[valparam] = terminate[key] # part of technical requirement to use starting configuration
    
    # set optimize parameter
    robjects.r('parameters$domain$optimize = \"' + optimize + '\"')

    # initial configuration. also include termination condition and optimization criterion
    initial_parameters['algorithm'] = algorithm
    initial_parameters['optimize'] = optimize
    finitial_configuration = ('myproject/tuning-settings/config-irace-' + algorithm.lower() + '-initial-parameters.txt')
    pd.DataFrame([initial_parameters]).to_csv(finitial_configuration, sep = '\t') # this is probably suboptimal, but its only a little suboptimal and it works, sooo
    robjects.r('scenario$configurationsFile = "%s"' % finitial_configuration)
    
    # actually run irace
    robjects.r('results = irace(scenario = scenario, parameters = parameters)')

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
    elif algorithm == 'GA':
        result = ga(instance = instance, cfg = cfg, terminate = terminate, fname_convdata = None)

    if tuner == 'irace':
        print(result[optimize])
    elif tuner == 'SMAC':
        print('Result of this algorithm run: %s, %f, %i, %f, %i, %s' % ('SUCCESS', result['time'], result['evals'], result[optimize], 1, 0) )
    
    sys.exit(0)