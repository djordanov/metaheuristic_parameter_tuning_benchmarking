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

import random
import tsplib95
import metaheuristics

## This a dummy example that shows how to parse the parameters defined in
## parameters.txt and does not need to call any other software.

if __name__=='__main__':
    if len(sys.argv) < 5:
        print("\nUsage: ./target-runner.py <configuration_id> <instance_id> <seed> <instance_path_name> <list of parameters>\n")
        sys.exit(1)

    # print("params: " + str(sys.argv[5:]))

    # Get the parameters as command line arguments.
    configuration_id = sys.argv[1]
    instance_id = sys.argv[2]
    seed = sys.argv[3]
    instance = sys.argv[4]
    cand_params = sys.argv[5:]

    # Set parameters
    problem: tsplib95.models.StandardProblem = tsplib95.load(instance)

    eval: metaheuristics.Eval = metaheuristics.Eval(problem)
    initial_solution = list(range(problem.dimension))
    random.shuffle(initial_solution)
    max_evals = 450 * problem.dimension

    # Tuned parameters
    initial_temperature = None
    repetitions = None
    cooling_factor = None
    
    # print("cand_params: " + str(cand_params))
    while len(cand_params) > 1:
        # Get and remove first and second elements.
        param = cand_params.pop(0)
        value = cand_params.pop(0)
        if param == "--initial_temperature":
            initial_temperature = float(value)
        elif param == "--repetitions":
            repetitions = int(value)
        elif param == "--cooling_factor":
            cooling_factor = float(value)
        else:
            target_runner_error("unknown parameter %s" % (param))
    
    # Run runner
    quality = metaheuristics.sa(eval, initial_solution, initial_temperature, repetitions, cooling_factor, max_evals)
    print(quality)
    
    sys.exit(0)

# Useful function to print errors.
def target_runner_error(msg):
    now = datetime.datetime.now()
    print(str(now) + " error: " + msg)
    sys.exit(1)

