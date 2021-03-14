## Dependencies ##

* Python packages in requirements.txt
* R and the R package irace (https://github.com/MLopez-Ibanez/irace)
* SMAC (http://www.cs.ubc.ca/labs/beta/Projects/SMAC/). The path to the smac executable needs to be defined in myproject/tuning_wrapper.py
* Concorde (http://www.math.uwaterloo.ca/tsp/concorde/index.html) and pyconcorde (https://github.com/jvkersch/pyconcorde/) for solving of test instances.

## Code Structure ##

* Problem instances are in myproject/instances/
* Re-implemented metaheuristics are in myproject/metaheuristic
* The external tuners (irace and smac) are wrapped and called via functions in myproject/tuning_wrapper. The tuners themselves call myproject/tuning_wrapper as a script, which then wraps the metaheuristic to be tuned 
* Functions to help run the metaheuristics are in myproject/run. They- and the tuners are executed via ad-hoc scripts in myproject/run,
* myproject/helpers contains helper functions to create consistent result file names, transform parameter configurations from format to format and process result data.


* Parameter spaces are defined in myproject/tuning-settings. They are partly rewritten on the fly for technical reasons.
* Data analyzation- and diagram generation scripts are in myproject/analyze.
* Data is saved into a folder structure in myproject/data. Irace result files are saved into myproject/data/irace, smac files into myproject/data/smac, the metaheuristic convergence data into myproject/data/conv and the end results of a tuning run into myproject/data/results
