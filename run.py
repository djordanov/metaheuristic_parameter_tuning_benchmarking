# pylint: disable=no-member

import rpy2.robjects.packages as rpackages
import pandas as pd

class Results:

    # static information
    runid: str = None

    # actual run results
    instances: list = []
    qualities: list = []
    evalcounts: list = []
    timestamps: list = []

    def __init__(self, runid):
        self.runid = runid

    def save(self):
        df: pd.DataFrame = pd.DataFrame({
                'instances': self.instances, 
                'qualities': self.qualities, 
                'evalcounts': self.evalcounts, 
                'timestamps': self.timestamps
        })
        df.to_csv(self.runid, index = False)

# run simulated annealing on all training- and test problems...
from pathlib import Path
import random
import tsplib95
from metaheuristics import Eval, sa

results = Results('baserun')
instancefolder = 'instances/20nodes'
entries = Path(instancefolder)

for _ in range(1):
    for entry in entries.iterdir():

        if entry.suffix != '.tsp':
                continue

        problem = tsplib95.load(entry)
        eval = Eval(problem)
        optimaltour = tsplib95.load(entry.with_suffix('.opt.tour').absolute())
        optimal_quality = problem.trace_tours(optimaltour.tours)[0]
        initial_solution = list(problem.get_nodes())
        random.shuffle(initial_solution)

        result = sa(eval, initial_solution, 200, 200, 0.8, {'evals': 100})
        quality_deviation = (result['quality'] - optimal_quality) / optimal_quality
        
        results.instances.append(entry.name)
        results.qualities.append(quality_deviation)
        results.evalcounts.append(result['countevals'])
        results.timestamps.append(result['timestamp'])

results.save()

# irace = rpackages.importr('irace')
# parameters = irace.readParameters('./tuning/sa-parameters.txt')
# scenario = irace.readScenario('./tuning/sa-scenario.txt')
# irace.checkIraceScenario(scenario = scenario, parameters = parameters)
# irace.irace(scenario = scenario, parameters = parameters)