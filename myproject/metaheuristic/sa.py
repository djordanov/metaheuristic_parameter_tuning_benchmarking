import logging

from pathlib import Path
import math
import random
import time
import numpy as np
import pandas as pd

import tsplib95

from myproject.metaheuristic.commons import Convdata, random_n2opt

# logging.basicConfig(level=logging.DEBUG) # logging

def accept(current_quality: float, neighbor_quality: float, current_temperature: float) -> bool:
    if neighbor_quality <= current_quality:
        return True
            
    probability = math.e**((current_quality - neighbor_quality)/current_temperature)
    if random.random() < probability:
        return True

    return False

def sa( instance: Path, 
        cfg: dict, 
        terminate: dict,
        fname_convdata: Path = None) -> dict:
    starttime = time.perf_counter()

    convdata = [] if fname_convdata != None else None

    # problem: tsplib95.models.StandardProblem
    problem: tsplib95.models.StandardProblem = tsplib95.load(instance)
    optimaltour: tsplib95.models.StandardProblem = tsplib95.load(Path(instance).with_suffix('.opt.tour').absolute())
    optimal_quality: int = problem.trace_tours(optimaltour.tours)[0]

    # setup
    curtour = list(problem.get_nodes())
    random.shuffle(curtour)
    curqual = problem.trace_tours([curtour])[0]
    evals = 1
    posmoves = [(a, b) for a in range(len(curtour)) for b in range(a, len(curtour)) if abs(a-b) > 1 and not (a == 0 and b == len(curtour) - 1)]

    temperature = cfg['initial_temperature']
    bestqual = curqual
    count_accepted = 0
    count_temperatures_wo_improvement = 0

    while not ('evals' in terminate and evals >= terminate['evals'] \
                or 'qualdev' in terminate and bestqual < optimal_quality * (1 + terminate['qualdev']) \
                or 'time' in terminate and time.perf_counter() - starttime > terminate['time'] \
                or 'noimprovement' in terminate \
                    and count_temperatures_wo_improvement > terminate['noimprovement']['temperatures'] \
                    and count_accepted / evals < terminate['noimprovement']['accportion']):
        
        count_temperatures_wo_improvement += 1
        for _ in range(cfg['repetitions']):

            # get neighbor
            neighsol = random_n2opt(curtour, posmoves)
            neighqual = problem.trace_tours([neighsol])[0]
            evals += 1

            # accept neighbor if 
            if accept(curqual, neighqual, temperature):
                curtour = neighsol
                curqual = neighqual
                count_accepted += 1

                if curqual < bestqual:
                    bestqual = curqual
                    count_temperatures_wo_improvement = 0    
            
            if convdata != None:
                qualdev = (bestqual - optimal_quality) / optimal_quality
                convdata.append(Convdata(instance.name, qualdev, evals, time.perf_counter() - starttime))

        # cool down
        temperature *= cfg['cooling_factor']
        temperature = max(temperature, 0.00001) # avoid rounding errors
    if convdata != None:
        print_headers: bool = False if fname_convdata.exists() else True
        pd.DataFrame(convdata).to_csv(fname_convdata.absolute(),  index = None, mode = 'a', header = print_headers)

    return {'qualdev': (bestqual - optimal_quality) / optimal_quality, 'evals': evals, 'time': time.perf_counter() - starttime}
