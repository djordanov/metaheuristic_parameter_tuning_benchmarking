import typing
from pathlib import Path
import random
import time
from operator import attrgetter, itemgetter
import heapq

import numpy as np
import pandas as pd 

import tsplib95
from myproject.metaheuristic.commons import Solution, Convdata

def displacement_mutation(problem: tsplib95.models.StandardProblem, tour: list) -> Solution:    
    substart = random.randint(0, len(tour)-1)
    subend = random.randint(substart+1, len(tour))

    subtour = tour[substart:subend]
    newtour = tour[:substart] + tour[subend:]
    addback = random.randint(0, len(newtour))
    newtour = newtour[:addback] + subtour + newtour[addback:]

    return Solution(problem.trace_tours([newtour])[0], newtour)

def neighnodes(tour: list, node: int) -> tuple:
    idx = tour.index(node)
    neighleft = tour[idx-1] if idx > 0 else tour[-1]
    neighright = tour[idx + 1] if idx + 1 < len(tour) else tour[0]
    return neighleft, neighright

def edge_recombination_crossover(problem: tsplib95.models.StandardProblem, parent1: list, parent2: list) -> Solution:
    
    # build adjacency matrix ...
    edge_map = {}
    for nnode in problem.get_nodes():
        neighs1 = neighnodes(parent1, nnode)
        neighs2 = neighnodes(parent2, nnode)
        edge_map[nnode] = { neighs1[0], neighs1[1], neighs2[0], neighs2[1] }

    # build new tour ...
    ntour = []
    nnode = random.choice([parent1[0], parent2[0]])
    while True:

        # add node, terminate if tour is finished, remove node from edge map
        ntour.append(nnode)
        if len(ntour) == len(parent1):
            break
        for candidates in edge_map.values():
            if nnode in candidates:
                candidates.remove(nnode)

        # select the next node
        candidates = edge_map.pop(nnode)
        if len(candidates) == 0:
            nnode = random.choice(list(set(parent1) - set(ntour))) # performance improvement here
        else:
            nnode = None
            for candidate in candidates:
                if not nnode or len(edge_map[candidate]) < len(edge_map[nnode]):
                    nnode = candidate

    return Solution(problem.trace_tours([ntour])[0], ntour)

def ga(instance: Path, cfg: dict, terminate: dict, fname_convdata: str):
    
    starttime = time.perf_counter()
    convdata = [] if fname_convdata != None else None

    # problem: tsplib95.models.StandardProblem
    problem: tsplib95.models.StandardProblem = tsplib95.load(instance.absolute())
    optimaltour: tsplib95.models.StandardProblem = tsplib95.load(instance.with_suffix('.opt.tour').absolute())
    optimal_quality: int = problem.trace_tours(optimaltour.tours)[0]

    # initialize population with random tours
    population = []
    for _ in range(cfg['popsize']):
        tour = list(problem.get_nodes())
        random.shuffle(tour)
        population.append(Solution(problem.trace_tours([tour])[0], tour))
    heapq._heapify_max(population)
    evals = len(population)
    iters_noimprovement = 0

    # initialize rank-based weights for selection
    mean = 10 * cfg['popsize']
    weights = np.linspace(mean*cfg['b'], 2 * mean - mean * cfg['b'], cfg['popsize'])
    cum_weights = weights.cumsum()

    # iterate over generations...
    while not ('evals' in terminate and evals >= terminate['evals'] \
        or 'qualdev' in terminate and heapq.nsmallest(1, population)[0].qual < optimal_quality * (1 + terminate['qualdev']) \
        or 'time' in terminate and time.perf_counter() - starttime > terminate['time']
        or 'noimprovement' in terminate and iters_noimprovement > terminate['noimprovement']['iterations']):
        
        # selection...
        parents = random.choices(population, cum_weights = cum_weights, k = 2)
        
        # recombination
        newsol = edge_recombination_crossover(problem, parents[0].tour, parents[1].tour)
        evals += 1

        # maybe mutation
        if random.random() < cfg['mut_rate']:
            newsol = displacement_mutation(problem, newsol.tour)
            evals += 11
        
        # add to population
        worst = heapq.nlargest(1, population)[0]
        if newsol < worst:
            heapq._heapreplace_max(population, newsol)
            iters_noimprovement = 0
        else: 
            iters_noimprovement += 1
            
    if convdata != None:
        print_headers: bool = False if fname_convdata.exists() else True
        pd.DataFrame(convdata).to_csv(fname_convdata.absolute(),  index = None, mode = 'a', header = print_headers)

    return {'qualdev': (heapq.nsmallest(1, population)[0].qual - optimal_quality) / optimal_quality, 'evals': evals, 'time': time.perf_counter() - starttime}