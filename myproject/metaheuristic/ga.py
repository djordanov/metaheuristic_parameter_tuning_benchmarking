import typing
from pathlib import Path
import random
import time
from operator import attrgetter, itemgetter

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

    return Solution(newtour, problem.trace_tours([newtour])[0])

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

    return Solution(ntour, problem.trace_tours([ntour])[0])

def order_crossover(problem: tsplib95.models.StandardProblem, parent1: list, parent2: list) -> Solution: 

    startp = random.randint(0, len(parent1)-1)
    endp = random.randint(startp, len(parent1))
    part_parent1 = parent1[startp:endp].copy()

    ntour = [None] * startp + part_parent1 + [None] * (endp - startp)

    point_ntour = endp
    point_parent2 = endp
    while not point_ntour == startp:
        if point_ntour == len(ntour):
            point_ntour = 0
        if point_parent2 == len(parent2):
            point_parent2 = 0

        city = parent2[point_parent2]
        if city not in ntour:
            ntour[point_ntour] = city
            point_ntour += 1
        point_parent2 += 1

    return Solution(ntour, problem.trace_tours(ntour)[0])

def tournament(participants: typing.List[Solution]) -> Solution:

    # completely unstochastic
    return min(participants, key = attrgetter('qual'))

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
        population.append(Solution(tour, problem.trace_tours([tour])[0]))
    evals = len(population)
    best = min(population, key = attrgetter('qual'))

    # iterate over generations...
    while not ('evals' in terminate and evals >= terminate['evals'] \
        or 'qualdev' in terminate and best.qual < optimal_quality * (1 + terminate['qualdev']) \
        or 'time' in terminate and time.perf_counter() - starttime > terminate['time']
        or 'noimprovement' in terminate and len(set([individual.qual for individual in population])) == 1):
        
        # mutation
        mutcount = sum([random.random() < cfg['mut_rate'] for _ in range(cfg['popsize'])])
        idx_mut_sols = random.sample(list(range(len(population))), mutcount)
        for idx_mut_sol in idx_mut_sols:
            population[idx_mut_sol] = displacement_mutation(problem, population[idx_mut_sol].tour)
        evals += len(idx_mut_sols)

        # select two individuals using two tournaments
        preselection = random.sample(population, k = cfg['tourn_size'] * 2)
        seltourn1 = preselection[:cfg['tourn_size']]
        seltourn2 = preselection[cfg['tourn_size']:]
        parent1 = tournament(seltourn1)
        parent2 = tournament(seltourn2)
        
        # recombination
        newsol = edge_recombination_crossover(problem, parent1.tour, parent2.tour)
        evals += 1
        
        # add to population
        worst = max(population, key = attrgetter('qual'))
        if newsol.qual > worst.qual:
            continue     
        population.remove(worst)
        population.append(newsol)
        worst = max(population, key = attrgetter('qual'))

        if newsol.qual < best.qual:
            best = newsol        
            
    if convdata != None:
        print_headers: bool = False if fname_convdata.exists() else True
        pd.DataFrame(convdata).to_csv(fname_convdata.absolute(),  index = None, mode = 'a', header = print_headers)

    return {'qualdev': (best.qual - optimal_quality) / optimal_quality, 'evals': evals, 'time': time.perf_counter() - starttime}