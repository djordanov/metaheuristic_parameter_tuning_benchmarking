import tsplib95
import random

import numpy as np
from operator import attrgetter, itemgetter
from collections import namedtuple

Convdata = namedtuple('Convdata', 'instance qualdev evals time')
Solution = namedtuple('Solution', 'qual tour')

def create_candidates_list(problem: tsplib95.models.StandardProblem, cand_list_size: int) -> tuple:
    candidates_list = {}
    
    for nfrom in problem.get_nodes():
        distances_nto = [(problem.get_weight(nfrom, nto), nto) for nto in problem.get_nodes()]
        distances_nto.sort()
        candidates = np.array(distances_nto[1:cand_list_size+1])[:,1]
        candidates_list[nfrom] = set(candidates)

    return candidates_list

def get_posmoves_nodes(candidate_list: list) -> list: # converts candidate list into list of tuples of allowed (nfrom, nto) pairings
    posmoves_nodes = []
    for nfrom in candidate_list:
        posmoves_nodes += [(nfrom, nto) for nto in candidate_list[nfrom]]
    return posmoves_nodes

def n2opt(tour: list, idx1: int, idx3: int) -> list:

    # decide which segment to reverse
    revmiddle = True if abs(idx1 - idx3) < (len(tour) / 2) else False

    if revmiddle:
        return tour[:idx1+1] + tour[idx3:idx1:-1] + tour[idx3+1:]
    else:
        if idx3 == len(tour) - 1: # in case of overflow from last to first 
            return [tour[0]] + tour[idx1+1:idx3+1] + tour[1:idx1+1] 
        return tour[idx3+1:] + tour[idx1+1:idx3+1] + tour[:idx1+1] # general case

def random_n2opt(tour: list, posmoves_idxs: list) -> list:    
    move = random.choice(posmoves_idxs)
    return n2opt(tour, move[0], move[1])

# improves a given tour via 2-opt iterative improvement local search, returns (local optimum, quality, evals required) tuple
def iterimprov_2opt(problem: tsplib95.models.StandardProblem, 
                        initsol: Solution,
                        minqual: float,
                        maxevals: int,
                        mode: str) -> tuple:

    # setup
    cursol = Solution(initsol.qual, initsol.tour)
    dim = len(cursol.tour)
    evals = 0
    posmoves = [(a, b) for a in range(dim) for b in range(a, dim) if abs(a-b) > 1 and not (a == 0 and b == dim - 1)]

    while evals < maxevals and minqual < cursol.qual:
        if mode == 'best':
            neighsols = []
            for move in posmoves:
                neightour = n2opt(cursol.tour, move[0], move[1])
                neighsols.append(Solution(problem.trace_tours([neightour])[0], neightour))    
            evals += len(neighsols)
            bestneigh = min(neighsols, key = attrgetter('qual'))

            if bestneigh.qual < cursol.qual:
                cursol = bestneigh
            else:
                return cursol, evals

        elif mode == 'first':
            rnd_posmoves = posmoves.copy()
            random.shuffle(rnd_posmoves)

            found = False
            for move in rnd_posmoves:
                neighbor = n2opt(cursol.tour, move[0], move[1])
                neighqual = problem.trace_tours([neighbor])[0]
                evals += 1
                if neighqual < cursol.qual:
                    cursol = Solution(neighqual, neighbor)
                    found = True
                    break
            if not found:
                return cursol, evals
        else:
            print('No iterative improvement procedure for: \"' + mode + '\"')

    return cursol, evals