import tsplib95
import random

def n2opt(tour: list, idx1: int, idx3: int) -> list:

    # decide which segment to reverse
    revmiddle = True if abs(idx1 - idx3) < (len(tour) / 2) else False

    if revmiddle:
        return tour[:idx1+1] + tour[idx3:idx1:-1] + tour[idx3+1:]
    else:
        if idx3 == len(tour) - 1: # in case of overflow from last to first 
            return [tour[0]] + tour[idx1+1:idx3+1] + tour[1:idx1+1] 
        return tour[idx3+1:] + tour[idx1+1:idx3+1] + tour[:idx1+1] # general case

def random_n2opt(tour: list, possible_moves: list) -> list:    
    move = random.choice(possible_moves)
    return n2opt(tour, tour.index(move[0]), tour.index(move[1]))

def iterimprov_first_2opt(problem: tsplib95.models.StandardProblem, 
                        inittour: list, 
                        initqual: float, 
                        minqual: float,
                        maxevals: int) -> tuple:
    # setup
    curtour = inittour
    curqual = initqual
    evals = 0
    posmoves = [(a, b) for a in range(len(inittour)) for b in range(a, len(inittour)) if abs(a-b) > 1 and not (a == 0 and b == len(inittour) - 1)]

    while evals < maxevals and minqual < curqual:
        neighbor = random_n2opt(curtour, posmoves)
        neighqual = problem.trace_tours([neighbor])[0]
        evals += 1
        if neighqual < curqual:
            curtour = neighbor
            curqual = neighqual
    
    return curtour, curqual, evals

def iterimprov_best_2opt(problem: tsplib95.models.StandardProblem, 
                        inittour: list, 
                        initqual: float, 
                        minqual: float,
                        maxevals: int) -> tuple:
    
    # setup
    curtour = inittour
    curqual = initqual
    evals = 0

    while evals < maxevals and minqual < curqual:
        posmoves = [(a, b) for a in range(len(inittour)) for b in range(a, len(inittour)) if abs(a-b) > 1 and not (a == 0 and b == len(inittour) - 1)]
        neighbors = [n2opt(curtour, move[0], move[1]) for move in posmoves]
        neighquals = problem.trace_tours(neighbors)
        evals += len(neighbors)            
        min_neighqual = min(neighquals)

        if min_neighqual < curqual:
            curqual = min_neighqual
            curtour = neighbors[neighquals.index(curqual)]
        else:
            return curtour, curqual, evals
    return curtour, curqual, evals

# improves a given tour via 2-opt iterative improvement local search, returns (local optimum, quality, evals required) tuple
def iterimprov_2opt(problem: tsplib95.models.StandardProblem, 
                        tour: list, 
                        quality: float, 
                        minqual: float,
                        maxevals: int,
                        mode: str) -> tuple:

    if mode == 'first':
        return iterimprov_first_2opt(problem, tour, quality, minqual, maxevals)
    elif mode == 'best':
        return iterimprov_best_2opt(problem, tour, quality, minqual, maxevals)
    else:
        print('No iterative improvement procedure for: \"' + mode + '\"')