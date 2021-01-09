import tsplib95
import random

def ileft(tour: list, index: int) -> int:
    if index == 0:
        return len(tour) - 1
    return index - 1

def iright(tour: list, index: int) -> int:
    if index == len(tour) - 1:
        return 0
    return index + 1

def distleft(tour: list, ifrom: int, ito: int) -> int:
    if ito < ifrom:
        return ifrom - ito
    
    return ifrom + len(tour) - ito

def distright(tour: list, ifrom: int, ito: int) -> int:
    if ifrom < ito:
        return ito - ifrom 
    
    return len(tour) - ifrom + ito

def n2opt(tour: list, i1: int, i2: int, i3: int) -> list: # i1 to i3 are indizes of the positions of the nodes

    # imagine the tour as a circle, i1 and i2 are two neighbors at the top, i3 and i4 two neighbors at the bottom
    # the edges between i1 & i2 and i3 & i4 are removed, edges between i1 & i3 and i2 & i4 are created
    # the new edges cross, forming an x. So that the tour becomes a new tour instead of just two half circles
    # the tour is called clockwise if i1 at the top left, i2 top right, i3 bottom left and i4 bottom right
    # the tour is called counterclockwise it i1 is top right, i2 top left, i3 bottom left and i4 bottom right

    # a move to the 'right' increases the index, until looping around to 0 again
    # a move to the 'left' decreases the index, until looping around to the highest again

    direction = 'clockwise' if i2 == iright(tour, i1) else 'counterclockwise'
    i4 = iright(tour, i3) if direction == 'clockwise' else ileft(tour, i3)

    # start at i1, go to i3, from i3 clockwise/ counterclockwise to i2, go to i4, from i4 clockwise/ counterclockwise to i1 ...
    
    # wire i1 to i3, from i3 to i2
    ntour = [tour[i1], tour[i3]]
    inext = ileft if direction == 'clockwise' else iright
    idx = inext(tour, i3)
    while idx != i1:
        ntour.append(tour[idx])
        idx = inext(tour, idx)

    # ok, i2 reached. wire to i4, then continue back to i1
    ntour.append(tour[i4])
    inext = iright if direction == 'clockwise' else ileft
    idx = inext(tour, i4)
    while idx != i1:
        ntour.append(tour[idx])
        idx = inext(tour, idx)
        
    return ntour

def random_n2opt(tour: list, candidate_matrix: dict = None) -> list:
    # n1
    n1 = random.choice(tour)
    i1 = tour.index(n1)
    
    # n2 
    i2 = random.choice([ileft(tour, i1), iright(tour, i1)])
    
    # n3
    # candidates = candidate_matrix[n1].copy()
    candidates = tour.copy() # for testing only
    candidates.remove(n1) # for testing only
    candidates.remove(tour[ileft(tour, i1)])
    candidates.remove(tour[iright(tour, i1)])
    n3 = random.choice(candidates)

    return n2opt(tour, i1, i2, tour.index(n3))

def iterimprov_first_2opt(problem: tsplib95.models.StandardProblem, 
                        inittour: list, 
                        initqual: float, 
                        minqual: float,
                        maxevals: int) -> tuple:
    # setup
    curtour = inittour
    curqual = initqual
    evals = 0

    while evals < maxevals and minqual < curqual:
        neighbor = random_n2opt(curtour)
        neighqual = problem.trace_tours([neighbor])[0]
        evals += 1
        if neighqual < curqual:
            curtour = neighbor
            curqual = neighqual
    
    return curtour, curqual, evals

def create_neighbors(tour: list) -> list: # add in candidate lists
    neighbors = []
    for i1 in range(len(tour)):
        i2l = ileft(tour, i1)
        i2r = iright(tour, i1)
        for i3 in set(range(len(tour))) - set([i1, i2l, i2r]):
            neighbors.append(n2opt(tour, i1, i2l, i3))
            neighbors.append(n2opt(tour, i1, i2r, i3))

    return neighbors

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
        neighbors = create_neighbors(curtour)
        neighquals = problem.trace_tours(neighbors)
        evals += len(neighbors)            
        min_neighqual = min(neighquals)

        if min_neighqual < curqual:
            curqual = min_neighqual
            curtour = neighbors[neighquals.index(curqual)]
        else:
            return curtour, curqual, evals
    return curtour, curqual, evals

def n2opt2(tour: list, idx1: int, idx3: int) -> list:

    # decide which segment to reverse
    revmiddle = True if abs(idx1 - idx3) < (len(tour) / 2) else False

    if revmiddle:
        return tour[:idx1+1] + tour[idx3:idx1:-1] + tour[idx3+1:]
    else:
        if idx3 == len(tour) - 1: # in case of crossover from last to first 
            return [tour[0]] + tour[idx1+1:idx3+1] + tour[1:idx1+1] 
        return tour[idx3+1:] + tour[idx1+1:idx3+1] + tour[:idx1+1] # general case
        

tour = list(range(1, 6))
posmoves = [(a, b) for a in range(len(tour)) for b in range(a, len(tour)) if abs(a-b) > 1 and not (a == 0 and b == len(tour) - 1)]
neighbors = [n2opt2(tour, posmove[0], posmove[1]) for posmove in posmoves]
print(neighbors)

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