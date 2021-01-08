import tsplib95
import random

def random_n2opt(solution: list) -> list:
    posmoves = [(a, b) for a in range(len(solution)) for b in range(a, len(solution)) if abs(a-b) > 1]
    permutation = random.choice(posmoves)
    return n2opt(solution, permutation[0], permutation[1])

def n2opt(solution: list, node1: int, node2: int) -> list:
     return solution[:node1+1] + solution[node1+1:node2+1][::-1] + solution[node2+1:]

# improves a given solution via 2-opt iterative improvement local search, returns (local optimum, quality, evals required) tuple
def iterimprov_2opt(problem: tsplib95.models.StandardProblem, 
                        solution: list, 
                        quality: float, 
                        mode: str,
                        minqual: float,
                        maxevals: int) -> tuple:

    # setup
    cursol = solution
    curqual = quality
    evals = 0

    while True:
        posmoves = [(a, b) for a in range(len(solution)) for b in range(a, len(solution)) if abs(a-b) > 1]
        
        if mode == 'first':
            random.shuffle(posmoves)
            movedone = False
            for move in posmoves:
                neighbor = n2opt(cursol, move[0], move[1])
                neighqual = problem.trace_tours([neighbor])[0]
                evals += 1
                if evals > maxevals:
                    return cursol, curqual, evals

                if neighqual < curqual:
                    movedone = True
                    cursol = neighbor
                    curqual = neighqual 

                    if curqual < minqual:
                        return cursol, curqual, evals
                    break

            if not movedone:
                return cursol, curqual, evals
        
        if mode == 'best':
            neighbors = [n2opt(cursol, posmove[0], posmove[1]) for posmove in posmoves]
            neighquals = problem.trace_tours(neighbors)
            evals += len(neighbors)            
            min_neighqual = min(neighquals)

            if min_neighqual < curqual:
                curqual = min_neighqual
                cursol = neighbors[neighquals.index(curqual)]

                if evals > maxevals or curqual < minqual:
                    return cursol, curqual, evals
            else:
                return cursol, curqual, evals