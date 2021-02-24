import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

from myproject.analyze.common_definitions import LINE_COLORS, LINE_STYLES, MARKERS
from myproject.helpers import tun_traj
    
termination_conditions = [
    {'qualdev': 0, 'evals': 1000}, 
    {'qualdev': 0, 'evals': 10000}, 
    {'qualdev': 0, 'evals': 100000}
]
optimize = 'qualdev'
algorithms = ['SA', 'ACO']

for terminate in termination_conditions:

    fig, ax = plt.subplots(figsize=(5.5, 3.14))

    for algorithm in algorithms:
        for tuner in ['irace', 'smac']:
            traj = tun_traj(tuner = tuner, tuning_budget = 5000, algorithm = algorithm, terminate = terminate, optimize = optimize)
            ax.plot(traj['runs'], traj[optimize], label = '{} tuned by {}'.format(algorithm, tuner), 
                color = LINE_COLORS[algorithm], linestyle = LINE_STYLES[tuner], alpha = 0.5)

    ax.set(yscale = 'log', yticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 10], ylim = [0.01, 10])
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(labelsize = 8)

    # axis grid and labels
    ax.grid()
    ax.set(xlabel = 'Experiments', ylabel = 'Quality Deviation Improvement')

    # legend
    ax.legend()

    fig.tight_layout()
    fig.savefig('myproject/data/figures/tuning-convergence/{}-evals.png'.format(terminate['evals']), bbox_inches='tight')