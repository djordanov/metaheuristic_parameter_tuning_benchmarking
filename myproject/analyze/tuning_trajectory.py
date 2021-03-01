import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.lines as mlines

from myproject.analyze.common_definitions import COLORS_METAHEURISTICS, STYLES_TUNERS, MARKERS_TUNEDTO
from myproject.helpers import tun_traj

legend_handles = [
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['SA'], label='SA'),
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['ACO'], label='ACO'),
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['GA'], label='GA'),
    mlines.Line2D([], [], linestyle=STYLES_TUNERS['irace'], label='irace'),
    mlines.Line2D([], [], linestyle=STYLES_TUNERS['smac'], label='SMAC')
]

termination_conditions = [
    {'qualdev': 0.0, 'evals': 1000}, 
    {'qualdev': 0.0, 'evals': 10000}, 
    {'qualdev': 0.0, 'evals': 100000}
]
optimize = 'qualdev'
algorithms = ['SA', 'ACO', 'GA']

fig, ax = plt.subplots(nrows = 3, ncols = 1, sharey = 'all', sharex = 'all', figsize=(9.6, 14.57))

for i, terminate in enumerate(termination_conditions):

    for algorithm in algorithms:
        for tuner in ['irace', 'smac']:
            traj = tun_traj(tuner = tuner, tuning_budget = 5000, algorithm = algorithm, terminate = terminate, optimize = optimize)
            ax[i].plot(traj['runs'], traj[optimize], label = '{} tuned by {}'.format(algorithm, tuner), 
                color = COLORS_METAHEURISTICS[algorithm], linestyle = STYLES_TUNERS[tuner], alpha = 0.5)

    ax[i].set(yscale = 'log', yticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 10], ylim = [0.01, 10])
    ax[i].get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax[i].get_xaxis().set_major_formatter(ticker.StrMethodFormatter('{x:,}'))

    # axis grid
    ax[i].grid()

    # title
    ax[i].set_title('Tuning convergence when tuning to {:,} evaluations'.format(terminate['evals']), fontsize = 12)
    
# x axis label and legend
ax[len(ax) - 1].set(xlabel = 'Experiments', ylabel = 'Quality Deviation Improvement')
fig.legend(handles = legend_handles)

fig.tight_layout()
fig.savefig('myproject/data/figures/tuning-convergence/together-evals.png', bbox_inches='tight')