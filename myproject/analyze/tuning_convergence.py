import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.lines as mlines

from myproject.analyze.common_definitions import COLORS_METAHEURISTICS, STYLES_TUNERS
from myproject.helpers import tun_traj, cmhrun_fname
from myproject.helpers import DEF_CFGS, BASE_TERM

legend_handles = [
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['SA'], label='SA', linewidth = 3),
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['ACO'], label='ACO', linewidth = 3),
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['GA'], label='GA', linewidth = 3),
    mlines.Line2D([], [], linestyle=STYLES_TUNERS['irace'], color = 'black', label='irace', linewidth = 3),
    mlines.Line2D([], [], linestyle=STYLES_TUNERS['smac'], color = 'black', label='SMAC', linewidth = 3)
]

termination_conditions = [
    {'qualdev': 0.0, 'evals': 1000},
    {'qualdev': 0.0, 'evals': 10000},
    {'qualdev': 0.0, 'evals': 100000}
]
metaheuristics = ['SA', 'ACO', 'GA']
optimize = 'qualdev'
tuning_budget = 5000

fig, ax = plt.subplots(nrows = 3, ncols = 1, sharey = 'all', sharex = 'all', figsize=(9.6, 14.57))

for i, terminate in enumerate(termination_conditions):
    for metaheuristic in metaheuristics: 
        for tuner in ['irace', 'smac']:
            traj = tun_traj(tuner = tuner, tuning_budget = tuning_budget, algorithm = metaheuristic, terminate = terminate, optimize = optimize)
            ax[i].plot(traj['runs'], traj[optimize], label = '{} tuned by {}'.format(metaheuristic, tuner), 
                color = COLORS_METAHEURISTICS[metaheuristic], linestyle = STYLES_TUNERS[tuner], alpha = 0.7, linewidth = 3)

        ax[i].set(yscale = 'log', yticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 10], ylim = [0.01, 10])
        ax[i].get_yaxis().set_major_formatter(ticker.ScalarFormatter())
        ax[i].get_xaxis().set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax[i].get_xaxis().set_minor_locator(ticker.AutoMinorLocator())

        # axis grid
        ax[i].grid()

        # title
        ax[i].set_title('Tuning convergence when tuning to {:,.0f} evaluations'.format(terminate['evals']), fontsize = 12)
    
# x axis label and legend
ax[len(ax) - 1].set(xlabel = 'Experiments', ylabel = 'Number of Evaluations Improvement')
fig.legend(handles = legend_handles)

fig.tight_layout()
fig.savefig('myproject/data/figures/tuning-convergence/together-evals.png', bbox_inches='tight')