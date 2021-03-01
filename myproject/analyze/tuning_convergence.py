import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.lines as mlines

from myproject.analyze.common_definitions import COLORS_METAHEURISTICS, STYLES_TUNERS
from myproject.helpers import tun_traj, cmhrun_fname
from myproject.helpers import DEF_CFGS, BASE_TERM

legend_handles = [
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['SA'], label='SA'),
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['ACO'], label='ACO'),
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['GA'], label='GA'),
    mlines.Line2D([], [], linestyle=STYLES_TUNERS['irace'], label='irace'),
    mlines.Line2D([], [], linestyle=STYLES_TUNERS['smac'], label='SMAC')
]

termination_conditions = {}
metaheuristics = ['SA', 'ACO', 'GA']
for metaheuristic in metaheuristics:
    mhrun_fname = cmhrun_fname(metaheuristic, DEF_CFGS[metaheuristic], BASE_TERM, 'qualdev')
    mhrun_results = pd.read_csv('myproject/data/results/' + mhrun_fname + '.csv')
    default_qualdev = round(mhrun_results.mean()['qualdev'], 15)
    termination_conditions[metaheuristic] = {'qualdev': default_qualdev, 'evals': 100000}
optimize = 'evals'

fig, ax = plt.subplots(nrows = 3, ncols = 1, sharey = 'all', sharex = 'all', figsize=(9.6, 14.57))

for i, metaheuristic in enumerate(metaheuristics):

    terminate = termination_conditions[metaheuristic]
    for tuner in ['irace', 'smac']:
        traj = tun_traj(tuner = tuner, tuning_budget = 1000, algorithm = metaheuristic, terminate = terminate, optimize = optimize)
        ax[i].plot(traj['runs'], traj[optimize], label = '{} tuned by {}'.format(metaheuristic, tuner), 
            color = COLORS_METAHEURISTICS[metaheuristic], linestyle = STYLES_TUNERS[tuner], alpha = 0.5)

    ax[i].set(yscale = 'log', yticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 10], ylim = [0.01, 10])
    ax[i].get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax[i].get_xaxis().set_major_formatter(ticker.StrMethodFormatter('{x:,}'))

    # axis grid
    ax[i].grid()

    # title
    ax[i].set_title('Tuning convergence when tuning to a quality deviation of {:.3} or 100,000 evaluations'.format(terminate['qualdev']), fontsize = 12)
    
# x axis label and legend
ax[len(ax) - 1].set(xlabel = 'Experiments', ylabel = 'Number of Evaluations Improvement')
fig.legend(handles = legend_handles)

fig.tight_layout()
fig.savefig('myproject/data/figures/tuning-convergence/together-evals.png', bbox_inches='tight')