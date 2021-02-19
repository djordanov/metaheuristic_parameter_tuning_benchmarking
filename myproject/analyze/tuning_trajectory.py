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
algorithms = ['SA', 'ACO', 'GA']

fig, ax = plt.subplots(nrows = len(termination_conditions), ncols = 1, sharex = True, sharey = True)

for i, terminate in enumerate(termination_conditions):
    for algorithm in algorithms:
        for tuner in ['irace', 'smac']:
            traj = tun_traj(tuner = tuner, tuning_budget = 5000, algorithm = algorithm, terminate = terminate, optimize = optimize)
            ax[i].plot(traj['runs'], traj[optimize], label = '{} tuned by {}'.format(algorithm, tuner), 
                color = LINE_COLORS[algorithm], linestyle = LINE_STYLES[tuner])

    # axes scales and ticks
    ax[i].set(xscale = 'log', xticks = [100, 1000, 5000], yscale = 'log', yticks = [0.1, 0.5, 1, 2, 10])
    ax[i].tick_params(labelsize = 5)
    ax[i].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax[i].get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    # axis grid and title
    ax[i].grid()
    ax[i].set_title('Metaheuristics tuned with termination after ' + str(terminate['evals']) + ' evals', size = 5, pad = 3)

# legend
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='right')

# axes labels
fig.text(0.5, 0.05, "Experiments", ha="center", va="center")
fig.text(0.05, 0.5, "{} as proportion of {} without tuning".format(optimize, optimize), ha="center", va="center", rotation=90)
fig.subplots_adjust(hspace=0.25)

fig.savefig('myproject/data/figures/tuning_trajectories.png', dpi = 2000)