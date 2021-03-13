# create one data profile data profiles (success rate in relation to quality deviation) 
# with one line for every metaheuristic - tuner combination

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.lines as mlines

from myproject.helpers import tun_conv
from myproject.analyze.common_definitions import COLORS_METAHEURISTICS, STYLES_TUNERS

legend_handles = [
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['SA'], label='SA'),
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['ACO'], label='ACO'),
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['GA'], label='GA'),
]

def plot(metric_results: pd.Series, ax, metaheuristic, tuner):
    metric_results = metric_results.sort_values()
    metric_results_unique = metric_results.unique()

    successes = [len(metric_results.loc[metric_results <= res]) for res in metric_results_unique]
    success_rates = np.array(successes) / len(metric_results)
    ax.plot(metric_results_unique, success_rates, color = COLORS_METAHEURISTICS[metaheuristic], linestyle = STYLES_TUNERS[tuner], linewidth = 3)

tuning_budget = 700
tuner = 'smac'
metaheuristics = ['SA', 'ACO', 'GA']
terminate = {'qualdev': 0.0, 'evals': 1000}
optimize = 'qualdev'
repetitions = 50

fix = 'runs'
cuts = [10, 25, 50, 75, 100, 150, 200, 300, 500, 700]

nrows = 5
ncols = 2
fig, ax = plt.subplots(nrows = nrows, ncols = ncols, sharey = 'all', sharex = 'all', figsize=(9.6, 15.57))

cut_idx = 0
for nrow in range(nrows):
    for ncol in range(ncols):
        cut = cuts[cut_idx]
        cut_idx += 1

        for metaheuristic in metaheuristics:
            tun_convs = pd.DataFrame({'runs': list(range(tuning_budget + 1))})
            for repetition in range(repetitions):
                conv = tun_conv(tuner = tuner, tuning_budget = tuning_budget, algorithm = metaheuristic, terminate = terminate, optimize = optimize, suffix = str(repetition))
                tun_convs[repetition] = conv[optimize]
            tun_convs = tun_convs.set_index('runs')

            metric_results = tun_convs.loc[round(cut)]
            plot(metric_results, ax[nrow, ncol], metaheuristic, tuner)

        ax[nrow, ncol].set(yticks = [0, 0.25, 0.5, 0.75, 1], ylim = [0, 1])
        ax[nrow, ncol].set(xscale = 'log', xticks = [0.03, 0.05, 0.1, 0.2, 0.5, 1], xlim = [0.03, 1.1])
        ax[nrow, ncol].get_xaxis().set_major_formatter(ticker.ScalarFormatter())

        # axis grid
        ax[nrow, ncol].grid()
            
        # x axis label and legend
        ax[nrow, ncol].set(title = '{} Experiments'.format(cut))
        ax[nrow, ncol].set(xlabel = 'Tuning Quality', ylabel = 'Success Rate')

fig.legend(handles = legend_handles)

fig.tight_layout()
fig.savefig('myproject/data/figures/data-profiles_tuning.png', bbox_inches='tight')