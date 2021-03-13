# create one data profile data profiles (success rate in relation to quality deviation) 
# with one line for every metaheuristic - tuner combination

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.lines as mlines

from myproject.helpers import cmhrun_fname, mhrun_fname_from_tuning_data
from myproject.helpers import BASE_TERM, DEF_CFGS
from myproject.analyze.common_definitions import COLORS_METAHEURISTICS, STYLES_TUNERS

legend_handles = [
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['SA'], label='SA'),
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['ACO'], label='ACO'),
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['GA'], label='GA'),
    mlines.Line2D([], [], linestyle=STYLES_TUNERS['notuner'], color = 'black', label='untuned'),
    mlines.Line2D([], [], linestyle=STYLES_TUNERS['irace'], color = 'black', label='irace'),
    mlines.Line2D([], [], linestyle=STYLES_TUNERS['smac'], color = 'black', label='SMAC')
]

def plot(mhrun_results: pd.DataFrame, ax, metaheuristic, tuner, optimize):
    mhrun_results = mhrun_results.sort_values(optimize)
    metric_results_unique = mhrun_results[optimize].unique()

    successes = [len(mhrun_results.loc[mhrun_results[optimize] <= res]) for res in metric_results_unique]
    success_rates = np.array(successes) / len(mhrun_results)
    ax.plot(metric_results_unique, success_rates, color = COLORS_METAHEURISTICS[metaheuristic], linestyle = STYLES_TUNERS[tuner], alpha = 0.7)

tuning_budget = 5000
tuners = ['irace', 'smac']
metaheuristics = ['SA', 'ACO', 'GA']
termination_conditions = [
    {'qualdev': 0.0, 'evals': 1000},
    {'qualdev': 0.0, 'evals': 10000},
    {'qualdev': 0.0, 'evals': 100000},
]
optimize = 'qualdev'

fig, ax = plt.subplots(nrows = 3, ncols = 1, sharey = 'all', sharex = 'all', figsize=(9.6, 15.57))

for i, terminate in enumerate(termination_conditions):
    for metaheuristic in metaheuristics:
        # plot untuned metaheuristic
        print("Plot " + metaheuristic + ' without tuning ')
        mhrun_fname = cmhrun_fname(metaheuristic, DEF_CFGS[metaheuristic], terminate)
        mhrun_results = pd.read_csv('myproject/data/results/' + mhrun_fname + '.csv')
        plot(mhrun_results, ax[i], metaheuristic, 'notuner', optimize)

        for tuner in tuners:
            print("Plot " + metaheuristic + ' tuned with ' + tuner)
            mhrun_fname = mhrun_fname_from_tuning_data(terminate, tuner, metaheuristic, optimize)
            mhrun_results = pd.read_csv('myproject/data/results/' + mhrun_fname + '.csv')
            plot(mhrun_results, ax[i], metaheuristic, tuner, optimize)

    ax[i].set(xscale = 'log', xlim = [0.0001, 5], ylim = [0, 1])
    ax[i].get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    # axis grid
    ax[i].grid()
        
    # title and axis labels
    ax[i].set(title = 'Termination after {:,d} evaluations'.format(terminate['evals']), xlabel = 'Quality Deviation', ylabel = 'Success Rate')

fig.legend(handles = legend_handles)
fig.tight_layout()
fig.savefig('myproject/data/figures/metaheuristics-data-profiles.png', bbox_inches='tight')