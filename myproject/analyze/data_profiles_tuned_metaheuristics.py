# create one data profile data profiles (success rate in relation to quality deviation) 
# with one line for every metaheuristic - tuner combination

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.lines as mlines

from myproject.helpers import mhrun_fname_from_tuning_data
from myproject.helpers import BASE_TERM, DEF_CFGS
from myproject.analyze.common_definitions import COLORS_METAHEURISTICS, STYLES_TUNERS

legend_handles = [
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['SA'], label='SA'),
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['ACO'], label='ACO'),
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['GA'], label='GA'),
    mlines.Line2D([], [], linestyle=STYLES_TUNERS['irace'], color = 'black', label='irace'),
    mlines.Line2D([], [], linestyle=STYLES_TUNERS['smac'], color = 'black', label='SMAC')
]

tuning_budget = 5000
tuners = ['irace', 'smac']
metaheuristics = ['SA', 'ACO', 'GA']
terminate = BASE_TERM
optimize = 'qualdev'

fig, ax = plt.subplots()

for tuner in tuners:
    for metaheuristic in metaheuristics:
        print("Plot " + metaheuristic + ' tuned with ' + tuner)
        mhrun_fname = mhrun_fname_from_tuning_data(terminate, tuner, metaheuristic, optimize)
        mhrun_results = pd.read_csv('myproject/data/results/' + mhrun_fname + '.csv')
        mhrun_results = mhrun_results.sort_values(optimize)

        metric_results_unique = mhrun_results[optimize].unique()
        successes = [len(mhrun_results.loc[mhrun_results[optimize] <= res]) for res in metric_results_unique]
        success_rates = np.array(successes) / len(mhrun_results)
        ax.plot(metric_results_unique, success_rates, label = '{} tuned by {}'.format(metaheuristic, tuner), 
            color = COLORS_METAHEURISTICS[metaheuristic], linestyle = STYLES_TUNERS[tuner], alpha = 0.7)

ax.set(xscale = 'log', xlim = [0.0001, 1], ylim = [0, 1])
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
# ax.get_xaxis().set_major_formatter(ticker.StrMethodFormatter('{x:,}'))

# axis grid
ax.grid()
    
# x axis label and legend
ax.set(xlabel = 'Quality Deviation', ylabel = 'Success Rate')
fig.legend(handles = legend_handles)

fig.tight_layout()
fig.savefig('myproject/data/figures/data-profiles.png', bbox_inches='tight')