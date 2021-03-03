import itertools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from myproject.analyze.common_definitions import COLORS_TUNEDTO, MARKERS_METAHEURISTICS

from myproject.helpers import cmhrun_fname, get_tuned_convergence_data
from myproject.helpers import BASE_TERM, DEF_CFGS

legend_handles = [
    mpatches.Patch(color=COLORS_TUNEDTO[0], label='Untuned'),
    mpatches.Patch(color=COLORS_TUNEDTO[1000], label='Tuned to 1,000 Evals'),
    mpatches.Patch(color=COLORS_TUNEDTO[10000], label='Tuned to 10,000 Evals'),
    mpatches.Patch(color=COLORS_TUNEDTO[100000], label='Tuned to 100,000 Evals'),
    mlines.Line2D([], [], marker=MARKERS_METAHEURISTICS['SA'], linestyle = "None", markersize=10, color='black', label='SA'),
    mlines.Line2D([], [], marker=MARKERS_METAHEURISTICS['ACO'], linestyle = "None", markersize=10, color='black', label='ACO'),
    mlines.Line2D([], [], marker=MARKERS_METAHEURISTICS['GA'], linestyle = "None", markersize=10, color='black', label='GA')
]

tuners = ['irace', 'smac']
metaheuristics = ['SA', 'ACO', 'GA']
levals = [0, 1000, 10000, 100000]
optimize = 'qualdev'

# create axes
fig, ax = plt.subplots(figsize = (9.6, 9.6))

for constellation in itertools.product(levals, metaheuristics, tuners):

    print(constellation)
    conv = None
    if constellation[0] == 0:
        metaheuristic = constellation[1]
        mhrun_fname = cmhrun_fname(metaheuristic, DEF_CFGS[metaheuristic], BASE_TERM, optimize)
        conv = pd.read_csv('myproject/data/conv/' + mhrun_fname + '.csv')
    else: 
        conv = get_tuned_convergence_data(terminate = {'qualdev': 0.0, 'evals': constellation[0]}, 
            tuner = constellation[2], metaheuristic = constellation[1], optimize = optimize)
    qd_means = conv.groupby('evals').mean()

    # interpolate in case eval points were skipped over in metaheuristic
    all_eval_points = list(range(1, 100002))
    qualdevs = np.full((100001,), np.nan)
    qd_means = qd_means.loc[qd_means.index <= 100000]
    for evals in qd_means.index:
        qualdevs[evals] = qd_means.loc[evals][optimize]

    conv_refurbished = pd.DataFrame({'evals': all_eval_points, optimize: qualdevs})
    conv_refurbished[optimize] = conv_refurbished[optimize].pad()
    conv_refurbished = conv_refurbished.loc[conv_refurbished['evals'].isin(levals)]

    ax.scatter(conv_refurbished.evals, conv_refurbished[optimize], alpha = 0.7, s = 130, color = COLORS_TUNEDTO[constellation[0]], 
        marker = MARKERS_METAHEURISTICS[constellation[1]], linewidths = 1, edgecolors = 'black')

# visual stuff...

# axis scales
ax.set(xscale = 'log', yscale = 'log', yticks = [0.01, 0.1, 0.5, 1, 2], ylim = [0.005, 5])
ax.get_xaxis().set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

# grid and axes labels
ax.grid()
ax.set(xlabel = 'Evals', ylabel = 'Quality Deviation')

# legend
ax.legend(handles = legend_handles)

# save figure
fig.tight_layout()
fig.savefig('myproject/data/figures/qualdev_at_termevals.png', bbox_inches='tight')