import itertools

import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import ticker

from myproject.analyze.common_definitions import COLORS_METAHEURISTICS, STYLES_TUNERS

from myproject.helpers import cmhrun_fname
from myproject.helpers import get_tuned_convergence_data
from myproject.helpers import BASE_TERM, DEF_CFGS

legend_handles = [
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['SA'], label='SA', linewidth = 3),
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['ACO'], label='ACO', linewidth = 3),
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['GA'], label='GA', linewidth = 3),
    mlines.Line2D([], [], label='untuned', linewidth = 3),
    mlines.Line2D([], [], linestyle=STYLES_TUNERS['irace'], label='irace', linewidth = 3),
    mlines.Line2D([], [], linestyle=STYLES_TUNERS['smac'], label='SMAC', linewidth = 3)
]

tuners = ['irace', 'smac']
metaheuristics = ['SA', 'ACO', 'GA']
levals = [0, 1000, 10000, 100000]

def plot_data(axes, df: pd.DataFrame, tuner: str, algorithm: str, ):
    grouped_by_evals = df.groupby('evals')
    conv = grouped_by_evals.mean()
    label = '{}'.format(algorithm) if tuner == None else '{} + {}'.format(algorithm, tuner)
    if tuner != None:
        axes.plot(conv.index, conv.qualdev, label = label, color = COLORS_METAHEURISTICS[algorithm], linestyle = STYLES_TUNERS[tuner], alpha = 0.7, linewidth = 3)
    else:
        axes.plot(conv.index, conv.qualdev, label = label, color = COLORS_METAHEURISTICS[algorithm], alpha = 0.7, linewidth = 3)   

fig, ax = plt.subplots(nrows = len(levals), ncols = 1, sharey = 'all', sharex = 'all', figsize=(9.6, 15.57))

for i, evals in enumerate(levals):    
    # plot data
    # default convergence without tuning
    title = 'Convergence when tuning to {:,} evaluations'.format(evals)
    if evals == 0:
        for metaheuristic in metaheuristics:
            title = 'Convergence with default parameters'.format(evals)
            mhrun_fname = cmhrun_fname(algorithm = metaheuristic, config = DEF_CFGS[metaheuristic], terminate = BASE_TERM, optimize = 'qualdev')
            df = pd.read_csv('myproject/data/conv/' + mhrun_fname + '.csv')
            plot_data(ax[i], df, None, metaheuristic)

    else:
        for tuner_mh in itertools.product(tuners, metaheuristics):
            df = get_tuned_convergence_data({'qualdev': 0.0, 'evals': evals}, tuner_mh[0], tuner_mh[1], 'qualdev')
            plot_data(ax[i], df, tuner_mh[0], tuner_mh[1])
        
    # visual stuff...

    # axis scales
    ax[i].set(xscale = 'log', yscale = 'log', yticks = [0.01, 0.1, 0.5, 1, 2], ylim = [0.005, 5])
    ax[i].get_xaxis().set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax[i].get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    # grid and title
    ax[i].grid()
    ax[i].set_title(title, fontsize = 12)
    
    
# axis labels
ax[len(levals) - 1].set(xlabel = 'Evals', ylabel = 'Quality Deviation')

# create legend
fig.legend(handles = legend_handles)

# save figure
fig.tight_layout()
fig.savefig('myproject/data/figures/convergence/all-evals.png'.format(evals), bbox_inches='tight')