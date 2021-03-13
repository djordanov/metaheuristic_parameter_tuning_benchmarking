import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.lines as mlines

from myproject.analyze.common_definitions import COLORS_METAHEURISTICS, STYLES_TUNERS
from myproject.helpers import tun_conv, cmhrun_fname
from myproject.helpers import DEF_CFGS, BASE_TERM

legend_handles = [
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['SA'], label='SA', linewidth = 3),
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['ACO'], label='ACO', linewidth = 3),
    mlines.Line2D([], [], color=COLORS_METAHEURISTICS['GA'], label='GA', linewidth = 3),
    mlines.Line2D([], [], linestyle=STYLES_TUNERS['irace'], color = 'black', label='irace', linewidth = 3),
    mlines.Line2D([], [], linestyle=STYLES_TUNERS['smac'], color = 'black', label='SMAC', linewidth = 3)
]

tuners = ['smac', 'irace']
metaheuristics = ['SA', 'ACO', 'GA']
optimize = 'evals'
tuning_budget = 700
qd_term_factors = [3, 2, 1]

fig, ax = plt.subplots(nrows = 3, ncols = 1, sharey = 'all', sharex = 'all', figsize=(9.6, 15.57))

for i, qd_term_factor in enumerate(qd_term_factors):
    for metaheuristic in metaheuristics:
        mhrun_fname = cmhrun_fname(metaheuristic, DEF_CFGS[metaheuristic], BASE_TERM)
        mhrun_results = pd.read_csv('myproject/data/results/' + mhrun_fname + '.csv')
        default_qualdev = round(mhrun_results.mean()['qualdev'], 15) * qd_term_factor
        terminate = {'qualdev': default_qualdev, 'evals': 100000}

        for tuner in ['irace', 'smac']:
            conv = tun_conv(tuner = tuner, tuning_budget = tuning_budget, algorithm = metaheuristic, terminate = terminate, optimize = optimize)
            ax[i].plot(conv['runs'], conv[optimize], label = '{} tuned by {}'.format(metaheuristic, tuner), 
                color = COLORS_METAHEURISTICS[metaheuristic], linestyle = STYLES_TUNERS[tuner], alpha = 0.7, linewidth = 3)

        ax[i].set(yscale = 'log', yticks = [0.2, 0.5, 1, 2, 5], ylim = [0.2, 5])
        ax[i].get_yaxis().set_major_formatter(ticker.ScalarFormatter())
        ax[i].get_xaxis().set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax[i].get_xaxis().set_minor_locator(ticker.AutoMinorLocator())

    # axis grid
    ax[i].grid()

    # title
    factor_as_string = 'thrice' if qd_term_factor == 3 else 'twice' if qd_term_factor == 2 else 'equal to' 
    title = 'Tuning convergence with a fixed-quality of {} times the average achieved quality deviation with default parameters'.format(factor_as_string)
    ax[i].set_title(title, fontsize = 12)
    
# x axis label and legend
ax[len(ax) - 1].set(xlabel = 'Experiments', ylabel = 'Tuning Quality')
fig.legend(handles = legend_handles)

fig.tight_layout()
fig.savefig('myproject/data/figures/tuning-convergence/together-fixed-qualdev-improv.png', bbox_inches='tight')