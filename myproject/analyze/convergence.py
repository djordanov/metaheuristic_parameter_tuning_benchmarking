import itertools
from myproject.run import mhruns

import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

from myproject.analyze.common_definitions import LINE_COLORS, LINE_STYLES, MARKERS

from myproject.helpers import cmhrun_fname, ctun_fname
from myproject.helpers import incumbents_smac, from_cand_params, config_to_cand_params_smac
from myproject.helpers import BASE_TERM, DEF_CFGS

tuners = ['irace', 'smac']
metaheuristics = ['SA', 'ACO', 'GA']
levals = [0, 1000, 10000, 100000]

felites_irace= {
    "5000-ACO-[('evals', 1000), ('qualdev', 0)]-qualdev.Rdata": 
        {'antcount': 439, 'alpha': 3.543, 'beta': 14.8282, 'pbest': 84.5142, 'evaporation': 0.4428},
    "5000-ACO-[('evals', 10000), ('qualdev', 0)]-qualdev.Rdata": 
        {'antcount': 132, 'alpha': 0.6219, 'beta': 5.4251, 'pbest': 62.2747, 'evaporation': 0.7806},
    "5000-ACO-[('evals', 100000), ('qualdev', 0)]-qualdev.Rdata": 
        {'antcount': 47, 'alpha': 0.8079, 'beta': 6.6904, 'pbest': 0.7138, 'evaporation': 0.9638},
    "5000-GA-[('evals', 1000), ('qualdev', 0)]-qualdev.Rdata": 
        {'popsize': 1, 'mut_rate': 0.8581, 'rank_weight': 0.2627},
    "5000-GA-[('evals', 10000), ('qualdev', 0)]-qualdev.Rdata": 
        {'popsize': 1, 'mut_rate': 0.6601, 'rank_weight': 16.519},
    "5000-GA-[('evals', 100000), ('qualdev', 0)]-qualdev.Rdata": 
        {'popsize': 281, 'mut_rate': 0.5634, 'rank_weight': 0.9341},
    "5000-SA-[('evals', 1000), ('qualdev', 0)]-qualdev.Rdata": 
        {'initial_temperature': 22.1271, 'repetitions': 7902, 'cooling_factor': 0.8808},
    "5000-SA-[('evals', 10000), ('qualdev', 0)]-qualdev.Rdata": 
        {'initial_temperature': 258.6476, 'repetitions': 857, 'cooling_factor': 0.6707},
    "5000-SA-[('evals', 100000), ('qualdev', 0)]-qualdev.Rdata": 
        {'initial_temperature': 235.7388, 'repetitions': 3831, 'cooling_factor': 0.8593}
}

def plot_data(axes, df: pd.DataFrame, tuner: str, algorithm: str, ):
    grouped_by_evals = df.groupby('evals')
    conv = grouped_by_evals.mean()
    label = '{}'.format(algorithm) if tuner == None else '{} + {}'.format(algorithm, tuner)
    if tuner != None:
        axes.plot(conv.index, conv.qualdev, label = label, color = LINE_COLORS[algorithm], linestyle = LINE_STYLES[tuner])
    else:
        axes.plot(conv.index, conv.qualdev, label = label, color = LINE_COLORS[algorithm])

def get_tuned_convergence_data(evals: int, tuner: str, metaheuristic: str, optimize: str) -> pd.DataFrame:
    terminate = {'qualdev': 0, 'evals': evals}
    tfname = ctun_fname(tuning_budget = 5000, algorithm = metaheuristic, terminate = terminate, optimize = optimize)

    if tuner == 'smac':
        incumbents = incumbents_smac(tfname)
        string_config_smac = incumbents['Configuration'][len(incumbents) - 1]
        cand_params = config_to_cand_params_smac(string_config_smac)
        alg, config, term, opt = from_cand_params(cand_params)
        mhrun_fname = cmhrun_fname(metaheuristic, config, BASE_TERM, optimize)
        return pd.read_csv('myproject/data/conv/' + mhrun_fname + '.csv')

    if tuner == 'irace': # this would require going into the R result file again, so the irace elites are saved in a dictionary
        mhrun_fname = cmhrun_fname(algorithm = metaheuristic, config = felites_irace[tfname + '.Rdata'], 
            terminate = BASE_TERM, optimize = optimize)
        return pd.read_csv('myproject/data/conv/' + mhrun_fname + '.csv')

def fill_subplot(evals, axes):
    # default convergence without tuning
    if evals == 0:
        for metaheuristic in metaheuristics:
            mhrun_fname = cmhrun_fname(algorithm = metaheuristic, config = DEF_CFGS[metaheuristic], terminate = BASE_TERM, optimize = 'qualdev')            
            df = pd.read_csv('myproject/data/conv/' + mhrun_fname + '.csv')
            plot_data(axes, df, None, metaheuristic)

    else:
        for tuner_mh in itertools.product(tuners, metaheuristics):
            df = get_tuned_convergence_data(evals, tuner_mh[0], tuner_mh[1], 'qualdev')
            plot_data(axes, df, tuner_mh[0], tuner_mh[1])
        
    # axis scales
    axes.set(xscale = 'log', yscale = 'log', yticks = [0.01, 0.1, 0.5, 1, 2])
    axes.tick_params(labelsize = 5)
    axes.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    axes.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    # axis grid and title
    axes.grid()
    axes.set_title('Metaheuristics tuned with termination after ' + str(evals) + ' evals', size = 5, pad = 3)

# convergence plot
fig, ax = plt.subplots(nrows = 4, ncols = 1, sharex = True, sharey = True)

for i, evals in enumerate(levals):
    fill_subplot(evals, ax[i])

# legend
handles, labels = ax[len(ax)-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='right')

# axes labels
fig.text(0.5, 0.05, "Evals", ha="center", va="center")
fig.text(0.05, 0.5, "Quality Deviation", ha="center", va="center", rotation=90)
fig.subplots_adjust(hspace=0.25)

fig.savefig('myproject/data/figures/convergencenow.png', dpi = 2000)