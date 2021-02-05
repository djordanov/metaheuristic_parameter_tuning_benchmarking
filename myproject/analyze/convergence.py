import pandas as pd
import matplotlib.pyplot as plt

from myproject.analyze.common_definitions import LINE_COLORS

fnames_notuning = {
    'SA': 'conv-SA-0-None-None-4836.csv',
    'ACO': 'conv-ACO-0-None-None-620.csv',
    'GA': 'conv-GA-0-None-None-4987.csv'
}

fnames_tuned = {
    1000: {
        'irace': {
            'SA': "SA-5000-{'initial_temperature': 27.2484, 'repetitions': 8026, 'cooling_factor': 0.1716}-None-1412.csv",
            'ACO': "ACO-5000-{'antcount': 439, 'alpha': 3.543, 'beta': 14.8282, 'evaporation': 0.4428, 'pbest': 84.5142}-None-920.csv",
            'GA': "GA-5000-{'popsize': 1, 'mut_rate': 0.851, 'rank_weight': 0.2627}-None-830.csv"
        },
        'smac': {
            'SA': "",
            'ACO': "ACO-5000-{'antcount': 67, 'alpha': 0.9553886797155056, 'beta': 13.368063653676876, 'evaporation': 0.058241608029438105, 'pbest': 0.781074288363884}-None-3710.csv",
            'GA': "GA-5000-{'popsize': 1, 'mut_rate': 0.9354300849532875, 'rank_weight': 4.2800906310470275}-None-1314.csv"
        }
    },
    10000: {
        'irace': {
            'SA': "",
            'ACO': "ACO-5000-{'antcount': 132, 'alpha': 0.6219, 'beta': 5.4251, 'evaporation': 0.7806, 'pbest': 62.2747}-None-7564.csv",
            'GA': ""
        },
        'smac': {
            'SA': "",
            'ACO': "ACO-5000-{'antcount': 345, 'alpha': 3.2050014761743935, 'beta': 17.686316543093692, 'evaporation': 0.9263612615269491, 'pbest': 0.11156702049905944}-None-8356.csv",
            'GA': ""
        }
    }
}

fig, axs = plt.subplots(nrows = 3, ncols = 1, sharex = 'col')
fig.tight_layout()

ax = axs[0]
for algorithm, fname in fnames_notuning.items():
    df = pd.read_csv('myproject/data/conv/' + fname)
    grouped_by_evals = df.groupby('evals')
    conv = grouped_by_evals.mean()
    ax.plot(conv.index, conv.qualdev, label = algorithm + ' with default parameters')
ax.set_title('Convergence Plot Metaheuristics with default parameters')
ax.set_ylabel('Qualtiy Deviation')
ax.legend()

ax = axs[1]
for tuner, fnames_algorithms in fnames_tuned[1000].items():
    for algorithm, fname in fnames_algorithms.items():
        if fname == "":
            continue
        df = pd.read_csv('myproject/data/conv/' + fname)
        grouped_by_evals = df.groupby('evals')
        conv = grouped_by_evals.mean()
        ax.plot(conv.index, conv.qualdev, label = algorithm + ' tuned to 1000 evals with ' + tuner)
ax.set_title('Convergence Plot Metaheuristics tuned to 1,000 evals')
ax.set_ylabel('Qualtiy Deviation')
ax.legend()

ax = axs[2]
for tuner, fnames_algorithms in fnames_tuned[10000].items():
    for algorithm, fname in fnames_algorithms.items():
        if fname == "":
            continue
        df = pd.read_csv('myproject/data/conv/' + fname)
        grouped_by_evals = df.groupby('evals')
        conv = grouped_by_evals.mean()
        ax.plot(conv.index, conv.qualdev, label = algorithm + ' tuned to 10000 evals with ' + tuner)
ax.set_title('Convergence Plot Metaheuristics tuned to 10,000 evals')
ax.set_ylabel('Qualtiy Deviation')
ax.legend()

plt.xlabel('Evals')
plt.savefig('myproject/data/figures/convergence.png', dpi = 1000)