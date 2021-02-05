import pandas as pd
import matplotlib.pyplot as plt

from myproject.analyze.common_definitions import LINE_COLORS

fnames_tuning = {
    1000: {
        'smac': {
            'SA': 'myproject/data/smac/5000-SA-{qualdev 0, evals 1000}-qualdev/NoScenarioFile/detailed-traj-run-114979431.csv',
            'GA': 'myproject/data/smac/5000-GA-{qualdev 0, evals 1000}-qualdev/NoScenarioFile/detailed-traj-run-482289583.csv',
            'ACO': 'myproject/data/smac/5000-ACO-{qualdev 0, evals 1000}-qualdev/NoScenarioFile/detailed-traj-run-103952363.csv'
        },
        'irace': {
            'SA': "myproject/data/irace/test5000-SA-{'qualdev': 0, 'evals': 1000}-qualdev",
            'GA': "myproject/data/irace/test5000-GA-{'qualdev': 0, 'evals': 1000}-qualdev",
            'ACO': "myproject/data/irace/test5000-ACO-{'qualdev': 0, 'evals': 1000}-qualdev"
        }
    },
    10000: {
        'smac': {
            'SA': 'myproject/data/smac/5000-SA-{qualdev 0, evals 10000}-qualdev/NoScenarioFile/detailed-traj-run-372562634.csv',
            'GA': 'myproject/data/smac/5000-GA-{qualdev 0, evals 10000}-qualdev/NoScenarioFile/detailed-traj-run-929863747.csv',
            'ACO': 'myproject/data/smac/5000-ACO-{qualdev 0, evals 10000}-qualdev/NoScenarioFile/detailed-traj-run-734026793.csv'
        },
        'irace': {
            'SA': "myproject/data/irace/test5000-SA-{'qualdev': 0, 'evals': 10000}-qualdev",
            'GA': "myproject/data/irace/test5000-GA-{'qualdev': 0, 'evals': 10000}-qualdev",
            'ACO': "myproject/data/irace/test5000-ACO-{'qualdev': 0, 'evals': 10000}-qualdev"
        }
    },
    100000: {}
}

def tuning_trajectory_smac(fname, algorithm, ax):
    df = pd.read_csv(fname, header = None, skiprows = 1)
    estimated_training_performance_incumbent = pd.to_numeric(df[1], errors = 'coerce')
    smac_cpu_time = df[4]
    smac_cpu_time = smac_cpu_time * 5000 / smac_cpu_time[len(smac_cpu_time)-1] # normalize time
    label = '{} tuned by smac'.format(algorithm)
    ax.plot(smac_cpu_time, estimated_training_performance_incumbent, 
             color = LINE_COLORS[algorithm], linestyle = '--', label = label)

def tuning_trajectory_irace(fname, algorithm, ax):
    fiterations = fname + '-iterations.csv'
    ftest_experiments = fname + '-test-experiments.csv'

    test_experiments = pd.read_csv(ftest_experiments)
    test_experiments_mean = test_experiments.mean()
    iterations = pd.read_csv(fiterations)

    tuning_trajectory = [( row.experiments, test_experiments_mean[str(row.elite)] ) \
                             for index, row in iterations.iterrows()]
    df = pd.DataFrame(tuning_trajectory, columns = ['experiments', 'performance'])
    df.experiments = df.experiments.cumsum()
    
    label = '{} tuned by irace'.format(algorithm)
    ax.plot('experiments', 'performance', data = df, \
             color = LINE_COLORS[algorithm], marker = 'o', label = label)
    
fig, axs = plt.subplots(nrows = 3, ncols = 1, sharex = 'col', sharey = 'col')
fig.tight_layout()

for i, evals in enumerate(fnames_tuning):
    ax = axs[i]
    for tuner in fnames_tuning[evals].keys():
        for metaheuristic in fnames_tuning[evals][tuner].keys():
            fname = fnames_tuning[evals][tuner][metaheuristic]
            if tuner == 'smac':
                tuning_trajectory_smac(fname, metaheuristic, ax)
            elif tuner == 'irace':
                tuning_trajectory_irace(fname, metaheuristic, ax)

    ax.set_title('Evals = ' + str(evals))
    ax.legend()
    ax.set_ylabel('Quality Deviation to Optimum')
    ax.set_xlim(0, 5000)
#    ax.set_ylim(0, 0.5) # this needs to be adjusted depending on evals/ depending on the data

plt.xlabel('Experiments')
plt.show()
# plt.savefig('myproject/data/figures/tuning_trajectories.png', dpi = 1000)