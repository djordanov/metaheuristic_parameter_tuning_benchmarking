import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# plot convergence
def convergence(df: pd.DataFrame):
    # clean dataframe
    df = df.drop('time', axis = 'columns')

    # compute mean
    dfg = df.groupby('evals')
    dfgm = dfg.mean()

    # create plot
    plt.plot(dfgm.index, dfgm.qualdev)
    plt.xlabel('Number of Quality Evaluations')
    plt.ylabel('Quality Deviation from Optimum')
    plt.semilogx()
    plt.show()

def evals2time(df: pd.DataFrame):
    # decimate the data
    dfs = df.sample(100000) if len(df) > 100000 else df

    # linear regression ...
    y = dfs.time
    x = dfs[['evals']]
    x = sm.add_constant(x)

    model = sm.OLS(y, x).fit()
    intercept = model.params.const
    evals = model.params.evals

    plt.scatter(dfs.evals, dfs.time, marker = 'o', color = 'cornflowerblue', alpha = 0.5, label = 'empirical result sa')
    plt.plot(dfs.evals, intercept + evals * dfs.evals, color = 'black', linewidth = '2', alpha = 1, label = 'linear regression sa')
    plt.ylabel('Wall Clock Time in Seconds')
    plt.xlabel('Number of Quality Evaluations')
    plt.legend()
    plt.show()

fnames = {
    1000: {
        'smac': {
            'SA': 'myproject/data/smac/5000-SA-{qualdev 0, evals 1000}-qualdev/NoScenarioFile/detailed-traj-run-483634792.csv',
            'GA': 'myproject/data/smac/5000-GA-{qualdev 0, evals 1000}-qualdev/NoScenarioFile/detailed-traj-run-115943854.csv',
            'ACO': 'myproject/data/smac/5000-ACO-{qualdev 0, evals 1000}-qualdev/NoScenarioFile/detailed-traj-run-367701444.csv'
        },
        'irace': {
            'SA': "myproject/data/irace/test5000-SA-{'qualdev': 0, 'evals': 1000}-qualdev",
            'GA': "myproject/data/irace/test5000-GA-{'qualdev': 0, 'evals': 1000}-qualdev",
            'ACO': "myproject/data/irace/test5000-ACO-{'qualdev': 0, 'evals': 1000}-qualdev"
        }
    },
    10000: {
        'smac': {
            'SA': 'myproject/data/smac/5000-SA-{qualdev 0, evals 10000}-qualdev/NoScenarioFile/detailed-traj-run-810234963.csv',
            'GA': 'myproject/data/smac/5000-GA-{qualdev 0, evals 10000}-qualdev/NoScenarioFile/detailed-traj-run-929863747.csv',
            'ACO': 'myproject/data/smac/5000-ACO-{qualdev 0, evals 10000}-qualdev/NoScenarioFile/detailed-traj-run-421595855.csv'
        },
        'irace': {
            # 'SA': "myproject/data/irace/test5000-SA-{'qualdev': 0, 'evals': 10000}-qualdev",
            'GA': "myproject/data/irace/test5000-GA-{'qualdev': 0, 'evals': 10000}-qualdev"
            # 'ACO': "myproject/data/irace/test5000-ACO-{'qualdev': 0, 'evals': 10000}-qualdev"
        }
    },
    100000: {}
}

colors = {
    'SA': 'blue',
    'GA': 'green',
    'ACO': 'red'
}

def tuning_trajectory_smac(fname, algorithm, ax):
    df = pd.read_csv(fname, header = None, skiprows = 1)
    estimated_training_performance_incumbent = pd.to_numeric(df[1], errors = 'coerce')
    smac_cpu_time = df[4]
    smac_cpu_time = smac_cpu_time * 5000 / smac_cpu_time[len(smac_cpu_time)-1] # normalize time
    label = '{} tuned by smac'.format(algorithm)
    ax.plot(smac_cpu_time, estimated_training_performance_incumbent, 
             color = colors[algorithm], linestyle = '--', label = label)

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
             color = colors[algorithm], marker = 'o', label = label)
    
fig, axs = plt.subplots(nrows = 3, ncols = 1, sharex = 'col', sharey = 'col')
fig.tight_layout()

for i, evals in enumerate(fnames):
    ax = axs[i]
    for tuner in fnames[evals].keys():
        for metaheuristic in fnames[evals][tuner].keys():
            fname = fnames[evals][tuner][metaheuristic]
            if tuner == 'smac':
                tuning_trajectory_smac(fname, metaheuristic, ax)
            elif tuner == 'irace':
                tuning_trajectory_irace(fname, metaheuristic, ax)

    ax.set_title('Evals = ' + str(evals))
    ax.legend()
    ax.set_xlabel('Experiments')
    ax.set_ylabel('Qualtiy Deviation to Optimum')
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 1) # this needs to be adjusted depending on evals/ depending on the data

plt.show()