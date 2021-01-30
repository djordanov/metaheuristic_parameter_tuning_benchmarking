# irace preprocessing
# further preprocessing of irace runs

import pandas as pd
from pandas.core.frame import DataFrame

fnames = [
    "myproject/data/irace/test5000-SA-{'qualdev': 0, 'evals': 1000}-qualdev",
    "myproject/data/irace/test5000-SA-{'qualdev': 0, 'evals': 10000}-qualdev",
    "myproject/data/irace/test5000-GA-{'qualdev': 0, 'evals': 1000}-qualdev",
    "myproject/data/irace/test5000-GA-{'qualdev': 0, 'evals': 10000}-qualdev",
    "myproject/data/irace/test5000-ACO-{'qualdev': 0, 'evals': 1000}-qualdev",
    "myproject/data/irace/test5000-ACO-{'qualdev': 0, 'evals': 10000}-qualdev"
]

d = {
    'run': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'conf': [1, 1, 1, 2, 2, 3, 3, 3, 1, 1],
    'result': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
    'meanconf': [1, 1.5, 2, 4, 4.5, 6, 6.5, 7, 3.75, 3.2]
}

df = pd.DataFrame(d)
grouped = df.groupby(df['conf'])
confmean = grouped['result'].expanding().mean()
confmean = confmean.reset_index()
confmean = confmean.sort_values('level_1')
df['computed'] = confmean['result'].values

# compute mean result so far for every cfg 
# then compute incumbent (approximated) and join incumbent quality
def compute_run_trajectory(run_results: pd.DataFrame, runc_thresh1, runc_thresh2) -> pd.DataFrame:

    # compute mean result so far
    df = run_results.copy()
    grouped = df.groupby(df['configuration'])
    mean_so_far = grouped['result'].expanding().mean()
    df['cfg_mean_sofar'] = mean_so_far.reset_index().sort_values('level_1')['result'].values

    # add cfg run count so far
    df['cfg_runs_sofar'] = 1
    df['cfg_runs_sofar'] = grouped['cfg_runs_sofar'].cumsum()

    # compute incumbents
    incumbents = []
    incumbent = df.iloc[0,]
    runcount_threshold = runc_thresh1
    for run in df.index:

        # update threshold
        if runcount_threshold == runc_thresh1 and incumbent.cfg_runs_sofar > runc_thresh2: 
            runcount_threshold = runc_thresh2
            
        # slice all valid incumbent candidates
        effective_cfgruns_threshold = min(runcount_threshold, incumbent.cfg_runs_sofar)
        dfs = df.loc[(df.index <= run) & (df.cfg_runs_sofar >= effective_cfgruns_threshold)]
        incumbent_candidates = dfs.groupby(dfs.configuration).tail(1)

        if len(incumbent_candidates) == 0:
            incumbents.append(incumbent)
            continue

        # get the best incumbent candidate
        incumbent_candidates = incumbent_candidates.sort_values('cfg_mean_sofar')
        incumbent = incumbent_candidates.iloc[0,]
        incumbents.append(incumbent)

    df['incumbent_mean_sofar'] = [incumbent.cfg_mean_sofar for incumbent in incumbents]
    df['incumbent_id'] = [incumbent.configuration for incumbent in incumbents]
    return df

# run trajectory irace
runc_thresh1, runc_thresh2 = 5, 10
for fname in fnames:
    fname_run_results = fname + '-run-results.csv'
    run_results = pd.read_csv(fname_run_results, index_col = 0)

    run_trajectory = compute_run_trajectory(run_results, runc_thresh1, runc_thresh2)
    fname_run_trajectory = fname + '-run-trajectory.csv'
    run_trajectory.to_csv(fname_run_trajectory)