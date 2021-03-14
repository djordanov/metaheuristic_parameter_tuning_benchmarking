from myproject.helpers import BASE_TERM, DEF_CFGS, cmhrun_fname
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker

from sklearn.linear_model import LinearRegression

time2evals = {
    'algorithm': [],
    'rsquared': [],
    'intercept': [],
    'coefficient': []
}

fig, ax = plt.subplots()
for algorithm, config in DEF_CFGS.items():
    
    # linear regression time -> evals
    fname = cmhrun_fname(algorithm, config, BASE_TERM)
    df = pd.read_csv('myproject/data/conv/' + fname + '.csv')
    x = df.time.values.reshape(-1, 1)
    y = df.evals.values.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    rsquared = model.score(x, y)
    intercept = model.intercept_[0]
    coefficient = model.coef_[0][0]
    
    # add to dictionary
    time2evals['algorithm'].append(algorithm)
    time2evals['rsquared'].append(rsquared)
    time2evals['intercept'].append(intercept)
    time2evals['coefficient'].append(coefficient)
    
    # scatter plot
    grouped_by_evals = df.groupby('evals')
    evals_mean_time = grouped_by_evals.mean()
    ax.scatter(evals_mean_time.time, evals_mean_time.index, label = algorithm)
    
# save model values
pd.DataFrame(time2evals).to_csv('myproject/data/time2evals.csv', index = None)
    
ax.set_xlabel('Elapsed Wall Clock Time in Seconds')
ax.set_ylabel('Number of Tour Quality Evaluations')
ax.get_yaxis().set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax.legend()

fig.tight_layout()
fig.savefig('myproject/data/figures/time2evals.png')