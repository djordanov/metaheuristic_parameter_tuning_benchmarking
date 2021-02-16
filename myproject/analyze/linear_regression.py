import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

fnames = {
    'SA': 'myproject/data/conv/SA-0-None-None.csv',
    'ACO': 'myproject/data/conv/ACO-0-None-None.csv',
    'GA': 'myproject/data/conv/GA-0-None-None.csv'
}

time2evals = {
    'algorithm': [],
    'rsquared': [],
    'intercept': [],
    'coefficient': []
}

for algorithm, file in fnames.items():
    
    # linear regression time -> evals
    df = pd.read_csv(file)
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
    plt.scatter(evals_mean_time.time, evals_mean_time.index, label = algorithm)
    
# save model values
pd.DataFrame(time2evals).to_csv('myproject/data/time2evals.csv', index = None)
    
plt.xlabel('Time in Seconds')
plt.ylabel('Number of Tour Quality Evaluations')
plt.legend()
plt.savefig('myproject/data/figures/time2evals.png', dpi = 1000)

