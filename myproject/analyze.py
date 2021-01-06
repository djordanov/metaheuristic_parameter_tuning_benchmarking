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

