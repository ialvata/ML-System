import pandas as pd
from collections.abc import Generator
from sklearn.model_selection import TimeSeriesSplit

def add_lags(df:pd.DataFrame)-> pd.DataFrame:
    """
    Each row of df must be a separate time point, which will be transformed
    into a lag. This function will transform a matrix of dim -> n_samples x n_columns
    into a matrix of dim -> 1 x (n_columns*n_lags)
    """
    n_lags = df.shape[0]
    lags=range(0,n_lags)
    appended_lags = []
    for lag in lags: 
        lag_df= df.iloc[[lag]].drop(columns=["Date Time"]).reset_index(drop=True)
        lag_df.columns=[x+"_lag_"+str(n_lags-lag) for x in lag_df.columns]
        appended_lags.append(lag_df)
    return pd.concat(appended_lags, axis=1) # by columns

def fetch_data(source:str,n_observations, num_lags)-> Generator[tuple[pd.DataFrame,pd.DataFrame], None, None]:
    df = pd.read_csv(source)
    tscv = TimeSeriesSplit(n_splits=n_observations,
                           max_train_size=num_lags, test_size=1)
    train_rows = []
    label_rows = []
    for train, test in tscv.split(df):
        # print(f"len(train) = {len(train)}, last of train = {train[-1]}, test = {test}")
        train_rows.append(add_lags(df.iloc[train]))
        label_rows.append(df.iloc[test].drop(columns=["Date Time"]))
    yield (
        pd.concat(train_rows, axis=0).reset_index(drop=True),
        pd.concat(label_rows, axis=0).reset_index(drop=True)
    )