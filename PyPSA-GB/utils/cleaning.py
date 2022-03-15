import pandas as pd
import os


def remove_double(df):
    '''
    If the last two rows have the same value, the last row will be removed
    The fixed dataframe will be returned

    Parameters
    ----------
    df : pd.DataFrame
        df to fix

    Returns
    ----------
    df : pd.DataFrame
        fixed df

    '''

    if (df.iloc[-1] == df.iloc[-2]).all():
        return df.iloc[:-1]

    else:
        return df


def unify_index(dfs, freq):
    '''
    Method to unify the indices of all dataframes in dfs

    Parameters
    ----------
    dfs : list of pd.DataFrame
        list of the dataframes to be made uniform in terms of their index
    freq : str
        either 'H' for full hour frequency or '0.5H' for half hour

    Returns
    ----------
    list of dataframes

    '''

    dfs = [remove_double(df) for df in dfs]
    # for df in dfs:
    #     df.index.tz_localize(None)

    # unify start and end of indexes
    starts = [df.index[0] for df in dfs]
    starts = [pd.Timestamp(start) if isinstance(start, str) else start for start in starts] 
    starts = [start.tz_localize(None) for start in starts]
    start = max(starts)

    ends = [df.index[-1] for df in dfs]
    ends = [pd.Timestamp(end) if isinstance(end, str) else end for end in ends] 
    ends = [end.tz_localize(None) for end in ends]
    end = min(ends)

    dfs = [df.loc[start:end] for df in dfs]
    goal_index = pd.date_range(start, end, freq=freq)

    for i, df in enumerate(dfs):

        for col in df.columns:
            if isinstance(df[col].dtypes, object):
                df[col] = pd.to_numeric(df[col], downcast='float')

        # case of the new dataframe being more fine-grained
        if len(goal_index) > len(df):
            dfs[i] = df.resample(freq).interpolate()

        # in case the old dataframe needs to be coarse grained
        elif len(goal_index) < len(df):
            dfs[i] = df.resample(freq).mean()

    # adjust global snaptshots
    fix_snapshots(dfs[0].index)

    return dfs


def fix_snapshots(data_snapshots, snapshots_path='LOPF_data/snapshots.csv'):
    '''
    Some data might not be available for the full range of snapshots, so
    snapshots has the largest set of common timestamps
    This method takes the currently snapshots availabe for the currently
    investigates data (such as generators or marginal cost) and adjusts the
    snapshots stored in LOPF/snapshots.py if necessary

    Parameters
    ----------
    data_snapshots : pd.TimeSeries
        snapshots for which data is available
    snapshots_path : str
        path to where snapshots are stored

    Returns
    ----------
    -

    '''
    # use the snapshots index
    snapshots = pd.read_csv(snapshots_path, index_col=0, parse_dates=True)
    snapshots.loc[data_snapshots[0]:data_snapshots[-1]].to_csv(snapshots_path)


def unify_snapshots(target, filenames, dir):
    '''
    goes over all dataframe stored in filenames and
    adjusts their length to match the one in target
    Method should be called once before converting data to
    a network

    Parameters
    ----------
    target : str
        file to dataframe to which all files will be adjusted
    filenames : list of str
        files for which index should be checked
    dir : str
        directory where all files are

    Returns
    ----------
    -

    '''

    target = pd.read_csv(os.path.join(dir, target), index_col=0, parse_dates=True).index
    target_length = len(target)

    for file in filenames:
        file = os.path.join(dir, file)
        df = pd.read_csv(file, index_col=0, parse_dates=True)

        if len(df) > target_length:
            df.loc[target[0]:target[-1]].to_csv(file)


if __name__ == '__main__':

    dummy_sn = pd.date_range('2020-01-01 00:00:00', '2020-01-01 03:00:00', freq='0.5H')
    path = 'utils/dump/snapshots.csv'
    pd.DataFrame(index=dummy_sn).to_csv(path)

    print('at start')
    print(pd.read_csv(path, index_col=0, parse_dates=True).index)

    sn = pd.date_range('2020-01-01 00:00:00', '2020-01-01 02:30:00', freq='0.5H')

    fix_snapshots(sn, snapshots_path=path)

    print('\n at end')
    print(pd.read_csv(path, index_col=0, parse_dates=True).index)
