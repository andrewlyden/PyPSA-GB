import pandas as pd


def write_snapshots(start, end, time_step):
    """writes the snapshots csv file

    Parameters
    ----------
    start : str
        start of simulation
    end : str
        end of simulation
    time_step : float
        defined as fraction of an hour, e.g., 0.5 is half hour
        currently set up as only hour or half hour
    Returns
    -------
    str
        returns the frequency
    """

    # SNAPSHOTS CSV FILE

    if time_step == 0.5:
        freq = '0.5H'
    elif time_step == 1.0:
        freq = 'H'
    else:
        raise Exception("Time step not recognised")

    dti = pd.date_range(
        start=start,
        end=end,
        freq=freq)
    df = pd.DataFrame(index=dti)
    df['weightings'] = time_step

    if time_step == 1.:
        appendix = df[-1:]
        new_index = df.index[-1] + pd.Timedelta(minutes=30)
        appendix.rename(index={appendix.index[0]: new_index}, inplace=True)
        df = df.append(appendix)

    df.index.name = 'name'
    df.to_csv('UC_data/snapshots.csv', header=True)
    df.to_csv('LOPF_data/snapshots.csv', header=True)

    return freq
