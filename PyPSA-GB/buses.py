import pandas as pd


def write_buses(year):
    """writes the buses csv file

    Parameters
    ----------
    Returns
    -------
    """

    # BUSES CSV FILE
    # first off the bus is a just a single bus
    data = {'name': 'bus'}
    df = pd.DataFrame(data=data, index=[0])
    df.to_csv('UC_data/buses.csv', index=False, header=True)

    file = '../data/network/buses.csv'
    df = pd.read_csv(file)

    if year <= 2020:
        df['carrier'] = 'AC'
    df.to_csv('LOPF_data/buses.csv', index=False, header=True)
