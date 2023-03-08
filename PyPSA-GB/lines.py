import pandas as pd


def write_lines(networkmodel=True):

    if networkmodel:
        file = '../data/network/lines.csv'
        df = pd.read_csv(file)
        df.to_csv('LOPF_data/lines.csv', index=False, header=True)
    else:
        file = '../data/network/links.csv'
        df = pd.read_csv(file)
        df.to_csv('LOPF_data/lines.csv', index=False, header=True)
