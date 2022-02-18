import pandas as pd


def write_lines():

    file = '../data/network/lines.csv'
    df = pd.read_csv(file)
    df.to_csv('LOPF_data/lines.csv', index=False, header=True)
