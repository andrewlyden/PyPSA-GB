"""Fetch NESO constraint data for validation comparison."""
import requests
import json
import pandas as pd
import io
import sys

def get_resource_urls(dataset_id):
    """Get download URLs from NESO API."""
    r = requests.get(f'https://api.neso.energy/api/3/action/datapackage_show?id={dataset_id}')
    data = r.json()
    resources = []
    for res in data['result']['resources']:
        resources.append({
            'name': res.get('name', ''),
            'path': res.get('path', ''),
            'format': res.get('format', ''),
        })
    return resources

def main():
    print("=" * 80)
    print("NESO CONSTRAINT DATA - RESOURCE URLS")
    print("=" * 80)

    for dataset_id, label in [
        ('constraint-breakdown', 'CONSTRAINT BREAKDOWN'),
        ('thermal-constraint-costs', 'THERMAL CONSTRAINT COSTS'),
        ('day-ahead-constraint-flows-and-limits', 'DAY-AHEAD CONSTRAINT FLOWS'),
    ]:
        print(f"\n--- {label} ---")
        resources = get_resource_urls(dataset_id)
        for r in resources:
            print(f"  {r['name']} [{r['format']}]: {r['path']}")

    # Try to download constraint breakdown 2019-2020 (covers our 2020 validation year)
    print("\n" + "=" * 80)
    print("DOWNLOADING CONSTRAINT BREAKDOWN 2019-2020")
    print("=" * 80)

    resources = get_resource_urls('constraint-breakdown')
    for r in resources:
        if '2019-2020' in r['name'] or '2019' in r['name']:
            print(f"\nFetching: {r['name']}")
            resp = requests.get(r['path'])
            df = pd.read_csv(io.StringIO(resp.text))
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst 5 rows:")
            print(df.head())
            print(f"\nUnique constraint types: {df.iloc[:, 0].nunique() if len(df.columns) > 0 else 'N/A'}")

    # Also get 2020-2021
    print("\n" + "=" * 80)
    print("DOWNLOADING CONSTRAINT BREAKDOWN 2020-2021")
    print("=" * 80)

    for r in resources:
        if '2020-2021' in r['name'] or '2020' in r['name']:
            print(f"\nFetching: {r['name']}")
            resp = requests.get(r['path'])
            df = pd.read_csv(io.StringIO(resp.text))
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst 5 rows:")
            print(df.head())

    # Download thermal constraint costs 2020-2021
    print("\n" + "=" * 80)
    print("DOWNLOADING THERMAL CONSTRAINT COSTS 20-21")
    print("=" * 80)

    resources = get_resource_urls('thermal-constraint-costs')
    for r in resources:
        if '20-21' in r['name'] or '2020' in r['name']:
            print(f"\nFetching: {r['name']}")
            resp = requests.get(r['path'])
            # Might be xlsx
            if r['format'].lower() in ('xlsx', 'xls'):
                df = pd.read_excel(io.BytesIO(resp.content))
            else:
                df = pd.read_csv(io.StringIO(resp.text))
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst 10 rows:")
            print(df.head(10))

    # Download day-ahead constraint flows (current file only, check if it has historical)
    print("\n" + "=" * 80)
    print("DOWNLOADING DAY-AHEAD CONSTRAINT FLOWS")
    print("=" * 80)

    resources = get_resource_urls('day-ahead-constraint-flows-and-limits')
    for r in resources:
        if r['format'].lower() == 'csv':
            print(f"\nFetching: {r['name']}")
            resp = requests.get(r['path'])
            df = pd.read_csv(io.StringIO(resp.text))
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst 10 rows:")
            print(df.head(10))
            if 'Date' in df.columns:
                print(f"\nDate range: {df['Date'].min()} to {df['Date'].max()}")
            # Show boundary names
            for col in df.columns:
                if 'boundary' in col.lower() or 'constraint' in col.lower() or 'group' in col.lower():
                    print(f"\nUnique {col}: {df[col].unique()[:30]}")

if __name__ == '__main__':
    main()
