import pandas as pd

xlsx_path = r"c:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1\data\network\ETYS\GB_network.xlsx"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 40)

sheets = ["Dem_per_node", "Emb_BMUs_to_GSP", "Dir_con_BMUs_to_node"]

for sheet in sheets:
    print("=" * 80)
    print(f"SHEET: {sheet}")
    print("=" * 80)
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Dtypes:\n{df.dtypes}")
        print(f"\nFirst 20 rows:")
        print(df.head(20).to_string())
    except Exception as e:
        print(f"Error reading sheet: {e}")
    print("\n")
