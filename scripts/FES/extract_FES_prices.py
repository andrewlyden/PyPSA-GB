"""
Extract fuel and carbon prices from FES Data Workbooks.

This script extracts commodity price assumptions from NESO FES workbooks:
- Fuel prices (gas, coal, oil, biomass) from CP1 sheet
- Carbon prices (EU ETS, UK carbon price) from CP2 sheet

Author: PyPSA-GB
License: MIT
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import re

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

try:
    from scripts.utilities.logging_config import setup_logging
except ImportError:
    logging.basicConfig(level=logging.INFO)
    def setup_logging(log_path):
        return logging.getLogger(__name__)

# Setup logging
logger = setup_logging(snakemake.log[0] if "snakemake" in dir() else "extract_FES_prices.log")


def extract_fuel_prices(workbook_path: str, fes_year: int) -> pd.DataFrame:
    """
    Extract fuel price assumptions from FES workbook.
    
    Robustly handles different FES workbook structures (2020-2025).
    
    Args:
        workbook_path: Path to FES workbook
        fes_year: FES year
        
    Returns:
        pd.DataFrame: Fuel prices with columns [year, fuel, price_gbp_per_mwh_thermal]
    """
    logger.info(f"Extracting fuel prices from {workbook_path}")
    
    # Try different sheet names
    sheet_names = ['CP1', 'Commodity Prices', 'Fuel Prices', 'Assumptions']
    
    for sheet_name in sheet_names:
        try:
            xl_file = pd.ExcelFile(workbook_path)
            if sheet_name not in xl_file.sheet_names:
                continue
                
            logger.info(f"  Trying sheet: {sheet_name}")
            
            # Read flexibly
            df_FES = pd.read_excel(workbook_path, sheet_name=sheet_name, dtype=str)
            
            # Find the data region
            header_row = None
            for idx, row in df_FES.iterrows():
                if any('year' in str(val).lower() for val in row if pd.notna(val)):
                    header_row = idx
                    break
                if any(re.match(r'20[2-5]\d', str(val)) for val in row if pd.notna(val)):
                    header_row = idx - 1 if idx > 0 else 0
                    break
            
            if header_row is None:
                continue
            
            # Re-read with header
            df_FES = pd.read_excel(workbook_path, sheet_name=sheet_name, header=header_row, dtype=str)
            df_FES = df_FES.dropna(how='all', axis=0).dropna(how='all', axis=1)
            
            # Find year column
            year_col = None
            for col in df_FES.columns:
                if 'year' in str(col).lower() or str(col).strip() == '':
                    year_col = col
                    break
            
            if year_col:
                df_FES = df_FES.set_index(year_col)
            
            # Transpose if years in columns
            if df_FES.index.dtype == 'object':
                for col in df_FES.columns:
                    if any(re.match(r'20[2-5]\d', str(val)) for val in df_FES[col] if pd.notna(val)):
                        df_FES = df_FES.T
                        break
            
            # Filter years
            df_FES.index = pd.to_numeric(df_FES.index, errors='coerce')
            df_FES = df_FES[df_FES.index.notna() & (df_FES.index >= 2020)]
            
            if len(df_FES) == 0:
                continue
            
            # Convert to numeric
            for col in df_FES.columns:
                df_FES[col] = pd.to_numeric(df_FES[col], errors='coerce')
            
            # Extract fuel prices
            fuel_prices = []
            for year in df_FES.index:
                for fuel_col in df_FES.columns:
                    if pd.isna(fuel_col) or str(fuel_col).strip() == '':
                        continue
                    
                    fuel_name = str(fuel_col).lower()
                    if 'gas' in fuel_name and 'biogas' not in fuel_name:
                        fuel, efficiency = 'gas', 0.50
                    elif 'coal' in fuel_name:
                        fuel, efficiency = 'coal', 0.35
                    elif 'oil' in fuel_name:
                        fuel, efficiency = 'oil', 0.35
                    elif 'biomass' in fuel_name or 'biogas' in fuel_name:
                        fuel, efficiency = 'biomass', 0.35
                    else:
                        continue
                    
                    price_pkwh = df_FES.loc[year, fuel_col]
                    if pd.isna(price_pkwh) or price_pkwh <= 0:
                        continue
                    
                    price_gbp_per_mwh_thermal = (price_pkwh * 10) / efficiency
                    fuel_prices.append({
                        'year': int(year),
                        'fuel': fuel,
                        'price_gbp_per_mwh_thermal': price_gbp_per_mwh_thermal
                    })
            
            if fuel_prices:
                result = pd.DataFrame(fuel_prices)
                logger.info(f"✓ Extracted {len(result)} fuel price records from {sheet_name}")
                return result
                
        except Exception as e:
            logger.debug(f"  Failed {sheet_name}: {e}")
            continue
    
    logger.warning("Using default fuel prices")
    return pd.DataFrame({
        "year": [2020, 2025, 2030, 2035, 2040, 2045, 2050] * 4,
        "fuel": ['gas'] * 7 + ['coal'] * 7 + ['oil'] * 7 + ['biomass'] * 7,
        "price_gbp_per_mwh_thermal": [20, 22, 24, 26, 28, 30, 32] * 4
    })


def extract_carbon_prices(workbook_path: str, fes_year: int) -> pd.DataFrame:
    """
    Extract carbon price assumptions from FES workbook.
    
    Robustly handles different FES workbook structures (2020-2025).
    
    Args:
        workbook_path: Path to FES workbook
        fes_year: FES year
        
    Returns:
        pd.DataFrame: Carbon prices with columns [year, carbon_price_gbp_per_tco2]
    """
    logger.info(f"Extracting carbon prices from {workbook_path}")
    
    # Try different sheet names
    sheet_names = ['CP2', 'Carbon Prices', 'Commodity Prices', 'Assumptions']
    
    for sheet_name in sheet_names:
        try:
            xl_file = pd.ExcelFile(workbook_path)
            if sheet_name not in xl_file.sheet_names:
                continue
            
            logger.info(f"  Trying sheet: {sheet_name}")
            
            # Read flexibly
            df = pd.read_excel(workbook_path, sheet_name=sheet_name, dtype=str)
            
            # Find header row with carbon keywords
            header_row = None
            for idx, row in df.iterrows():
                row_str = ' '.join(str(v).lower() for v in row if pd.notna(v))
                if 'carbon' in row_str or ('uk' in row_str and 'price' in row_str):
                    header_row = idx
                    break
            
            if header_row is None:
                continue
            
            # Re-read with header
            df = pd.read_excel(workbook_path, sheet_name=sheet_name, header=header_row, dtype=str)
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            
            # Find year and carbon columns
            year_col, carbon_col = None, None
            for col in df.columns:
                col_str = str(col).lower()
                if 'year' in col_str or str(col).strip() == '':
                    year_col = col
                if 'uk' in col_str and ('carbon' in col_str or 'price' in col_str):
                    carbon_col = col
                elif carbon_col is None and 'carbon' in col_str:
                    carbon_col = col
            
            if year_col:
                df = df.set_index(year_col)
            
            # Transpose if needed
            if df.index.dtype == 'object':
                for col in df.columns:
                    if any(re.match(r'20[2-5]\d', str(val)) for val in df[col] if pd.notna(val)):
                        df = df.T
                        break
            
            # Filter years
            df.index = pd.to_numeric(df.index, errors='coerce')
            df = df[df.index.notna() & (df.index >= 2020)]
            
            if len(df) == 0:
                continue
            
            # Find carbon column if still missing
            if carbon_col is None or carbon_col not in df.columns:
                for col in df.columns:
                    if 'carbon' in str(col).lower() or 'co2' in str(col).lower():
                        carbon_col = col
                        break
                if carbon_col is None and len(df.columns) > 0:
                    carbon_col = df.columns[0]
            
            # Extract prices
            df[carbon_col] = pd.to_numeric(df[carbon_col], errors='coerce')
            carbon_prices = pd.DataFrame({
                'year': df.index.astype(int),
                'carbon_price_gbp_per_tco2': df[carbon_col].values
            })
            carbon_prices = carbon_prices[carbon_prices['carbon_price_gbp_per_tco2'].notna()]
            
            if len(carbon_prices) > 0:
                logger.info(f"✓ Extracted {len(carbon_prices)} carbon price records from {sheet_name}")
                return carbon_prices
        
        except Exception as e:
            logger.debug(f"  Failed {sheet_name}: {e}")
            continue
    
    logger.warning("Using default carbon prices")
    return pd.DataFrame({
        "year": [2020, 2025, 2030, 2035, 2040, 2045, 2050],
        "carbon_price_gbp_per_tco2": [30, 40, 55, 70, 85, 95, 110]
    })


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("FES Fuel and Carbon Price Extraction")
    logger.info("=" * 80)
    
    # Get parameters from Snakemake
    workbook_path = snakemake.input.workbook
    fes_year = snakemake.params.fes_year
    
    output_fuel = snakemake.output.fuel_prices
    output_carbon = snakemake.output.carbon_prices
    
    logger.info(f"FES Year: {fes_year}")
    logger.info(f"Workbook: {workbook_path}")
    
    # Extract prices
    fuel_prices = extract_fuel_prices(workbook_path, fes_year)
    carbon_prices = extract_carbon_prices(workbook_path, fes_year)
    
    # Save outputs
    logger.info(f"Saving fuel prices to {output_fuel}")
    fuel_prices.to_csv(output_fuel, index=False)
    
    logger.info(f"Saving carbon prices to {output_carbon}")
    carbon_prices.to_csv(output_carbon, index=False)
    
    logger.info("=" * 80)
    logger.info("FES Price Extraction Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

