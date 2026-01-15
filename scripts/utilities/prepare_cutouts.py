import cartopy.io.shapereader as shpreader
import geopandas as gpd
import os
import shutil

import atlite


def prepare_cutouts_years():

    years = list(range(2010, 2022 + 1))
    for y in years:
        shpfilename = shpreader.natural_earth(resolution='10m',
                                              category='cultural',
                                              name='admin_0_countries')
        reader = shpreader.Reader(shpfilename)
        UK = gpd.GeoSeries({r.attributes['NAME_EN']: r.geometry
                            for r in reader.records()},
                           crs={'init': 'epsg:4326'}).reindex(['United Kingdom'])

        # Define the cutout; this will not yet trigger any major operations
        path = 'UK-{}.nc'.format(y)
        time = str(y)
        cutout = atlite.Cutout(path=path,
                               module="era5",
                               bounds=UK.unary_union.bounds,
                               time=time)
        
        # print(cutout.available_features)

        # This is where all the work happens
        # (this can take some time, for 2018 it took circa 3 hours).
        # features = ['height', 'wind', 'temperature', 'runoff']
        cutout.prepare()

def prepare_cutout_year(y, output_path):

    file_path = f'data/atlite/cutouts/uk-{str(y)}.nc'
    
    # Check if the file exists in data dir
    if os.path.exists(file_path):
        print(f"Cutout for year {str(y)} exists in data dir, copying.")
        shutil.copy(file_path, output_path)
        return

    print(f"Cutout for year {str(y)} does not exist in data dir, downloading...")

    shpfilename = shpreader.natural_earth(resolution='10m',
                                            category='cultural',
                                            name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    UK = gpd.GeoSeries({r.attributes['NAME_EN']: r.geometry
                        for r in reader.records()},
                        crs={'init': 'epsg:4326'}).reindex(['United Kingdom'])

    # Define the cutout; this will not yet trigger any major operations
    path = output_path
    time = str(y)
    print(f"Preparing cutout: {path} for year {time}")
    cutout = atlite.Cutout(path=path,
                            module="era5",
                            bounds=UK.unary_union.bounds,
                            time=time)

    # This is where all the work happens
    # (this can take some time, for 2018 it took circa 3 hours).
    # features = ['height', 'wind', 'temperature', 'runoff']
    print(f"Downloading and processing ERA5 data for {y}... (may take 2-4 hours)")
    cutout.prepare()
    print(f"Cutout for year {y} completed successfully!")

if __name__ == "__main__":
    # Get years from snakemake params
    years = snakemake.params.years
    outputs = snakemake.output
    
    print(f"Preparing cutouts for years: {years}")
    
    for i, year in enumerate(years):
        output_path = outputs[i]
        print(f"\n[{i+1}/{len(years)}] Generating cutout for {year}...")
        prepare_cutout_year(year, output_path)
    
    print("\nAll cutouts prepared successfully!")

