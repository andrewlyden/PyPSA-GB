Note on Regional Breakdown of FES data

The files of 'FES-2021--Leading_the_Way--dxcapacity-storage--gridsupplypoints.csv' are directly download from https://www.futureenergyscenarios.com/2022-FES/electricity-maps.html. Only available from FES2022.

There is another potential way to get this distribution from FES breakdown workbook (https://www.nationalgrideso.com/document/263896/download). However, we found the GSP items used in tabs of different categories (i.e. generation and demand) are over 500, but in the GSP info tab there are only about 300 GSP items with geographic info. We tried this approach but decided to drop it due to such inconsistency.


Note on GSP geographic information

the most complete geo info for GSP is https://data.nationalgrideso.com/system/gis-boundaries-for-gb-grid-supply-points

the gsp_gnode_directconnect_region_lookup.csv is download from the above page, dated at 2018, but enough for filling the missing GSP so far.