"""Frozen copy of the GSP clean-up in scripts/FES/FES_data.py before 26 Oct 2026 (test oracle only).

``data/FES/gsp_aliases.csv`` replaced these branches; ``test_scenario_sources.py`` checks that the
alias table gives exactly what this code gave, edition by edition. Do not edit or import elsewhere.
"""


def legacy_clean_gsp_names(FES_year, raw_FES_data):
    raw_FES_data = raw_FES_data.copy()
    if FES_year == 2021:
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar B', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar A', 'Dunbar')
        # remove whitespace at end of strings
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.strip()
        # remove row with Cain, which I cant find existence of
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Cain']
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Ferrybridge B (_F)', 'Ferrybridge B')
        # exact match for these
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Dumfries 11' if x == 'Dumfries' else x)
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Crookston A' if x == 'Crookston' else x)
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Grangemouth A' if x == 'Grangemouth' else x)
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Kilbowie 11' if x == 'Kilbowie' else x)
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Saltcoats A' if x == 'Saltcoats' else x)
        # remove Finstown
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Finstown']
        # remove shetland
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Shetland']

    elif FES_year == 2022:
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Ferrybridge B (_M)', 'Ferrybridge B')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar B', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar A', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('East Claydon (_B)', 'East Claydon')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Crookston A' if x == 'Crookston' else x)
        # remove Finstown
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Finstown']
        # remove shetland
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Shetland']

    elif FES_year == 2023:
        # replace the 951 row with CLYM
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Broadford-Skye', 'Broadford')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].replace({'Grange': 'Margam'})
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('East Claydon (_B)', 'East Claydon')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Ferrybridge B (_M)', 'Ferrybridge B')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar B', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar A', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Grangemouth A' if x == 'Grangemouth' else x)
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Kearsley 132kV' if x == 'Kearsley' else x)
        # delete row with Roosecote as can't see it has anything, bar some battery storage
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Roosecote']
        # delete row with Stornoway as only has little wind and no coordinates
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Stornoway']
        # remove Finstown
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Finstown']
        # remove shetland
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Shetland']

    elif FES_year == 2024:
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('East Claydon (_B)', 'East Claydon')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('East Claydon (_H)', 'East Claydon')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Ferrybridge B (_M)', 'Ferrybridge B')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Kirkby (_G)', 'Kirkby SPEN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Kirkby (_D)', 'Kirkby SPEN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Ormonde', 'Ormonde (Heysham)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Robin Rigg', 'Robin Rigg (Harker)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Walney 2', 'Walney 2 (Heysham)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Tynemouth 1', 'Tynemouth 132kV')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Tynemouth 2', 'Tynemouth 132kV')
        # raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Tynemouth 132kV 2', 'Tynemouth 132kV')
        # raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Tynemouth 132kV 1', 'Tynemouth 132kV')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Hams Hall', 'Hams Hall (Lea Marston 132kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Hart Moor', 'Hartmoor')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Camblesforth', 'Camblesforth 66kV (Drax)')
        # raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Skelton Margam (Grange 66kV)', 'Margam (Grange 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Fiddlers Ferry', 'Fiddlers Ferry (Cuerdley 132kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('St. Asaph', 'St Asaph (Bodelwyddan)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Broadford-Skye', 'Broadford')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Axminster_H', 'Axminster H')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Clacton', 'Clacton (Bramford)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Loudwater', 'Loudwater (Amersham)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Sheringham Shoal', 'Sheringham Shoal (Norwich)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Beddington (_C)', 'Beddington LPN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Canal Bank', 'Canal Bank (Willesden 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Greenford', 'Greenford (Willesden 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Littlebrook (_C)', 'Littlebrook LPN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Lodge Road', 'Lodge Road (St Johns Wood)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('New Cross', 'New Cross 66kV')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Perivale', 'Perivale (Willesden 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Wesley Avenue (Willesden)', 'Wesley Avenue (Willesden 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Beddington (_J)', 'Beddington SPN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Littlebrook (_J)', 'Littlebrook SPN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].replace({'Drax': 'Camblesforth 66kV (Drax)', 'Grange': 'Margam (Grange 66kV)'})
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar B', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar A', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Kearsley 132kV' if x == 'Kearsley' else x)
        # delete row with Port Ham as only small and no gsp data
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Port Ham']
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Roosecote']
        # remove these too Kellwood Road, Long Park, New Cumnock, Well Street Paisley, Stornoway
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('Kellwood Road')]
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('Long Park')]
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('New Cumnock')]
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('Well Street Paisley')]
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('Stornoway')]
        # remove Finstown
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Finstown']
        # remove shetland
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Shetland']

    elif FES_year >= 2025:
        # FES 2025+ - Start with 2024 logic and expand as needed
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('East Claydon (_B)', 'East Claydon')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('East Claydon (_H)', 'East Claydon')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Ferrybridge B (_M)', 'Ferrybridge B')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Kirkby (_G)', 'Kirkby SPEN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Kirkby (_D)', 'Kirkby SPEN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Ormonde', 'Ormonde (Heysham)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Robin Rigg', 'Robin Rigg (Harker)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Walney 2', 'Walney 2 (Heysham)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Tynemouth 1', 'Tynemouth 132kV')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Tynemouth 2', 'Tynemouth 132kV')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Hams Hall', 'Hams Hall (Lea Marston 132kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Hart Moor', 'Hartmoor')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Camblesforth', 'Camblesforth 66kV (Drax)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Fiddlers Ferry', 'Fiddlers Ferry (Cuerdley 132kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('St. Asaph', 'St Asaph (Bodelwyddan)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Broadford-Skye', 'Broadford')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Axminster_H', 'Axminster H')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Clacton', 'Clacton (Bramford)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Loudwater', 'Loudwater (Amersham)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Sheringham Shoal', 'Sheringham Shoal (Norwich)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Beddington (_C)', 'Beddington LPN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Canal Bank', 'Canal Bank (Willesden 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Greenford', 'Greenford (Willesden 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Littlebrook (_C)', 'Littlebrook LPN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Lodge Road', 'Lodge Road (St Johns Wood)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('New Cross', 'New Cross 66kV')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Perivale', 'Perivale (Willesden 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Wesley Avenue (Willesden)', 'Wesley Avenue (Willesden 66kV)')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Beddington (_J)', 'Beddington SPN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Littlebrook (_J)', 'Littlebrook SPN')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].replace({'Drax': 'Camblesforth 66kV (Drax)', 'Grange': 'Margam (Grange 66kV)'})
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar B', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].str.replace('Dunbar A', 'Dunbar')
        raw_FES_data['GSP'] = raw_FES_data['GSP'].apply(lambda x: 'Kearsley 132kV' if x == 'Kearsley' else x)
        # Remove unmapped GSPs
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Port Ham']
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Roosecote']
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('Kellwood Road', na=False)]
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('Long Park', na=False)]
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('New Cumnock', na=False)]
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('Well Street Paisley', na=False)]
        raw_FES_data = raw_FES_data[~raw_FES_data['GSP'].str.contains('Stornoway', na=False)]
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Finstown']
        raw_FES_data = raw_FES_data[raw_FES_data['GSP'] != 'Shetland']
    return raw_FES_data
