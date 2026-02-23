"""Check which bus prefixes exist for new offshore wind farm connections."""
import pypsa

n = pypsa.Network('resources/network/Historical_2023_etys_solved.nc')

# Check prefixes for potential connection points
for prefix in ['SGRW', 'CRSS', 'CRIR', 'CREB', 'KEAD', 'LACA', 'TEAL',
               'NEART', 'DGBK', 'INCH', 'COCK', 'TORN', 'TERA', 'BFOT']:
    buses = [b for b in n.buses.index if b.startswith(prefix)]
    if buses:
        v_noms = sorted(set(n.buses.loc[b, 'v_nom'] for b in buses))
        print(f"  {prefix}: {len(buses)} buses, voltages: {v_noms}")
    else:
        print(f"  {prefix}: NOT FOUND")
