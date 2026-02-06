import pypsa
import os

# Check if DSR exists in different network stages
stages = [
    ('Base', 'resources/network/HT35_flex.nc'),
    ('With flexibility', 'resources/network/HT35_flex_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors_flexibility.nc'),
    ('Clustered', 'resources/network/HT35_flex_clustered.nc'),
    ('Solved', 'resources/network/HT35_flex_solved.nc'),
]

print('DSR PRESENCE CHECK ACROSS NETWORK STAGES')
print('=' * 80)

for stage_name, path in stages:
    try:
        n = pypsa.Network(path)
        dsr = n.generators[n.generators.carrier == 'demand response']
        print(f'\n{stage_name}:')
        print(f'  File: {os.path.basename(path)}')
        print(f'  DSR generators: {len(dsr)}')
        if len(dsr) > 0:
            print(f'  DSR capacity: {dsr.p_nom.sum():,.0f} MW')
            print(f'  DSR marginal cost: £{dsr.marginal_cost.iloc[0]:.0f}/MWh')
        else:
            print('  ❌ No DSR generators found')
    except FileNotFoundError:
        print(f'\n{stage_name}: ❌ File not found')
    except Exception as e:
        print(f'\n{stage_name}: ❌ Error - {e}')

print('\n' + '=' * 80)
