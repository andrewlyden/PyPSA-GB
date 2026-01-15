import requests

# Get the dataset package info from NESO API
url = 'https://api.neso.energy/api/3/action/package_show?id=regional-breakdown-of-fes-data-electricity'
r = requests.get(url, timeout=30)
data = r.json()

print('Available GSP Info Resources:\n')
for resource in data['result']['resources']:
    name = resource['name']
    if 'GSP' in name or 'Grid Supply Point' in name:
        print(f'Name: {name}')
        print(f'URL: {resource["url"]}')
        print(f'Format: {resource["format"]}')
        print(f'Created: {resource.get("created", "N/A")[:10]}')
        print()

