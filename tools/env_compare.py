"""Compare envs/pypsa-gb.yaml (conda/pip specs) to installed packages in conda env 'pypsa-gb'.

Usage: python tools/env_compare.py
"""
import json
import re
import subprocess
from pathlib import Path

yaml_path = Path('envs/pypsa-gb.yaml')
if not yaml_path.exists():
    print('ERROR: envs/pypsa-gb.yaml not found')
    raise SystemExit(1)

text = yaml_path.read_text()
lines = [l.rstrip() for l in text.splitlines()]
pkgs_yaml = []
pip_pkgs = []
in_pip = False
for l in lines:
    if l.strip().startswith('- pip:'):
        in_pip = True
        continue
    if in_pip:
        m = re.match(r"\s*-\s*(.+)", l)
        if m:
            pip_pkgs.append(m.group(1))
        continue
    m = re.match(r"\s*-\s*([^#\n]+)", l)
    if m:
        pkg = m.group(1).strip()
        if pkg:
            pkgs_yaml.append(pkg)

# helper
def simplify(s):
    return re.split(r'[\s,>=<\[]', s)[0].lower()

yaml_names = {simplify(p):p for p in pkgs_yaml}
pip_names = {simplify(p):p for p in pip_pkgs}

# get conda list JSON
proc = subprocess.run(['conda','list','-n','pypsa-gb','--json'], capture_output=True, text=True)
if proc.returncode!=0:
    print('Failed to get conda list:', proc.stderr)
    raise SystemExit(1)
conda_list = json.loads(proc.stdout)
installed = {pkg['name'].lower():pkg['version'] for pkg in conda_list}

# compare
only_in_yaml = sorted(k for k in yaml_names if k not in installed and k not in pip_names)
only_in_installed = sorted(k for k in installed if k not in yaml_names and k not in pip_names)
version_mismatches = []
for name, raw in yaml_names.items():
    if name in installed:
        m = re.search(r'([>=<~!^]+\s*[0-9\.]+)', raw)
        if m:
            want = m.group(1).replace(' ','')
            have = installed[name]
            if not re.match(r"^"+re.escape(want.lstrip('>=<~!^')) , have):
                version_mismatches.append((name, raw, have))

print('=== ENV YAML SUMMARY ===')
print('Conda packages specified:', len(yaml_names))
print('Pip packages specified:', len(pip_names))
print('Installed packages in env pypsa-gb:', len(installed))
print()

print('Packages referenced in YAML but NOT installed (conda scope):')
if only_in_yaml:
    for k in only_in_yaml[:200]:
        print(' -', yaml_names[k])
else:
    print(' - None')

print('\nPackages installed but not referenced in YAML (showing first 200):')
if only_in_installed:
    for k in only_in_installed[:200]:
        print(' -', k, installed[k])
else:
    print(' - None')

print('\nVersion constraints that may not match installed versions (simple heuristic):')
if version_mismatches:
    for name, want, have in version_mismatches[:200]:
        print(f' - {name}: YAML spec "{want}" vs installed "{have}"')
else:
    print(' - None')

print('\nDone.')
