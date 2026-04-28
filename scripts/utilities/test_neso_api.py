"""Quick test of NESO Day-Ahead Constraint API fetch."""
import requests
import json

api = "https://api.neso.energy/api/3/action/datastore_search_sql"
rid = "38a18ec1-9e40-465d-93fb-301e80fd1352"

sql = (
    f'SELECT "Constraint Group", "Date (GMT/BST)", "Limit (MW)", "Flow (MW)" '
    f'FROM "{rid}" '
    f"WHERE \"Date (GMT/BST)\" >= '2020-01-07T00:00:00' "
    f"AND \"Date (GMT/BST)\" < '2020-01-09T01:00:00' "
    f"AND \"Constraint Group\" IN ('SCOTEX','SSHARN','SSE-SP','ESTEX') "
    f'ORDER BY "Constraint Group", "Date (GMT/BST)"'
)

print(f"SQL: {sql}\n")

resp = requests.get(api, params={"sql": sql}, timeout=60)
data = resp.json()
print(f"Success: {data['success']}")
records = data["result"]["records"]
print(f"Records: {len(records)}")

# Show first few per group
from collections import Counter
groups = Counter(r["Constraint Group"] for r in records)
print(f"\nGroups: {dict(groups)}")

for r in records[:5]:
    print(f"  {r['Constraint Group']:10s} {str(r['Date (GMT/BST)']):25s} limit={r['Limit (MW)']:>8} flow={r['Flow (MW)']:>8}")

print("\n--- Sample per group ---")
seen = set()
for r in records:
    g = r["Constraint Group"]
    if g not in seen:
        seen.add(g)
        print(f"  {g:10s} limit={float(r['Limit (MW)']):8.0f} MW  flow={float(r['Flow (MW)']):8.0f} MW")

# Check if we hit the API limit (pagination)
print(f"\nTotal records returned: {len(records)}")
print("NOTE: If exactly 200, we may be hitting the default API limit. Need to add LIMIT clause.")

# Test single boundary query to check if we get more records
print("\n--- Single boundary query (SCOTEX, 7 days) ---")
sql3 = (
    f'SELECT "Constraint Group", "Date (GMT/BST)", "Limit (MW)", "Flow (MW)" '
    f'FROM "{rid}" '
    f"WHERE \"Date (GMT/BST)\" >= '2020-01-01T00:00:00' "
    f"AND \"Date (GMT/BST)\" < '2020-01-08T00:00:00' "
    f"AND \"Constraint Group\" = 'SCOTEX' "
    f'ORDER BY "Date (GMT/BST)" LIMIT 32000'
)
resp3 = requests.get(api, params={"sql": sql3}, timeout=60)
data3 = resp3.json()
records3 = data3["result"]["records"]
print(f"SCOTEX 7-day query: {len(records3)} records")
if records3:
    dates3 = [r["Date (GMT/BST)"] for r in records3]
    print(f"  first={dates3[0]}, last={dates3[-1]}")
    limits = [float(r["Limit (MW)"]) for r in records3]
    print(f"  Limit range: {min(limits):.0f} - {max(limits):.0f} MW")
