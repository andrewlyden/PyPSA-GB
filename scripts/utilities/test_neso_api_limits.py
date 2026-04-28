"""Test NESO API limits for larger date ranges."""
import requests

api = "https://api.neso.energy/api/3/action/datastore_search_sql"
rid = "38a18ec1-9e40-465d-93fb-301e80fd1352"

# Test 1 month for single boundary
sql = (
    f'SELECT "Constraint Group", "Date (GMT/BST)", "Limit (MW)", "Flow (MW)" '
    f'FROM "{rid}" '
    f"WHERE \"Date (GMT/BST)\" >= '2020-01-01T00:00:00' "
    f"AND \"Date (GMT/BST)\" < '2020-02-01T00:00:00' "
    f"AND \"Constraint Group\" = 'SCOTEX' "
    f'ORDER BY "Date (GMT/BST)" LIMIT 32000'
)
resp = requests.get(api, params={"sql": sql}, timeout=60)
data = resp.json()
records = data["result"]["records"]
print(f"SCOTEX 1-month: {len(records)} records (expected ~1488 = 31*48)")
if records:
    print(f"  first={records[0]['Date (GMT/BST)']}")
    print(f"  last={records[-1]['Date (GMT/BST)']}")
    limits = [float(r["Limit (MW)"]) for r in records]
    print(f"  Limit range: {min(limits):.0f} - {max(limits):.0f} MW")

# Test 1 year for single boundary
sql2 = (
    f'SELECT "Constraint Group", "Date (GMT/BST)", "Limit (MW)", "Flow (MW)" '
    f'FROM "{rid}" '
    f"WHERE \"Date (GMT/BST)\" >= '2020-01-01T00:00:00' "
    f"AND \"Date (GMT/BST)\" < '2021-01-01T00:00:00' "
    f"AND \"Constraint Group\" = 'SCOTEX' "
    f'ORDER BY "Date (GMT/BST)" LIMIT 32000'
)
resp2 = requests.get(api, params={"sql": sql2}, timeout=120)
data2 = resp2.json()
records2 = data2["result"]["records"]
print(f"\nSCOTEX 1-year: {len(records2)} records (expected ~17568 = 366*48)")
if records2:
    print(f"  first={records2[0]['Date (GMT/BST)']}")
    print(f"  last={records2[-1]['Date (GMT/BST)']}")
    limits2 = [float(r["Limit (MW)"]) for r in records2]
    print(f"  Limit range: {min(limits2):.0f} - {max(limits2):.0f} MW")
