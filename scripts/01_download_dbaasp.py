# 01_download_dbaasp.py - retrieve one JSON record per peptide from the DBAASP web service.
# Input : peptides.csv (column 'ID'; the list of DBAASP peptides that have synergy data).
# Output: raw/<ID>.json for every peptide. Safe to re-run: files already present are skipped.
import requests, json, os, time, pandas as pd

ids = pd.read_csv("peptides.csv")["ID"].tolist()
os.makedirs("raw", exist_ok=True)

s = requests.Session()
s.headers.update({"Accept": "application/json",
                  "User-Agent": "Mozilla/5.0 (Synergy-AMP capstone)"})

def get_card(pid, tries=5):
    last = None
    for k in range(tries):
        try:
            r = s.get(f"https://dbaasp.org/peptides/{pid}", timeout=30)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    last = f"non-JSON: {r.text[:150]!r}"
            else:
                last = f"HTTP {r.status_code}: {r.text[:150]!r}"
        except requests.exceptions.RequestException as e:
            last = str(e)[:150]
        time.sleep(2 * (k + 1))          # wait longer after each failed try
    raise RuntimeError(last)

c = get_card(ids[0])                      # connectivity check on the first record
print("test OK:", c.get("id"), c.get("name"), "| synergies:", len(c.get("synergies") or []))

failed = []
for i, pid in enumerate(ids):
    path = f"raw/{pid}.json"
    if os.path.exists(path):
        continue
    try:
        json.dump(get_card(pid), open(path, "w"))
    except Exception as e:
        failed.append(pid)
        if len(failed) <= 5:
            print("failed", pid, e)
    time.sleep(1.0)                       # rate limit: at most one request per second
    if i % 50 == 0:
        print(f"{i}/{len(ids)} done, {len(failed)} failed so far")
print("failed total:", len(failed))
