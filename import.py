import requests, json, pandas as pd

ids = pd.read_csv("peptides.csv")["ID"].tolist()

def get_card(pid):
    r = requests.get(f"https://dbaasp.org/peptides/{pid}", timeout=30)
    r.raise_for_status()
    return r.json()

card = get_card(ids[0])   # PGLa
print(card.keys() if isinstance(card, dict) else type(card))

def find_keys(obj, word="synerg", path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}/{k}"
            if word in k.lower(): print(p)
            find_keys(v, word, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]): find_keys(v, word, f"{path}[{i}]")

find_keys(card)