import os, urllib.request

BASE = "https://physionet.org/files/apnea-ecg/1.0.0/"
records = [f"a{i:02d}" for i in range(1, 21)] + [f"b{i:02d}" for i in range(1, 6)] + [f"c{i:02d}" for i in range(1, 11)]
demo = ["a01", "a02", "a03", "b01", "c01"]
os.makedirs("data", exist_ok=True)

def get(name):
    path = "data/" + name
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "déjà là"
    try:
        req = urllib.request.Request(BASE + name, headers={"User-Agent": "Mozilla/5.0"})
        content = urllib.request.urlopen(req, timeout=60).read()
        open(path, "wb").write(content)
        return "ok"
    except Exception as ex:
        return "ERREUR " + str(ex)

for n, r in enumerate(records, 1):
    print(n, "/ 35", r, [get(r + "." + e) for e in ("apn", "qrs")], flush=True)
for r in demo:
    print("démo", r, [get(r + "." + e) for e in ("hea", "dat")], flush=True)
print("Terminé")
