import wfdb, numpy as np, pandas as pd
from scipy.signal import welch

FS = 100
records = [f"a{i:02d}" for i in range(1, 21)] + [f"b{i:02d}" for i in range(1, 6)] + [f"c{i:02d}" for i in range(1, 11)]

def band(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return np.sum(p[m]) * (f[1] - f[0]) if m.any() else np.nan

rows = []
for r in records:
    qrs = wfdb.rdann(f"data/{r}", "qrs").sample
    apn = wfdb.rdann(f"data/{r}", "apn")
    t = qrs[1:] / FS
    rr_all = np.diff(qrs) / FS
    ok = (rr_all > 0.3) & (rr_all < 2.0)
    t, rr_all = t[ok], rr_all[ok]
    for minute, (s, sym) in enumerate(zip(apn.sample, apn.symbol)):
        a = s / FS
        b = a + 60
        rr = rr_all[(t >= a) & (t < b)]
        row = {"record": r, "minute": minute, "label": int(sym == "A")}
        if len(rr) >= 20:
            d = np.diff(rr)
            row.update(meanrr=rr.mean(), sdnn=rr.std(), rmssd=np.sqrt(np.mean(d**2)),
                       pnn50=np.mean(np.abs(d) > 0.05), minrr=rr.min(), maxrr=rr.max(),
                       medrr=np.median(rr), cv=rr.std() / rr.mean(), nbeats=len(rr))
            w = (t >= a - 120) & (t < b + 120)
            if w.sum() >= 100:
                tt, rv = t[w], rr_all[w]
                grid = np.arange(tt[0], tt[-1], 0.25)
                x = np.interp(grid, tt, rv)
                x = x - x.mean()
                f, p = welch(x, fs=4, nperseg=min(len(x), 512))
                vlf, lf, hf = band(f, p, 0.01, 0.04), band(f, p, 0.04, 0.15), band(f, p, 0.15, 0.4)
                tot = vlf + lf + hf
                row.update(vlf=vlf / tot, lf=lf / tot, hf=hf / tot, lfhf=lf / (hf + 1e-9), power=tot)
        rows.append(row)

df = pd.DataFrame(rows)
for col in ["meanrr", "sdnn", "rmssd", "pnn50", "vlf", "lf", "hf"]:
    g = df.groupby("record")[col]
    df[col + "_z"] = (df[col] - g.transform("mean")) / (g.transform("std") + 1e-9)
for col in ["meanrr_z", "sdnn_z", "rmssd_z", "vlf_z"]:
    for k in (-3, -2, -1, 1, 2, 3):
        df[f"{col}_s{k}"] = df.groupby("record")[col].shift(k)
    for w in (5, 11):
        df[f"{col}_r{w}"] = df.groupby("record")[col].transform(lambda x: x.rolling(w, center=True, min_periods=1).mean())

df.to_csv("features.csv", index=False)
print("Minutes :", len(df), "| Apnee :", round(100 * df.label.mean(), 1), "%")
print("Colonnes :", df.shape[1], "| Enregistrements :", df.record.nunique())
