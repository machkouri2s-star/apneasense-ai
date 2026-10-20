import streamlit as st, pandas as pd, numpy as np, plotly.graph_objects as go
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="ApneaSense AI", layout="wide")
df = pd.read_csv("predictions.csv")

def runs(mask):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if (not v) and start is not None:
            out.append((start, i - 1)); start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out

def hit(a, b):
    return a[0] <= b[1] and b[0] <= a[1]

def rec_data(r):
    s = df[df.record == r].set_index("minute")
    return s.reindex(range(int(s.index.max()) + 1))

st.title("ApneaSense AI")
st.caption("Explainable sleep apnea detection from overnight ECG. Research and educational prototype, not a diagnostic tool.")

with st.sidebar:
    rec = st.selectbox("Recording", sorted(df.record.unique()))
    thr = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
    minlen = st.slider("Minimum event length (min)", 1, 10, 2)
    band = st.slider("Low-confidence band (+/-)", 0.05, 0.3, 0.15, 0.05)

tab1, tab2 = st.tabs(["Night timeline", "Scientific validation"])

with tab1:
    d = rec_data(rec)
    pm = (d.proba >= thr).fillna(False).values
    tm = (d.label == 1).fillna(False).values
    ev = [e for e in runs(pm) if e[1] - e[0] + 1 >= minlen]
    pred_min = sum(e[1] - e[0] + 1 for e in ev)
    u = ((d.proba - thr).abs() < band)
    c = st.columns(4)
    c[0].metric("Events detected", len(ev))
    c[1].metric("Predicted apnea time", f"{pred_min} min")
    c[2].metric("Apnea burden (pred. / expert)", f"{100 * pred_min / len(d):.0f}% / {100 * tm.sum() / len(d):.0f}%")
    c[3].metric("Low-confidence minutes", int(u.sum()))
    hrs = d.index.values / 60
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hrs, y=tm.astype(int), line_shape="hv", fill="tozeroy", name="Expert annotation", line=dict(color="#7A8F9A", width=1), opacity=0.35))
    fig.add_trace(go.Scatter(x=hrs, y=d.proba.values, name="Model probability", line=dict(color="#0E8C80")))
    fig.add_trace(go.Scatter(x=hrs[u.values], y=d.proba[u].values, mode="markers", name="Low confidence", marker=dict(color="orange", size=6)))
    for s, e in ev:
        fig.add_vrect(x0=s / 60, x1=(e + 1) / 60, fillcolor="red", opacity=0.25, line_width=0)
    fig.add_hline(y=thr, line_dash="dash", line_color="gray")
    fig.update_layout(height=420, xaxis_title="Hours since recording start", yaxis_title="P(apnea)", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red bands: predicted apnea events. Grey area: expert annotation. Orange dots: segments where the model is uncertain.")
    if ev:
        tbl = pd.DataFrame({
            "Start (h:mm)": [f"{s // 60}:{s % 60:02d}" for s, e in ev],
            "End (h:mm)": [f"{(e + 1) // 60}:{(e + 1) % 60:02d}" for s, e in ev],
            "Duration (min)": [e - s + 1 for s, e in ev],
            "Mean confidence": [round(float(d.proba.iloc[s:e + 1].mean()), 2) for s, e in ev]})
        st.dataframe(tbl, use_container_width=True)

with tab2:
    ok = df.dropna(subset=["proba"])
    y = ok.label
    pr = (ok.proba >= thr).astype(int)
    st.subheader("Minute-level (all recordings, unseen nights)")
    c = st.columns(5)
    c[0].metric("Accuracy", f"{accuracy_score(y, pr):.3f}")
    c[1].metric("Sensitivity", f"{recall_score(y, pr):.3f}")
    c[2].metric("Specificity", f"{recall_score(y, pr, pos_label=0):.3f}")
    c[3].metric("F1", f"{f1_score(y, pr):.3f}")
    c[4].metric("AUROC", f"{roc_auc_score(y, ok.proba):.3f}")
    cm = confusion_matrix(y, pr)
    st.dataframe(pd.DataFrame(cm, index=["Expert: normal", "Expert: apnea"], columns=["Pred: normal", "Pred: apnea"]))
    n_p = n_t = tp_p = tp_t = 0
    for r in df.record.unique():
        d = rec_data(r)
        pe = [e for e in runs((d.proba >= thr).fillna(False).values) if e[1] - e[0] + 1 >= minlen]
        te = runs((d.label == 1).fillna(False).values)
        n_p += len(pe); n_t += len(te)
        tp_p += sum(any(hit(p, t) for t in te) for p in pe)
        tp_t += sum(any(hit(t, p) for p in pe) for t in te)
    pe_prec = tp_p / max(n_p, 1)
    pe_rec = tp_t / max(n_t, 1)
    pe_f1 = 2 * pe_prec * pe_rec / max(pe_prec + pe_rec, 1e-9)
    st.subheader("Event-level")
    c = st.columns(3)
    c[0].metric("Event precision", f"{pe_prec:.3f}")
    c[1].metric("Event recall", f"{pe_rec:.3f}")
    c[2].metric("Event F1", f"{pe_f1:.3f}")
    st.caption("A predicted event counts as correct if it overlaps an expert event by at least one minute.")
    st.subheader("Accuracy per recording (robustness)")
    acc = ok.groupby("record").apply(lambda g: accuracy_score(g.label, (g.proba >= thr).astype(int)))
    st.bar_chart(acc)
    st.caption("Validation: 5-fold cross-validation grouped by recording, so no night appears in both training and test.")
