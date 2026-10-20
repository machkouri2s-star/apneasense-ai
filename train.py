import pandas as pd, numpy as np, xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix

df = pd.read_csv("features.csv")
df = df.dropna(subset=["meanrr"]).reset_index(drop=True)
feats = [c for c in df.columns if c not in ("record", "minute", "label")]

def make_model(y):
    return xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                             eval_metric="logloss", random_state=42)

df["proba"] = np.nan
for tr, te in GroupKFold(n_splits=5).split(df, df.label, df.record):
    m = make_model(df.label.iloc[tr])
    m.fit(df.iloc[tr][feats], df.label.iloc[tr])
    df.loc[df.index[te], "proba"] = m.predict_proba(df.iloc[te][feats])[:, 1]

y, p = df.label, df.proba
pred = (p >= 0.5).astype(int)
print("Validation par enregistrement (5 groupes, nuits jamais vues)")
print("Accuracy    :", round(accuracy_score(y, pred), 3))
print("Sensibilite :", round(recall_score(y, pred), 3))
print("Specificite :", round(recall_score(y, pred, pos_label=0), 3))
print("F1          :", round(f1_score(y, pred), 3))
print("AUROC       :", round(roc_auc_score(y, p), 3))
print("Matrice de confusion [[TN FP] [FN TP]] :")
print(confusion_matrix(y, pred))

df[["record", "minute", "label", "proba"]].to_csv("predictions.csv", index=False)
final = make_model(df.label)
final.fit(df[feats], df.label)
final.save_model("model.json")
print("Fichiers crees : predictions.csv, model.json")
