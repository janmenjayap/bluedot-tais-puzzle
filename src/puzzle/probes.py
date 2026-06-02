import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

def _metrics(clf, Xte, yte, scores=None):
    pred = clf.predict(Xte)
    out = {"acc": float((pred == yte).mean()),
           "bal_acc": float(balanced_accuracy_score(yte, pred))}
    try:
        s = scores if scores is not None else clf.predict_proba(Xte)[:, 1]
        out["auc"] = float(roc_auc_score(yte, s))
    except ValueError:
        out["auc"] = float("nan")
    return out

def fit_eval_linear(Xtr, ytr, Xte, yte, C=1.0):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(sc.transform(Xtr), ytr)
    return _metrics(clf, sc.transform(Xte), yte)

def fit_eval_nonlinear(Xtr, ytr, Xte, yte, hidden=(64,), seed=0):
    sc = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=2000, random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    return _metrics(clf, sc.transform(Xte), yte)
