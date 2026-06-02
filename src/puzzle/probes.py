import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

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

# ---------------------------------------------------------------------------
# Model registries: each value is a zero-arg factory returning a fresh estimator.
# ---------------------------------------------------------------------------
LINEAR_MODELS = {
    "logreg": lambda: LogisticRegression(C=1.0, max_iter=2000),
    "linsvm": lambda: LinearSVC(C=1.0, max_iter=5000),
    "lda":    lambda: LinearDiscriminantAnalysis(),
}
NONLINEAR_MODELS = {
    "mlp64":  lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=0),
    "mlp2":   lambda: MLPClassifier(hidden_layer_sizes=(2,),  max_iter=1000, random_state=0),
    "rbfsvm": lambda: SVC(kernel="rbf", C=10, gamma="scale"),
    "knn":    lambda: KNeighborsClassifier(n_neighbors=15),
    "gboost": lambda: GradientBoostingClassifier(random_state=0),
}

def _fit_eval(make_clf, Xtr, ytr, Xte, yte):
    """Fit estimator from factory on Xtr, evaluate on Xte. Scaler fit on train only."""
    sc = StandardScaler().fit(Xtr)
    clf = make_clf()
    clf.fit(sc.transform(Xtr), ytr)
    return float((clf.predict(sc.transform(Xte)) == yte).mean())

def _correct_vector(make_clf, Xtr, ytr, Xte, yte):
    """Return per-sample correctness vector on Xte. Scaler fit on train only."""
    sc = StandardScaler().fit(Xtr)
    clf = make_clf()
    clf.fit(sc.transform(Xtr), ytr)
    return (clf.predict(sc.transform(Xte)) == yte).astype(float)

def bootstrap_gap_ci(Xtr, ytr, Xte, yte, n_boot=1000, seed=0, lin=None, non=None):
    """95% CI on (nonlinear_acc - linear_acc), bootstrapping the test set with fixed probes.

    Parameters
    ----------
    lin, non : zero-arg callables returning fresh estimators (default: logreg / mlp64)
    """
    lin = lin if lin is not None else LINEAR_MODELS["logreg"]
    non = non if non is not None else NONLINEAR_MODELS["mlp64"]
    c_lin = _correct_vector(lin, Xtr, ytr, Xte, yte)
    c_non = _correct_vector(non, Xtr, ytr, Xte, yte)
    rng = np.random.default_rng(seed)
    n = len(yte)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(float(c_non[idx].mean() - c_lin[idx].mean()))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

def loto_cv(X, y, tids, fit_eval_fn=None, make_clf=None):
    """Leave-one-template-out CV accuracy, averaged over templates.

    Pass either ``fit_eval_fn`` (callable returning dict with key ``'acc'``)
    or ``make_clf`` (zero-arg estimator factory).
    """
    accs = []
    for t in np.unique(tids):
        tr, te = tids != t, tids == t
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        if make_clf is not None:
            accs.append(_fit_eval(make_clf, X[tr], y[tr], X[te], y[te]))
        else:
            accs.append(fit_eval_fn(X[tr], y[tr], X[te], y[te])["acc"])
    return float(np.mean(accs))

def identify_F(Xtr, Ytr, Xte, Yte):
    """Return (F_index, per-feature accuracy gaps) using logreg vs MLP at the given activations.

    Ytr / Yte : shape (n_samples, n_features) — one binary label column per feature.
    Returns the column index with the largest nonlinear–linear gap.
    """
    gaps = []
    for j in range(Ytr.shape[1]):
        l = fit_eval_linear(Xtr, Ytr[:, j], Xte, Yte[:, j])["acc"]
        n = fit_eval_nonlinear(Xtr, Ytr[:, j], Xte, Yte[:, j])["acc"]
        gaps.append(n - l)
    return int(np.argmax(gaps)), gaps
