import numpy as np
from src.puzzle.probes import fit_eval_linear, fit_eval_nonlinear

def test_linear_probe_recovers_linear_signal():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(800, 8)); ytr = (Xtr[:, 0] > 0).astype(int)
    Xte = rng.normal(size=(400, 8)); yte = (Xte[:, 0] > 0).astype(int)
    res = fit_eval_linear(Xtr, ytr, Xte, yte)
    assert res["acc"] > 0.95
    assert res["auc"] > 0.95

def test_linear_probe_fails_xor_but_nonlinear_succeeds():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(2000, 8)); ytr = ((Xtr[:,0]>0) ^ (Xtr[:,1]>0)).astype(int)
    Xte = rng.normal(size=(800, 8));  yte = ((Xte[:,0]>0) ^ (Xte[:,1]>0)).astype(int)
    lin = fit_eval_linear(Xtr, ytr, Xte, yte)
    non = fit_eval_nonlinear(Xtr, ytr, Xte, yte)
    assert lin["acc"] < 0.60        # linear cannot do XOR
    assert non["acc"] > 0.90        # nonlinear can

# --- new robustness helpers ---
from src.puzzle.probes import bootstrap_gap_ci, loto_cv, identify_F

def test_bootstrap_gap_ci_positive_on_xor():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(2000,8)); ytr=((Xtr[:,0]>0)^(Xtr[:,1]>0)).astype(int)
    Xte = rng.normal(size=(1000,8)); yte=((Xte[:,0]>0)^(Xte[:,1]>0)).astype(int)
    lo, hi = bootstrap_gap_ci(Xtr, ytr, Xte, yte, n_boot=200, seed=0)
    assert lo > 0.1 and hi >= lo            # nonlinear clearly beats linear; CI is a valid interval

def test_loto_cv_runs_on_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300,8)); y=(X[:,0]>0).astype(int)
    tids = rng.integers(0,3,size=300)
    acc = loto_cv(X, y, tids, fit_eval_fn=fit_eval_linear)
    assert 0.8 < acc <= 1.0

def test_identify_F_picks_the_xor_column():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(1500,8)); Xte = rng.normal(size=(600,8))
    # feature 0 = linear, feature 1 = XOR of dims 2,3 (nonlinear)
    Ytr = np.stack([(Xtr[:,0]>0), ((Xtr[:,2]>0)^(Xtr[:,3]>0))], axis=1).astype(int)
    Yte = np.stack([(Xte[:,0]>0), ((Xte[:,2]>0)^(Xte[:,3]>0))], axis=1).astype(int)
    F, gaps = identify_F(Xtr, Ytr, Xte, Yte)
    assert F == 1
