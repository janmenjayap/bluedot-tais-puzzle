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
