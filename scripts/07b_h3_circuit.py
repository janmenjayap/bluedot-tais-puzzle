# scripts/07b_h3_circuit.py
# Test whether two h3 neurons constitute the decoder for country's magnitude code.
# Finds the two h3 neurons with the largest equal-and-opposite loadings on the
# food axis at h2, then separates correlational probe evidence from the model's
# trained readout and a causal mean-ablation check.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.puzzle.data import FEATURE_NAMES
from src.puzzle.model import load_head

COUNTRY = FEATURE_NAMES.index("country")
FOOD = FEATURE_NAMES.index("food")

d = np.load("artifacts/activations/acts.npz")
emb_tr, emb_te = d["train_emb"], d["test_emb"]
h2_tr, h2_te = d["train_h2"], d["test_h2"]
h3_tr, h3_te = d["train_h3"], d["test_h3"]
lab_tr, lab_te = d["train_labels"], d["test_labels"]

ctr, cte = lab_tr[:, COUNTRY], lab_te[:, COUNTRY]
ftr, fte = lab_tr[:, FOOD], lab_te[:, FOOD]

# Embedding-level linear probes (used in WHY section of write-up)
acc_emb_country = LogisticRegression(C=1, max_iter=5000).fit(emb_tr, ctr).score(emb_te, cte)
acc_emb_food = LogisticRegression(C=1, max_iter=5000).fit(emb_tr, ftr).score(emb_te, fte)
print(f"Embedding linear probe — country: {acc_emb_country:.4f}, food: {acc_emb_food:.4f}")

# Food direction in raw h2 space (unit vector)
wf = LogisticRegression(C=1, max_iter=5000).fit(h2_tr, ftr).coef_[0]
wf_unit = wf / np.linalg.norm(wf)

# h3 weight matrix: model.layers[6].weight shape [64_out, 64_in]
model = load_head()
W = model.layers[6].weight.detach().numpy()  # [64, 64]

# Loading = how much each h3 neuron's input weights align with the food direction
loadings = W @ wf_unit  # [64]

idx_sorted = np.argsort(loadings)[::-1]
print("Top 5 positive loadings:")
for i in idx_sorted[:5]:
    print(f"  neuron {i:2d}: {loadings[i]:+.4f}")
print("Top 5 negative loadings:")
for i in idx_sorted[-5:]:
    print(f"  neuron {i:2d}: {loadings[i]:+.4f}")

pos_idx = int(idx_sorted[0])
neg_idx = int(idx_sorted[-1])
pos_load = float(loadings[pos_idx])
neg_load = float(loadings[neg_idx])
ratio = abs(pos_load / neg_load)

print(f"\nTop positive: neuron {pos_idx}, loading = {pos_load:+.4f}")
print(f"Top negative: neuron {neg_idx}, loading = {neg_load:+.4f}")
print(f"Ratio |pos/neg| = {ratio:.3f}  (1.0 = perfectly equal and opposite)")

# Correlational check: fit a new probe to only these two h3 activations.
acc_2_probe = (LogisticRegression(C=1, max_iter=5000)
               .fit(h3_tr[:, [pos_idx, neg_idx]], ctr)
               .predict(h3_te[:, [pos_idx, neg_idx]]) == cte).mean()

# Full linear probe at h3 (for reference)
acc_h3 = (LogisticRegression(C=1, max_iter=5000)
          .fit(h3_tr, ctr)
          .predict(h3_te) == cte).mean()

# Sufficiency under the trained output layer, without fitting a new decoder.
output = model.layers[8]
country_weights = output.weight[COUNTRY].detach().numpy()
country_bias = float(output.bias[COUNTRY].detach())
selected = np.array([pos_idx, neg_idx])
selected_logits = h3_te[:, selected] @ country_weights[selected] + country_bias
acc_2_model_readout = ((selected_logits > 0) == cte).mean()

# Necessity: replace only the selected neurons with their training-set means,
# then apply the complete trained output layer.
h3_te_ablated = h3_te.copy()
h3_te_ablated[:, selected] = h3_tr[:, selected].mean(axis=0)
ablated_logits = h3_te_ablated @ country_weights + country_bias
acc_mean_ablation = ((ablated_logits > 0) == cte).mean()
full_model_logits = h3_te @ country_weights + country_bias
acc_full_model = ((full_model_logits > 0) == cte).mean()

print(f"\nNew 2-neuron logistic probe country acc : {acc_2_probe:.4f}")
print(f"Trained readout using only 2 neurons   : {acc_2_model_readout:.4f}")
print(f"Trained readout after mean ablation    : {acc_mean_ablation:.4f}")
print(f"Unablated trained country output       : {acc_full_model:.4f}")
print(f"Full linear probe (h3) country acc: {acc_h3:.4f}")

pd.DataFrame([{
    "pos_neuron": pos_idx, "pos_loading": pos_load,
    "neg_neuron": neg_idx, "neg_loading": neg_load,
    "ratio_abs": ratio,
    "acc_2neuron_new_probe": acc_2_probe,
    "acc_2neuron_model_readout": acc_2_model_readout,
    "acc_2neuron_mean_ablation": acc_mean_ablation,
    "acc_full_model_country": acc_full_model,
    "acc_h3_linear": acc_h3,
    "acc_emb_country": acc_emb_country,
    "acc_emb_food": acc_emb_food,
}]).to_csv("artifacts/results/07b_h3_circuit.csv", index=False)
print("\nsaved 07b_h3_circuit.csv")
