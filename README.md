# BlueDot Technical AI Safety Puzzle #1: Submission

This release branch contains my analysis and reproducibility materials for the
[original BlueDot puzzle](https://github.com/SamDower/bluedot-tais-puzzle#readme).
The submission identifies `country` as the nonlinearly represented feature at
`h2`, explains its sign-symmetric magnitude code on the food direction, and
constructs an interleaved circular representation for Task 3.

## Submission links

- [Read the hosted report](https://bluedot-tais-p1-janmenjayap.web.app/).
- [Browse the submission branch](https://github.com/janmenjayap/bluedot-tais-puzzle/tree/release/puzzle-1-submission).
- [View the auditable Markdown source](docs/writeup.md) or the
    [self-contained HTML](docs/report.html).
- [Download the Google Docs submission file](docs/BlueDot_Puzzle_1_Submission.docx).
- [Browse the result artifacts](artifacts/results/) and the
    [script evidence classification](scripts/README.md).
- [Open the immutable `v1.0.0` release](https://github.com/janmenjayap/bluedot-tais-puzzle/releases/tag/v1.0.0)
    at commit
    [`341d3d5`](https://github.com/janmenjayap/bluedot-tais-puzzle/commit/341d3d517d13e3fbd74a14922b5c837794e77033).

## Original puzzle

We trained a small classifier on short text inputs to predict eight binary features simultaneously, at over 95% accuracy on each:

- `number` — contains a digit or written-out number (`3`, `seven`, …)
- `question` — phrased as a question (ends in `?`, or starts with `who/what/why/…`)
- `color` — contains a color word (`red`, `blue`, …)
- `food` — mentions food (`pizza`, `apple`, `soup`, …)
- `sentiment` — has positive vs. negative sentiment
- `country` — contains a country name (`Japan`, `France`, `USA`, …)
- `person` — contains a person's name (`Alice`, `Mark`, …)
- `body_part` — contains a body-part word (`hand`, `eye`, …)

After a particular layer L of this model, seven of these features are represented linearly, where a single direction in the activation space describes that feature. However, one feature F is represented in a different way. Your job is to figure out which feature it is and how it is represented.

## Your three tasks

**1. Find F.** Identify which of the eight features is not represented linearly.

**2. Explain how F is represented.** Describe the geometric structure the model uses to represent F at layer L. Show the analysis you used to convince yourself. 

**3. [Open ended] Train a model with an even weirder representation of F.** Train your own model that encodes F (or some other feature) in a more interesting way than ours. "More interesting" is up to you to define and defend. 

## What you'll submit

A single google doc, documenting what you tried, what worked, what didn't, and what structure emerged in the trained model. We'll happily read about your failures if the path to them was thoughtful. 

## What you get

Prizes for best submissions:

- 1st place: $1,000.
- 2nd place: $750.
- 3rd place: $500.
- Honourable mentions: $250 each.

All submissions that answer parts 1 and 2 correctly will be considered for our Technical AI Safety course (featuring rapid grant and career transition grant opportunities).


## The model architecture

The model consists of the 
`sentence-transformers/all-MiniLM-L6-v2` text encoder followed by a mean pool to get a single 384-dimensional representation of that input. This is then fed through a 5 layer MLP with ReLUs between the layers. The resulting 8 logits are then fed through individual sigmoid functions to recover the predicted probabilities for the 8 features. 

![Model architecture](model_architecture.png)

The 8 probabilities don't need to sum to 1 because the eight features aren't mutually exclusive. The model was trained with per-feature binary cross-entropy across the eight outputs.


## What's in this repo

- `model.pt` — trained classifier state dict (~150 KB).
- `data/train.jsonl` — 7000 lines of
  `{"text": "...", "labels": [1, 1, 0, 0, 1, 0, 0, 1]}`. Labels indexed by
  `feature_names.json`.
- `data/test.jsonl` — 1500 lines, same format. Use this as a held-out test set.
- `feature_names.json` — the eight feature names, indexed 0–7.


## Reproduce the submission

Clone the release branch, explicitly fetch its latest remote state, and then set
up the environment:

```bash
git clone --branch release/puzzle-1-submission --single-branch \
    https://github.com/janmenjayap/bluedot-tais-puzzle.git
cd bluedot-tais-puzzle
git fetch origin release/puzzle-1-submission
git merge --ff-only FETCH_HEAD
./setup.sh
conda activate bluedot-impact-puzzle-1-py311
pytest -q
```

The setup uses Python 3.11 with the compatible Torch 2.2 / torchvision 0.17
pair declared in `requirements.txt`. It excludes user site-packages during
installation and later activated sessions, and runs `pip check` before reporting
success. The complete analysis commands and artifact checks are in the
[report's reproducibility section](https://bluedot-tais-p1-janmenjayap.web.app/#reproducibility).

For the exact submitted snapshot rather than the moving branch, fetch and detach
the immutable release before setup:

```bash
git fetch origin tag v1.0.0
git switch --detach v1.0.0
```

## Submission artifacts

- [`docs/BlueDot_Puzzle_1_Submission.docx`](docs/BlueDot_Puzzle_1_Submission.docx)
    is the single document to import into Google Docs. It includes all three
    measured figures.
- [`docs/report.html`](docs/report.html) is a self-contained rendering of
    [`docs/writeup.md`](docs/writeup.md).
- [`scripts/README.md`](scripts/README.md) distinguishes the frozen five-seed
    Task 3 result from historical, test-contaminated architecture exploration.

## Code to get you started

```python
import torch, torch.nn as nn
from sentence_transformers import SentenceTransformer

# --- 1. Define the MLP head  ---
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),   # hidden 0
            nn.Linear(64, 64),  nn.ReLU(),   # hidden 1
            nn.Linear(64, 64),  nn.ReLU(),   # hidden 2  ← non-linear activation here (post-ReLU)
            nn.Linear(64, 64),  nn.ReLU(),   # hidden 3
            nn.Linear(64, 8),                # logits
        )
    def forward(self, x):
        return self.layers(x)

# --- 2. Load encoder (downloaded from HF) and head (local file) ---
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
m = Head()
m.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=False))
m.eval()

# --- 3. Get predictions ---
texts = ["example input one", "example input two"]

with torch.no_grad():
    embeddings = torch.from_numpy(
        enc.encode(texts, convert_to_numpy=True)   # (N, 384), mean-pooled
    )
    logits = m(embeddings)                          # (N, 8)
    probs  = torch.sigmoid(logits)                  # (N, 8) — independent per feature
    preds  = (probs > 0.5).int()                    # (N, 8) — binary predictions

# --- 4. Get activations at the right spot (post-ReLU of hidden 2) ---
# layers[0:6] = Linear, ReLU, Linear, ReLU, Linear, ReLU  → output is hidden 2 post-ReLU
with torch.no_grad():
    layer2_acts = m.layers[:6](embeddings)          # (N, 64)
```
