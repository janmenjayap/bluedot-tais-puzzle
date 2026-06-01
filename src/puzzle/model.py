import torch, torch.nn as nn

# Slice end-indices into m.layers (nn.Sequential) for each tap, post-ReLU where applicable.
# layers: 0 Lin,1 ReLU,2 Lin,3 ReLU,4 Lin,5 ReLU,6 Lin,7 ReLU,8 Lin
TAPS = {"h0": 2, "h1": 4, "h2": 6, "h3": 8, "logits": 9}  # m.layers[:TAPS[name]]

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, 8),
        )
    def forward(self, x):
        return self.layers(x)

def load_head(path="model.pt"):
    m = Head()
    # weights_only=True: model.pt is a plain state dict (tensors only), so this is
    # safe and avoids unpickling arbitrary objects. (Puzzle code used False.)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    return m
