import json
import numpy as np

FEATURE_NAMES = json.load(open("feature_names.json"))

def load_split(path):
    texts, labels, template_ids = [], [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(row["labels"])
            template_ids.append(row.get("template_id", -1))
    return texts, np.array(labels, dtype=np.int64), np.array(template_ids, dtype=np.int64)
