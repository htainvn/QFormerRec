#!/usr/bin/env python
"""Blend the model's per-row scores with the bias and CF baselines. CPU, seconds.

    python scripts/blend.py --logits_dir /content/logs/eval_blend/20260812xxx \
        --data_dir /content/data/ml-1m --mf_ckpt /content/ckpt/mf_ml1m_d256.pth

WHY
---
The model and the baselines carry different, non-overlapping signal. Measured on
ML-1M test:

    item-mean (b_i)                   AUC 0.7027   UAUC 0.6750
    user propensity (b_u) added       AUC 0.7190   UAUC 0.6750   <- b_u worth +0.0163
    MF <p_u, q_i>                     AUC 0.6495   UAUC 0.6346   but 0.7211 on warm
    the model                         AUC 0.7259   UAUC 0.6925

and the model has essentially NO user information at all: shuffling the whole
memory bank costs it 0.0005 AUC. So `b_u` is +0.0163 of AUC that the model simply
does not have, and MF beats popularity on the warm split, which the model may not
fully capture either.

`b_u` is also the one component that provably CANNOT hurt UAUC: adding a constant
per user leaves every within-user ordering untouched. Measured exactly -- `item
only` and `item+user` give the identical UAUC to four decimals. So blending is the
only way to raise AUC that is mathematically incapable of costing UAUC, which a
new loss term is not.

PROTOCOL
--------
Weights are fitted on VALIDATION and applied to test. Standardisation uses
validation statistics. Test is never touched during fitting. The single-feature
`model` row must reproduce the raw evaluation numbers exactly -- a one-feature
logistic regression is a monotone transform, so AUC and UAUC are unchanged. If it
does not, the join is wrong and nothing below it should be read.

Report the weights alongside the numbers: they are interpretable, and a reviewer
will want to see how much of the result is the LLM and how much is a groupby.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qformerrec.metrics import uauc_score  # noqa: E402

# the four ML-1M split sizes; unique, so the dumps identify themselves
DEFAULT_SPLITS = {10401: "valid", 7331: "test", 4153: "test_warm", 3178: "test_cold"}
FEATURES = ["model", "b_i", "b_u", "mf"]


def logit(p, lo=0.02, hi=0.98):
    p = np.clip(p, lo, hi)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits_dir", required=True,
                    help="an eval output dir containing logits_<n>.npz")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--mf_ckpt", required=True)
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 strength")
    args = ap.parse_args()

    dumps = {}
    for p in glob.glob(os.path.join(args.logits_dir, "**", "logits_*.npz"), recursive=True):
        z = np.load(p)
        n = len(z["label"])
        name = DEFAULT_SPLITS.get(n, f"n={n}")
        dumps[name] = pd.DataFrame({k: z[k] for k in ("uid", "iid", "label", "logit")})
        print(f"[dump] {name:<10} n={n:<6} {p}")
    assert "valid" in dumps, (
        f"no validation dump found in {args.logits_dir} -- the blend must be fitted on "
        f"valid. Found: {sorted(dumps)}"
    )

    tr = pd.read_pickle(os.path.join(args.data_dir, "train_ood2.pkl"))[["uid", "iid", "label"]]
    pi, pu, pr = tr.groupby("iid").label.mean(), tr.groupby("uid").label.mean(), tr.label.mean()
    sd = torch.load(args.mf_ckpt, map_location="cpu")
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    U = sd["user_embedding.weight"].float().numpy()
    I = sd["item_embedding.weight"].float().numpy()

    def feats(d):
        return np.column_stack([
            d.logit.values.astype(np.float64),
            logit(d.iid.map(pi).fillna(pr).values.astype(np.float64)),
            logit(d.uid.map(pu).fillna(pr).values.astype(np.float64)),
            (U[d.uid.values] * I[d.iid.values]).sum(1).astype(np.float64),
        ])

    Xv = feats(dumps["valid"])
    mu, sg = Xv.mean(0), Xv.std(0) + 1e-9
    order = [s for s in ("test", "test_warm", "test_cold", "valid") if s in dumps]

    print(f"\n{'features':<22}" + "".join(f"{s:>26}" for s in order))
    for cols in (["model"], ["model", "b_u"], ["model", "b_i", "b_u"], FEATURES):
        ix = [FEATURES.index(c) for c in cols]
        m = LogisticRegression(C=args.C, max_iter=2000).fit(
            (Xv[:, ix] - mu[ix]) / sg[ix], dumps["valid"].label.values
        )                                         # FITTED ON VALIDATION ONLY
        line = f"{'+'.join(cols):<22}"
        for s in order:
            d = dumps[s]
            sc = m.decision_function((feats(d)[:, ix] - mu[ix]) / sg[ix])
            line += (f"{roc_auc_score(d.label.values, sc):>13.4f}"
                     f"{uauc_score(d.uid.values, sc, d.label.values)[0]:>13.4f}")
        print(line)
        print(" " * 22 + "w = " + str(dict(zip(cols, np.round(m.coef_[0], 3)))))

    print(f"\ncolumns are AUC then UAUC per split. The `model` row is a monotone "
          "transform of the raw score, so it must equal the numbers the eval printed; "
          "if it does not, the join is wrong.")


if __name__ == "__main__":
    main()
