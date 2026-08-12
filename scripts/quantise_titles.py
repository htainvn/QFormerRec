#!/usr/bin/env python
"""Write memory-index variants whose `title_emb` is quantised into G groups.

    python scripts/quantise_titles.py \
        --memory_index /content/ckpt/memory_index_ml1m_title.pkl \
        --n_clusters 18,200 \
        --out_template /content/ckpt/memory_index_ml1m_title_g{g}.pkl

WHY THIS EXISTS, AND WHY PCA DOES NOT DO IT
-------------------------------------------
A 4096-d title vector is one fixed vector per item id, so "the titles help" and
"the model was handed a free perfect ID embedding" predict the same result. The
first attempt at separating them reduced the table with PCA -- and PCA cannot do
it, by construction: it minimises reconstruction error, i.e. it *tries to keep
points apart*. Measured on ML-1M at k=16, the nearest-neighbour cosine was still
0.9882 and the near-duplicate rate stayed at 0.00%. Sixteen continuous dimensions
separate 3 255 items with enormous margin; reducing dimensionality is simply not
the same operation as reducing identifiability.

Quantisation is. After clustering, every item in a group shares *the same vector*,
so individual identity is destroyed by construction while semantics at the group's
granularity survive intact. Sweeping G traces the curve that answers the question:

    G = 18       roughly the granularity of ML-1M's genre list
    G = 200      ~16 items per vector: finer than genre, still no identity
    G = n_items  the original table

If performance saturates at a small G, the model is using semantics at that
granularity. If it only pays at full width, it is using identity. A single
have/have-not comparison cannot tell those apart; the curve can.

The script verifies its own control: after quantisation, any item whose group has
at least two members has a nearest neighbour at cosine 1.0, so `near_dup` must go
to ~100%. If it does not, the control did not take and the run would be
uninterpretable -- which is exactly the failure PCA hid.

The output is a normal memory index. Nothing in the model changes:
`memory.use_title=True` reads whatever table the artifact carries.
"""

import argparse
import copy
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans


def near_dup_rate(X, sample=512, thresh=0.99, seed=0):
    """Fraction of sampled rows whose nearest OTHER row sits above `thresh`."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], size=min(sample, X.shape[0]), replace=False)
    A = X[idx].astype(np.float32)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    S = A @ A.T
    np.fill_diagonal(S, -1.0)
    return float((S.max(axis=1) > thresh).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory_index", required=True, help="an index built with --llm_path")
    ap.add_argument("--n_clusters", default="18,200",
                    help="comma separated group counts G")
    ap.add_argument("--out_template", required=True,
                    help="output path containing '{g}', e.g. .../index_title_g{g}.pkl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    assert "{g}" in args.out_template, "--out_template must contain '{g}'"

    with open(args.memory_index, "rb") as f:
        mi = pickle.load(f)
    assert "title_emb" in mi, (
        f"{args.memory_index} has no `title_emb` -- rebuild it with "
        "scripts/build_memory.py --llm_path <vicuna dir>"
    )
    T = np.asarray(mi["title_emb"]).astype(np.float32)
    cov = np.asarray(mi["title_covered"]).astype(bool)
    obs = T[cov]
    print(f"[in ] {args.memory_index}: title_emb={T.shape} covered={cov.sum()}/{len(cov)}")
    print(f"[in ] near_dup(full width) = {near_dup_rate(obs):.2%}   "
          "<- the number PCA could not move")

    for g in [int(x) for x in args.n_clusters.split(",") if x.strip()]:
        g = min(g, obs.shape[0])
        km = KMeans(n_clusters=g, n_init=10, random_state=args.seed).fit(obs)
        cent = km.cluster_centers_.astype(np.float32)
        quant = cent[km.labels_]

        # keep the artifact's convention: unit mean row norm over covered rows, so
        # the title_proj rescale and the LayerNorm after it hold unchanged
        scale = float(np.linalg.norm(quant, axis=1).mean())
        quant = quant / max(scale, 1e-6)

        out = np.zeros_like(T)
        out[cov] = quant
        sizes = np.bincount(km.labels_, minlength=g)
        nd = near_dup_rate(quant)
        evr = 1.0 - (((obs - cent[km.labels_]) ** 2).sum() /
                     ((obs - obs.mean(0)) ** 2).sum())
        print(f"[G={g:>5}] group sizes min/med/max = {sizes.min()}/{int(np.median(sizes))}/"
              f"{sizes.max()}   var_kept={evr:.3f}   near_dup={nd:.2%}")
        if nd < 0.9:
            print(f"           WARNING: near_dup is only {nd:.2%}. Identity survived the "
                  "quantisation, so this variant does NOT control for the fingerprint.")

        new = copy.copy(mi)
        new["title_emb"] = out.astype(np.float16)
        new["meta"] = dict(mi.get("meta", {}))
        new["meta"].update({
            "title_quantised_groups": int(g),
            "title_quantise_seed": args.seed,
            "title_quantise_var_kept": float(evr),
            "title_quantise_near_dup": float(nd),
            "title_quantise_source": os.path.abspath(args.memory_index),
        })
        path = args.out_template.format(g=g)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(new, f, protocol=4)
        print(f"           -> {path}")

    print("\nTrain one stage2+stage3 pair per variant with "
          "`model.qformer.memory.use_title=True` and "
          "`model.qformer.memory.memory_index_path=<variant>`; keep "
          "`title_pca_dim: 0` -- the quantisation replaces it.")


if __name__ == "__main__":
    main()
