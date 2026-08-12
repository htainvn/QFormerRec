#!/usr/bin/env python
"""Is the Q-Former's attention actually candidate-aware? CPU only, no LLM.

    python scripts/check_candidate_attention.py \
        --ckpt /content/logs/stage3_ml1m_collmstyle2/20260809231/checkpoint_best.pth \
        --memory_index /content/ckpt/memory_index_ml1m.pkl \
        --mf_ckpt /content/ckpt/mf_ml1m_d256.pth \
        --data /content/data/ml-1m/valid_ood2.pkl

Candidate-conditioned pooling is the ONE thing this architecture does that a
plain MLP on the user embedding cannot. Everything else in the design -- squeezing
a user into L tokens, reading a bias term, pooling the candidate-independent
neighbour/genre/cluster slots -- has a simpler equivalent. So whether the
attention actually varies with the candidate is the central design question, and
it needs no LLM to answer: the memory bank, the query generator and the Q-Former
are a few hundred thousand parameters at d_q=128.

The metric that settles it is `js_within_user`: for one user, take the history
attention distributions produced for several DIFFERENT candidates and measure how
far apart they are. Every per-batch average already logged (`attn_by_type`,
`per_slot`, `hist_attn_top1`) is blind to this by construction -- two rows whose
attention is completely different average to something that looks unremarkable.

Three variants are reported:

  base            as trained
  -candidate      `use_candidate=False`: queries are the learned Q0, no FiLM
  -slot_prior     `use_slot_prior=False`: drops the log(1/(rank+1)) recency bias

Read them together. If `js_within_user` is ~0 in `base`, the same user gets the
same summary no matter which item is being scored, and the conditioning is dead
however healthy the FiLM weights look. If it is ~0 in `base` but nonzero under
`-slot_prior`, the conditioning works and the recency prior is burying it -- that
is a different fix (weaken the prior) from a dead FiLM (bias the logits directly).
"""

import argparse
import os
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qformerrec.models.qformer_cie import (  # noqa: E402
    HIST_PAD,
    CandidateAwareQueryGenerator,
    CollabQFormer,
    MemoryEncoder,
)


def pit_slots(his, width):
    """Same construction as RecOODDataset._pit_hist_slots: most-recent-first."""
    used = [int(x) for x in his if int(x) != 0][-width:][::-1]
    out = np.full(width, HIST_PAD, dtype=np.int64)
    out[: len(used)] = used
    return out


def js_divergence(P):
    """Mean pairwise Jensen-Shannon divergence between the rows of P (n, k).

    JS rather than KL: symmetric, bounded by log 2, and finite when a slot has
    zero mass under one candidate and not the other -- which is exactly the case
    that matters here.
    """
    n = P.shape[0]
    if n < 2:
        return None
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    tot, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            p, q = P[i], P[j]
            m = 0.5 * (p + q)
            kl = lambda a, b: np.sum(np.where(a > 0, a * np.log(np.clip(a, 1e-12, None)
                                                               / np.clip(b, 1e-12, None)), 0.0))
            tot += 0.5 * kl(p, m) + 0.5 * kl(q, m)
            cnt += 1
    return tot / cnt


def build(mi, sd, d1, d_q, n_query, k_hist, use_candidate, use_slot_prior, n_layers, n_heads):
    torch.manual_seed(0)
    me = MemoryEncoder(mi, d1=d1, d_q=d_q, k_hist=k_hist, dropout=0.0,
                       use_slot_prior=use_slot_prior, history_source="pit").eval()
    qg = CandidateAwareQueryGenerator(d1, d_q, n_query, use_candidate=use_candidate).eval()
    qf = CollabQFormer(d_q, n_layers=n_layers, n_heads=n_heads, dropout=0.0).eval()
    report = {}
    for name, mod in (("memory_encoder", me), ("query_gen", qg), ("qformer", qf)):
        sub = {k[len(name) + 1:]: v for k, v in sd.items() if k.startswith(name + ".")}
        msg = mod.load_state_dict(sub, strict=False)
        report[name] = (len(sub), len(msg.missing_keys), len(msg.unexpected_keys))
    return me, qg, qf, report


@torch.no_grad()
def attention(me, qg, qf, uid, iid, pit, U, I):
    mem, mask, tids, prior, _ = me(uid, U[uid], lambda x: I[x], lambda x: U[x], pit_hist=pit)
    q, _ = qg(I[iid])
    bias = qg.slot_bias(tids) + torch.log(prior.clamp_min(1e-6)).unsqueeze(1)
    _, A = qf(q, mem, mask, bias)
    k = me.k_hist
    hm = mask[:, 1: 1 + k].float()
    ah = A[:, :, 1: 1 + k] * hm.unsqueeze(1)
    return ah / ah.sum(-1, keepdim=True).clamp_min(1e-9), hm      # (B, L, k), (B, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--memory_index", required=True)
    ap.add_argument("--mf_ckpt", required=True)
    ap.add_argument("--data", required=True, help="a *_ood2.pkl split")
    ap.add_argument("--d_q", type=int, default=128)
    ap.add_argument("--n_query", type=int, default=4)
    ap.add_argument("--k_hist", type=int, default=50)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--min_rows", type=int, default=4,
                    help="a user needs this many rows to have several candidates to compare")
    ap.add_argument("--max_users", type=int, default=200)
    args = ap.parse_args()

    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
    blob = torch.load(args.ckpt, map_location="cpu")
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    print(f"[ckpt] {args.ckpt}  epoch={blob.get('epoch') if isinstance(blob, dict) else '?'}")

    with open(args.memory_index, "rb") as f:
        mi = pickle.load(f)

    # MF: prefer the copy inside the checkpoint (stage 3 tunes it), else the artifact
    if "rec_encoder.user_embedding.weight" in sd:
        U = sd["rec_encoder.user_embedding.weight"].float()
        I = sd["rec_encoder.item_embedding.weight"].float()
        print("[mf] using the tuned tables from the checkpoint")
    else:
        m = torch.load(args.mf_ckpt, map_location="cpu")
        m = m["model"] if isinstance(m, dict) and "model" in m else m
        U, I = m["user_embedding.weight"].float(), m["item_embedding.weight"].float()
        print(f"[mf] using {args.mf_ckpt} (checkpoint has no rec_encoder)")
    d1 = U.shape[1]

    df = pd.read_pickle(args.data)[["uid", "iid", "his"]]
    big = df.groupby("uid").size()
    users = list(big[big >= args.min_rows].index[: args.max_users])
    df = df[df.uid.isin(users)]
    print(f"[data] {args.data}: {len(df)} rows over {len(users)} users "
          f"with >= {args.min_rows} rows each")

    uid = torch.as_tensor(df.uid.values, dtype=torch.long)
    iid = torch.as_tensor(df.iid.values, dtype=torch.long)
    pit = torch.as_tensor(np.stack([pit_slots(h, args.k_hist) for h in df.his.values]))

    print(f"\n{'variant':<14} {'js_within_user':>15} {'top1':>8} {'ent_ratio':>10}")
    for tag, cand, prior_on in (("base", True, True),
                                ("-candidate", False, True),
                                ("-slot_prior", True, False)):
        me, qg, qf, rep = build(mi, sd, d1, args.d_q, args.n_query, args.k_hist,
                                cand, prior_on, args.n_layers, args.n_heads)
        if tag == "base":
            for k, (n, miss, unexp) in rep.items():
                print(f"  [load] {k}: {n} tensors from ckpt, {miss} missing, {unexp} unexpected")
            assert rep["query_gen"][0] > 0 and rep["qformer"][0] > 0, (
                "no query_gen/qformer tensors found in this checkpoint -- wrong file?"
            )

        ah, hm = attention(me, qg, qf, uid, iid, pit, U, I)
        n_valid = hm.sum(-1).clamp(min=1.0)
        ent = -(ah * ah.clamp_min(1e-9).log()).sum(-1)
        keep = hm.sum(-1) >= 2
        ent_ratio = float((ent[keep] / n_valid[keep].log().unsqueeze(1)).mean())
        top1 = float(ah.max(-1).values.mean())

        # the number that matters: same user, different candidates
        js, npairs = [], 0
        A = ah.mean(1).numpy()                       # pool the L tokens -> (N, k)
        for u in users:
            m = (df.uid.values == u)
            if m.sum() < 2:
                continue
            d = js_divergence(A[m])
            if d is not None:
                js.append(d)
                npairs += 1
        print(f"{tag:<14} {np.mean(js):>15.5f} {top1:>8.4f} {ent_ratio:>10.4f}")

    print(f"\nusers compared: {npairs};  js is bounded by log 2 = {np.log(2):.4f}")
    print("js_within_user ~ 0  -> the same user gets the same summary for every candidate:\n"
          "                      the candidate conditioning is not operating.\n"
          "js rises under -slot_prior -> conditioning works but the recency prior buries it;\n"
          "                      weaken the prior rather than touching FiLM.")


if __name__ == "__main__":
    main()
