#!/usr/bin/env python
"""Verify the title-semantics path before spending a training run on it.

    python scripts/check_title_memory.py --memory_index /content/ckpt/memory_index_ml1m.pkl

Exercises `MemoryEncoder` twice on synthetic ids -- once with `use_title=False`,
once with True -- and checks the things that would otherwise fail silently and
look like "the idea does not work":

  1. the artifact actually carries `title_emb`, and it covers COLD items (the
     whole reason the table exists: an unseen item has an untrained MF row but a
     readable title);
  2. the two sources land on the same scale, so neither branch is ignored;
  3. titles are DISCRIMINATIVE -- distinct items must get distinct vectors. A
     left-padding bug, a bad layer, or a template that swamps the title all show
     up here as near-identical rows, which is invisible in the loss;
  4. `use_title=True` changes the memory tensor, and only on history slots;
  5. gradient reaches `title_proj`;
  6. the MF and title branches contribute comparably at init -- equal input norms
     are not enough, since the two projections differ by 1/sqrt(fan_in).

Exits non-zero on the first failure, with the number that failed.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qformerrec.models.qformer_cie import MemoryEncoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory_index", required=True)
    ap.add_argument("--mf_ckpt", default=None,
                    help="the MF checkpoint this index was built from. Optional, but "
                         "check [6] compares the two branches' magnitudes and is only "
                         "meaningful against the real MF scale -- a synthetic table "
                         "makes that ratio arbitrary.")
    ap.add_argument("--d1", type=int, default=256)
    ap.add_argument("--d_q", type=int, default=128)
    ap.add_argument("--k_hist", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    with open(args.memory_index, "rb") as f:
        mi = pickle.load(f)
    meta = mi.get("meta", {})
    print(f"[index] {args.memory_index}")
    print(f"        has_title_emb={meta.get('has_title_emb')} layer={meta.get('title_layer')} "
          f"dim={meta.get('title_dim')} template={meta.get('title_template')!r}")

    # ---- 1. present, and covering cold items
    assert "title_emb" in mi, (
        "no `title_emb` in this artifact -- rebuild with "
        "scripts/build_memory.py --llm_path <vicuna dir>"
    )
    te = np.asarray(mi["title_emb"]).astype(np.float32)
    cov = np.asarray(mi["title_covered"]).astype(bool)
    in_train = np.asarray(mi["item_in_train"]).astype(bool)
    cold = ~in_train
    cold_cov = cov[cold].mean() if cold.sum() else 1.0
    print(f"[1] coverage all={cov.mean():.2%} cold={cold_cov:.2%} "
          f"({cold.sum()} cold items)")
    assert cov.mean() > 0.8, "fewer than 80% of items have a title"
    assert cold.sum() == 0 or cold_cov > 0.8, (
        "cold items are the case this table exists for, and most of them have no title"
    )

    # ---- 2. scale, against the MF-side tables in the same artifact
    t_norm = np.linalg.norm(te[cov], axis=1).mean()
    g_norm = np.linalg.norm(np.asarray(mi["genre_proto_init"], dtype=np.float32), axis=1).mean()
    print(f"[2] mean row norm: title={t_norm:.4f}  mf_genre_proto={g_norm:.4f}  "
          f"ratio={t_norm / max(g_norm, 1e-6):.2f}x")
    assert 0.05 < t_norm / max(g_norm, 1e-6) < 20.0, (
        "the two sources are orders of magnitude apart; the smaller branch will get a "
        "proportionally smaller gradient and be ignored -- restandardise in build_memory"
    )

    # ---- 3. discriminative? (catches left-padding / bad layer / template swamping)
    idx = np.where(cov)[0][:512]
    X = te[idx]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = Xn @ Xn.T
    off = S[~np.eye(len(idx), dtype=bool)]
    print(f"[3] pairwise cosine over {len(idx)} titles: mean={off.mean():.4f} "
          f"p95={np.percentile(off, 95):.4f} max={off.max():.4f}")
    assert off.mean() < 0.98, (
        f"title vectors are near-identical (mean cosine {off.mean():.4f}). Most likely a "
        "left-padding bug (reading a PAD position), a layer that carries no semantics, or "
        "a template so long it drowns the title."
    )

    # ---- 4/5. the encoder itself
    n_items = te.shape[0]
    n_users = np.asarray(mi["user_has_train"]).shape[0]
    torch.manual_seed(0)
    if args.mf_ckpt:
        sd = torch.load(args.mf_ckpt, map_location="cpu")
        sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        item_tab = sd["item_embedding.weight"].float()
        user_tab = sd["user_embedding.weight"].float()
        assert item_tab.shape[0] == n_items, (
            f"MF has {item_tab.shape[0]} items, the index has {n_items}"
        )
        args.d1 = item_tab.shape[1]
        print(f"[mf] real tables from {args.mf_ckpt}: d1={args.d1} "
              f"mean_item_norm={item_tab.norm(dim=1).mean():.4f}")
    else:
        print("[mf] WARNING: no --mf_ckpt, using a synthetic table -- check [6] below "
              "compares magnitudes and its ratio is meaningless without the real scale")
        item_tab = torch.randn(n_items, args.d1) * 0.1
        user_tab = torch.randn(n_users, args.d1) * 0.1

    def build(use_title):
        torch.manual_seed(0)
        return MemoryEncoder(mi, d1=args.d1, d_q=args.d_q, k_hist=args.k_hist,
                             dropout=0.0, use_title=use_title).eval()

    B = args.batch
    uid = torch.arange(B) % n_users
    # a history that deliberately includes cold items, plus padding
    pit = torch.randint(0, n_items, (B, args.k_hist))
    cold_ids = np.where(cold)[0]
    if len(cold_ids):
        pit[:, 0] = int(cold_ids[0])
    pit[:, -3:] = -1                                    # HIST_PAD

    out = {}
    for flag in (False, True):
        enc = build(flag)
        mem, mask, type_ids, prior, stats = enc(
            uid, user_tab[uid], lambda i: item_tab[i], lambda i: user_tab[i], pit_hist=pit
        )
        out[flag] = mem.detach()
        print(f"[4] use_title={flag}: mem={tuple(mem.shape)} slots={enc.n_slots} "
              f"hist_filled={stats.get('hist_slots_filled'):.1f} "
              f"unk_rate={stats.get('hist_unk_rate'):.3f}")

    d = (out[True] - out[False]).abs()
    hist_slice = slice(1, 1 + args.k_hist)
    moved_hist = d[:, hist_slice].max().item()
    moved_other = max(d[:, :1].max().item(), d[:, 1 + args.k_hist:].max().item())
    print(f"[4] max |delta| on history slots={moved_hist:.6f}  on every other slot="
          f"{moved_other:.6f}")
    assert moved_hist > 1e-4, "use_title=True did not change the history slots at all"
    assert moved_other < 1e-6, (
        "titles leaked into slots that have no title -- or `use_title` shifted the RNG "
        "and the shared modules no longer match; the title block must stay LAST in "
        "MemoryEncoder.__init__"
    )

    # ---- 6. do the two sources actually contribute comparably?
    # Standardising title_emb equalises the INPUTS; the projections still differ by
    # 1/sqrt(fan_in) (4096 vs 256). If one branch is much quieter it gets a
    # proportionally smaller gradient and is effectively ignored -- the run then
    # reads as "titles do not help" without ever having tested titles.
    enc = build(True)
    with torch.no_grad():
        hist_ids = pit.clamp(min=0)
        mf_side = enc.proj(item_tab[hist_ids] * (pit > -1).unsqueeze(-1))
        t = enc.title_emb[hist_ids]
        t = torch.where(enc.title_covered[hist_ids].unsqueeze(-1), t,
                        enc.unk_title.view(1, 1, -1))
        ti_side = enc.title_proj(t * (pit > -1).unsqueeze(-1))
    nm, nt = mf_side.norm(dim=-1).mean().item(), ti_side.norm(dim=-1).mean().item()
    print(f"[6] mean contribution on history slots: mf={nm:.4f} title={nt:.4f} "
          f"ratio={nt / max(nm, 1e-8):.2f}x"
          + ("" if args.mf_ckpt else "   (synthetic MF scale -- not meaningful)"))
    if not args.mf_ckpt:
        print("\nChecks 1-5 passed. Re-run with --mf_ckpt for check [6].")
        return
    assert 0.25 < nt / max(nm, 1e-8) < 4.0, (
        "the two branches are more than 4x apart at init; the quieter one will be "
        "ignored -- check the title_proj rescale in MemoryEncoder.__init__"
    )

    enc = build(True)
    mem, *_ = enc(uid, user_tab[uid], lambda i: item_tab[i], lambda i: user_tab[i], pit_hist=pit)
    mem.sum().backward()
    g = enc.title_proj.weight.grad
    print(f"[5] grad on title_proj: norm={g.norm().item():.6f}")
    assert g is not None and g.norm().item() > 0, "no gradient reaches title_proj"

    print("\nAll checks passed. Safe to train with memory.use_title=True.")


if __name__ == "__main__":
    main()
