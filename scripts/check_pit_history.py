#!/usr/bin/env python
"""Gate to run before retraining: verify the point-in-time history slots.

The Q-Former's history slots come from each row's own ``his`` column -- the same
list CoLLM renders into ``<ItemTitleList>``. That is point-in-time history, not
leakage, but it is worth proving rather than asserting, and worth proving again
whenever the data changes.

Checks (all fatal):
  1. splits are timestamp-ordered:  train.max <= valid.min <= test.min
  2. every `his` entry's (user, item) pair has a POSITIVE occurrence in the data.
     Note the phrasing: a handful of pairs appear with *both* labels (100 of them
     on Amazon-Book, 0 on ML-1M), so "the first label we happen to see is 1" is
     the wrong test -- it fails on 1792 Amazon history entries that are in fact
     legitimate positives.
  3. per-row causality, ties allowed: every history item's own timestamp is
     <= the row's timestamp -- i.e. no item from the row's future
  4. index 0 appears only as the leading padding element

Reports (not fatal, but read them):
  * provenance of the k_i history items actually used, per split. This is the
    check that the change took effect at all: with k_i=10 ML-1M test should come
    back roughly 9% train / 14% valid / 77% test. If it is mostly train, the
    per-row path is not being used.
  * the untrained-MF-row ("unk_item") rate for those slots.
  * slot fill, so you can size k_hist.

    python scripts/check_pit_history.py --data_dir data/ml-1m/ --k_hist 10 20 50
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

SPLITS = ["train", "valid", "test"]


COLS = ["uid", "iid", "label", "timestamp", "his"]


def read_split(data_dir, split):
    """Read one split, keeping only the columns we use.

    Deliberately one split at a time and never concatenated: Amazon-Book's
    train_ood2.pkl is 727 463 rows whose `his` lists expand to several GB, so
    holding all three fat frames at once is what pushes a Colab runtime over.
    """
    p = os.path.join(data_dir, f"{split}_ood2.pkl")
    assert os.path.exists(p), f"missing {p}"
    df = pd.read_pickle(p)
    missing = [c for c in COLS if c not in df.columns]
    assert not missing, f"{p} is missing columns {missing}"
    out = df[COLS].copy()
    del df
    return out


def pit_history(his):
    """The row's own history with the index-0 padding dropped (oldest first)."""
    return [int(x) for x in his if int(x) != 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--k_hist", type=int, nargs="+", default=[10, 50])
    ap.add_argument("--sample_rows", type=int, default=0,
                    help="check causality on a random subsample (0 = all rows)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---- pass 1: split ordering, the positive-pair set, and the earliest
    # timestamp per (user, item). Diagnostics only -- nothing here is fitted.
    ts, neg_pairs, span = {}, set(), {}
    pos_by_split = {s: set() for s in SPLITS}
    for split in SPLITS:
        d = read_split(args.data_dir, split)
        span[split] = (d.timestamp.min(), d.timestamp.max())
        print(f"[load] {split}: {len(d)} rows, pos_rate={d.label.mean():.4f}")
        for u, i, t, l in zip(d.uid.values, d.iid.values,
                              d.timestamp.values, d.label.values):
            k = (int(u), int(i))
            if k not in ts or t < ts[k]:
                ts[k] = t
            (pos_by_split[split] if l == 1 else neg_pairs).add(k)
        del d
    # a pair counts as positive if ANY split has it positive; provenance below
    # attributes each history item to the earliest split that has it positive
    pos_pairs = set().union(*pos_by_split.values())

    assert span["train"][1] <= span["valid"][0], (
        f"train/valid overlap: {span['train'][1]} > {span['valid'][0]}"
    )
    assert span["valid"][1] <= span["test"][0], (
        f"valid/test overlap: {span['valid'][1]} > {span['test'][0]}"
    )
    print("[ok] splits are timestamp-ordered: train <= valid <= test")

    dual = pos_pairs & neg_pairs
    if dual:
        print(f"[note] {len(dual)} (user, item) pairs appear with BOTH labels in the "
              "data (duplicate rows with conflicting labels). That is why the check "
              "below asks whether a pair has *a* positive occurrence rather than "
              "trusting the first label seen.")

    # ---- pass 2: label provenance, per-row causality, padding position,
    # and the k_hist provenance report -- all in one sweep per split.
    tot = dict(earlier=0, tie=0, later=0, no_ts=0, n=0)
    later_examples, no_pos_examples = [], []
    n_no_pos = 0
    lead0 = nonlead0 = n_checked = 0
    prov = {s: {k: Counter() for k in args.k_hist} for s in SPLITS}
    unk = {s: {k: [0, 0] for k in args.k_hist} for s in SPLITS}   # [unk, total]
    filled = {s: {k: [] for k in args.k_hist} for s in SPLITS}
    train_items = None

    for split in SPLITS:
        d = read_split(args.data_dir, split)
        if split == "train":
            train_items = set(int(x) for x in d.iid.unique())
        rows = d
        if args.sample_rows and args.sample_rows < len(d):
            rows = d.sample(args.sample_rows, random_state=args.seed)
        for u, i, t, h in zip(rows.uid.values, rows.iid.values,
                              rows.timestamp.values, rows.his.values):
            hl = list(h)
            n_checked += 1
            if hl and int(hl[0]) == 0:
                lead0 += 1
            if any(int(x) == 0 for x in hl[1:]):
                nonlead0 += 1
            hist = pit_history(hl)
            for j in hist:
                k = (int(u), j)
                if k not in pos_pairs:
                    n_no_pos += 1
                    if len(no_pos_examples) < 5:
                        no_pos_examples.append(k)
                jt = ts.get(k)
                if jt is None:
                    tot["no_ts"] += 1
                    continue
                tot["n"] += 1
                if jt < t:
                    tot["earlier"] += 1
                elif jt == t:
                    tot["tie"] += 1
                else:
                    tot["later"] += 1
                    if len(later_examples) < 5:
                        later_examples.append((int(u), j, int(jt), int(t)))
            for k_i in args.k_hist:
                used = hist[-k_i:]
                filled[split][k_i].append(len(used))
                for j in used:
                    unk[split][k_i][1] += 1
                    if train_items is not None and j not in train_items:
                        unk[split][k_i][0] += 1
                    for s2 in SPLITS:
                        if (int(u), j) in pos_by_split[s2]:
                            prov[split][k_i][s2] += 1
                            break
                    else:
                        prov[split][k_i]["?"] += 1
        del d, rows

    n_hist = tot["n"]
    assert n_no_pos == 0, (
        f"{n_no_pos} history entries have no positive occurrence anywhere "
        f"(examples: {no_pos_examples}) -- `his` is not positives-only"
    )
    print(f"[ok] every history entry's (user, item) pair has a positive occurrence "
          f"({n_hist} entries checked)")

    assert nonlead0 == 0, f"index 0 appears at a non-leading position in {nonlead0} rows"
    print(f"[ok] index 0 is leading-padding only (leading in {lead0}/{n_checked} rows)")

    assert tot["later"] == 0, (
        f"per-row causality violated: {tot['later']}/{n_hist} history items are dated "
        f"AFTER their row. Examples (uid, iid, item_ts, row_ts): {later_examples}"
    )
    print(f"[ok] per-row causality: {tot['earlier'] / n_hist:.4%} strictly earlier, "
          f"{tot['tie'] / n_hist:.4%} same-timestamp ties, 0 later "
          f"({n_hist} history entries checked)"
          + (f", {tot['no_ts']} with no timestamp" if tot["no_ts"] else ""))

    # ---- provenance report
    print("\nProvenance of the k_hist history items actually used")
    print("(this is the check that the per-row path took effect: for ML-1M test at")
    print(" k_hist=10 expect roughly 9% train / 14% valid / 77% test; Amazon-Book")
    print(" near 70 / 16 / 14)\n")
    hdr = f"{'k_hist':>6} {'split':<6} {'train':>7} {'valid':>7} {'test':>7} " \
          f"{'unk_item':>9} {'filled':>13}"
    print(hdr)
    print("-" * len(hdr))
    for k_i in args.k_hist:
        for split in ["test", "valid", "train"]:
            c = prov[split][k_i]
            n = max(sum(c.values()), 1)
            u_n, u_d = unk[split][k_i]
            print(f"{k_i:>6} {split:<6} " +
                  " ".join(f"{c[s2] / n:>6.1%} " for s2 in SPLITS) +
                  f"{u_n / max(u_d, 1):>8.2%} " +
                  f"{np.mean(filled[split][k_i]):>7.2f}/{k_i:<5}")
    print("\n`unk_item` = history item never seen in the train split, so its MF row is "
          "untrained;\nthose slots get the single learned unk_item vector and slot type "
          "SLOT_HIST_UNK.")
    print("\nall point-in-time history checks passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)
