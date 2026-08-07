#!/usr/bin/env python
"""Gate to run before retraining: verify the point-in-time history slots.

The Q-Former's history slots come from each row's own ``his`` column -- the same
list CoLLM renders into ``<ItemTitleList>``. That is point-in-time history, not
leakage, but it is worth proving rather than asserting, and worth proving again
whenever the data changes.

Checks (all fatal):
  1. splits are timestamp-ordered:  train.max <= valid.min <= test.min
  2. `his` holds positives only    (label == 1 for every entry)
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


def load(data_dir):
    out = {}
    for s in SPLITS:
        p = os.path.join(data_dir, f"{s}_ood2.pkl")
        assert os.path.exists(p), f"missing {p}"
        out[s] = pd.read_pickle(p)[["uid", "iid", "label", "timestamp", "his"]]
        print(f"[load] {s}: {len(out[s])} rows")
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

    sp = load(args.data_dir)
    train, valid, test = sp["train"], sp["valid"], sp["test"]

    # ---- 1. split ordering -------------------------------------------------
    assert train.timestamp.max() <= valid.timestamp.min(), (
        f"train/valid overlap: {train.timestamp.max()} > {valid.timestamp.min()}"
    )
    assert valid.timestamp.max() <= test.timestamp.min(), (
        f"valid/test overlap: {valid.timestamp.max()} > {test.timestamp.min()}"
    )
    print("[ok] splits are timestamp-ordered: train <= valid <= test")

    allrows = pd.concat(sp.values(), ignore_index=True)

    # (u, i) -> earliest timestamp / label, across all splits. Diagnostic only:
    # nothing here is fitted or fed to the model.
    ts, lab = {}, {}
    for u, i, t, l in zip(allrows.uid.values, allrows.iid.values,
                          allrows.timestamp.values, allrows.label.values):
        k = (int(u), int(i))
        if k not in ts or t < ts[k]:
            ts[k] = t
        lab.setdefault(k, int(l))
    pos_by_split = {s: set(zip(d[d.label == 1].uid, d[d.label == 1].iid)) for s, d in sp.items()}
    train_items = set(train.iid.unique())

    # ---- 2/3/4. label purity, per-row causality, padding position ----------
    rows = allrows
    if args.sample_rows and args.sample_rows < len(rows):
        rows = rows.sample(args.sample_rows, random_state=args.seed)
        print(f"[note] causality checked on a {len(rows)}-row subsample")

    lead0 = nonlead0 = 0
    labels = Counter()
    earlier = tie = later = 0
    later_examples = []
    for u, i, t, h in zip(rows.uid.values, rows.iid.values,
                          rows.timestamp.values, rows.his.values):
        hl = list(h)
        if hl and int(hl[0]) == 0:
            lead0 += 1
        if any(int(x) == 0 for x in hl[1:]):
            nonlead0 += 1
        for j in pit_history(hl):
            k = (int(u), j)
            labels[lab.get(k, "unknown")] += 1
            jt = ts.get(k)
            if jt is None:
                labels["no_timestamp"] += 1
                continue
            if jt < t:
                earlier += 1
            elif jt == t:
                tie += 1
            else:
                later += 1
                if len(later_examples) < 5:
                    later_examples.append((int(u), j, int(jt), int(t)))

    n_hist = earlier + tie + later
    assert set(labels) <= {1}, (
        f"`his` is not positives-only: label counts {dict(labels)}"
    )
    print(f"[ok] `his` holds positives only ({labels[1]} entries, all label==1)")

    assert nonlead0 == 0, f"index 0 appears at a non-leading position in {nonlead0} rows"
    print(f"[ok] index 0 is leading-padding only (leading in {lead0}/{len(rows)} rows)")

    assert later == 0, (
        f"per-row causality violated: {later}/{n_hist} history items are dated AFTER "
        f"their row. Examples (uid, iid, item_ts, row_ts): {later_examples}"
    )
    print(f"[ok] per-row causality: {earlier / n_hist:.4%} strictly earlier, "
          f"{tie / n_hist:.4%} same-timestamp ties, 0 later "
          f"({n_hist} history entries checked)")

    # ---- provenance report -------------------------------------------------
    print("\nProvenance of the k_hist history items actually used")
    print("(this is the check that the per-row path took effect: for ML-1M test at")
    print(" k_hist=10 expect roughly 9% train / 14% valid / 77% test)\n")
    hdr = f"{'k_hist':>6} {'split':<6} {'train':>7} {'valid':>7} {'test':>7} " \
          f"{'unk_item':>9} {'filled':>13}"
    print(hdr)
    print("-" * len(hdr))
    for k_i in args.k_hist:
        for split in ["test", "valid", "train"]:
            d = sp[split]
            c, unk, n, filled = Counter(), 0, 0, []
            for u, h in zip(d.uid.values, d.his.values):
                used = pit_history(h)[-k_i:]
                filled.append(len(used))
                for i in used:
                    n += 1
                    for s in SPLITS:
                        if (u, i) in pos_by_split[s]:
                            c[s] += 1
                            break
                    else:
                        c["?"] += 1
                    if i not in train_items:
                        unk += 1
            tot = max(sum(c.values()), 1)
            print(f"{k_i:>6} {split:<6} " +
                  " ".join(f"{c[s] / tot:>6.1%} " for s in SPLITS) +
                  f"{unk / max(n, 1):>8.2%} " +
                  f"{np.mean(filled):>7.2f}/{k_i:<5}")
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
