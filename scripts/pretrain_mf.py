#!/usr/bin/env python
"""Stage 0: pretrain the MF collaborative model.

A parameterised version of CoLLM's ``baseline_train_mf_ood.py`` (whose paths and
hyper-parameters are hard-coded inside a commented-out ``__main__``). Same model,
loss, optimiser, early-stopping rule and defaults, so the resulting checkpoint is
interchangeable with CoLLM's:

    Adam(lr=1e-3, weight_decay=1e-4), BCEWithLogits, d=256, batch=2048,
    early stopping on validation AUC with patience=100.

Reference numbers from CoLLM (ML-1M): test AUC 0.6482, test UAUC 0.6361.
"""

import argparse
import os
import random
import sys

import numpy as np
import omegaconf
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.environ.get("COLLM_ROOT",
                                                  os.path.join(HERE, "..", "..", "CoLLM"))))

from qformerrec.compat import install_import_shims  # noqa: E402

install_import_shims()

# UAUC from qformerrec.metrics, not CoLLM's uAUC_me: the latter returns NaN on
# scikit-learn >= ~1.3 (see qformerrec/metrics.py), and this is the gate you
# compare against CoLLM's published MF row -- a NaN here would be silent.
from minigpt4.models.rec_base_models import MatrixFactorization  # noqa: E402

from qformerrec.metrics import uauc_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embedding_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--max_epoch", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cols = ["uid", "iid", "label"]
    tr = pd.read_pickle(os.path.join(args.data_dir, "train_ood2.pkl"))[cols].values
    va = pd.read_pickle(os.path.join(args.data_dir, "valid_ood2.pkl"))[cols].values
    te = pd.read_pickle(os.path.join(args.data_dir, "test_ood2.pkl"))[cols].values

    user_num = int(max(tr[:, 0].max(), va[:, 0].max(), te[:, 0].max())) + 1
    item_num = int(max(tr[:, 1].max(), va[:, 1].max(), te[:, 1].max())) + 1
    print(f"users={user_num} items={item_num} train={len(tr)} valid={len(va)} test={len(te)}")

    cfg = omegaconf.OmegaConf.create(
        {"user_num": user_num, "item_num": item_num, "embedding_size": args.embedding_size}
    )
    model = MatrixFactorization(cfg).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.BCEWithLogitsLoss()

    loaders = {
        "train": DataLoader(tr, batch_size=args.batch_size, shuffle=True),
        "valid": DataLoader(va, batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(te, batch_size=args.batch_size, shuffle=False),
    }

    @torch.no_grad()
    def evaluate(split):
        model.eval()
        pre, lab, usr = [], [], []
        for batch in loaders[split]:
            batch = batch.to(args.device)
            s = model(batch[:, 0].long(), batch[:, 1].long())
            usr.extend(batch[:, 0].cpu().numpy())
            pre.extend(s.detach().cpu().numpy())
            lab.extend(batch[:, -1].cpu().numpy())
        auc = roc_auc_score(lab, pre)
        uauc, _ = uauc_score(usr, pre, lab)
        return auc, uauc

    best_auc, best, bad = -1.0, None, 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for epoch in range(args.max_epoch):
        model.train()
        for batch in loaders["train"]:
            batch = batch.to(args.device)
            loss = criterion(model(batch[:, 0].long(), batch[:, 1].long()), batch[:, -1].float())
            opt.zero_grad()
            loss.backward()
            opt.step()

        v_auc, v_uauc = evaluate("valid")
        if v_auc > best_auc:
            t_auc, t_uauc = evaluate("test")
            best_auc, bad = v_auc, 0
            best = {"epoch": epoch, "valid_auc": v_auc, "valid_uauc": v_uauc,
                    "test_auc": t_auc, "test_uauc": t_uauc}
            torch.save(model.state_dict(), args.out)
        else:
            bad += 1
        if epoch % 10 == 0:
            print(f"epoch {epoch}: valid_auc={v_auc:.4f} valid_uauc={v_uauc:.4f} bad={bad}")
        if bad >= args.patience:
            print("early stop")
            break

    print("best:", best)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
