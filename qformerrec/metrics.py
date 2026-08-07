"""Evaluation metrics, kept free of any ``minigpt4`` import.

``scripts/pretrain_mf.py`` needs UAUC without dragging in the whole CoLLM package,
and the task module needs the same function, so it lives here.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def uauc_score(users, scores, labels):
    """Per-user AUC averaged over users -- CoLLM's UAUC, version-independently.

    CoLLM's ``uAUC_me`` wraps ``roc_auc_score`` in a bare ``except`` to skip
    users whose rows are all one class. That relied on scikit-learn *raising*
    for single-class input; since sklearn ~1.3 it returns ``nan`` with an
    ``UndefinedMetricWarning`` instead, so those users are no longer skipped and
    ``auc_for_user.mean()`` comes out ``nan``. On the released ML-1M valid split
    that is 44 of 283 evaluable users -- i.e. UAUC, the metric this whole
    project is selected and reported on, silently becomes NaN.

    This computes what ``uAUC_me`` intends on any sklearn version: skip users
    with fewer than 2 rows or only one label present, average the rest. It is
    numerically identical to ``uAUC_me`` under sklearn <= 1.2.
    """
    users = np.asarray(users).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    assert users.shape == scores.shape == labels.shape, (
        users.shape, scores.shape, labels.shape
    )
    assert np.isfinite(scores).all(), "non-finite scores reached the UAUC computation"

    order = np.argsort(users, kind="stable")
    u_sorted = users[order]
    bounds = np.flatnonzero(np.diff(u_sorted)) + 1
    groups = np.split(order, bounds)

    aucs, n_single_row, n_single_class = [], 0, 0
    for idx in groups:
        if len(idx) < 2:
            n_single_row += 1
            continue
        y = labels[idx]
        if y.min() == y.max():
            n_single_class += 1
            continue
        aucs.append(roc_auc_score(y, scores[idx]))
    stats = {
        "n_users": len(groups),
        "n_scored": len(aucs),
        "n_single_row": n_single_row,
        "n_single_class": n_single_class,
    }
    if not aucs:
        return float("nan"), stats
    uauc = float(np.mean(aucs))
    assert np.isfinite(uauc), "UAUC is not finite"
    return uauc, stats
