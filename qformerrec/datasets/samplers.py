"""User-grouped batch sampler.

``L_rank`` is a within-user pairwise loss, so it only produces gradient when a
batch contains both a positive and a negative row for the same user. Random
sampling on ML-1M (740 train users, 33k rows) gives a same-user pos/neg pair
only rarely, so we co-locate ``m`` rows from each of ``U`` users per batch.
"""

import logging

import numpy as np
from torch.utils.data import BatchSampler


class UserGroupedBatchSampler(BatchSampler):
    """Sample ``n_users_per_batch`` users x ``n_per_user`` rows, mixing labels.

    For each chosen user we take up to ``ceil(m/2)`` positives and the rest
    negatives (falling back to whatever the user has). Users with fewer than 2
    usable rows are filled in from a global random pool, so batch size stays
    exactly ``U * m``.
    """

    def __init__(self, uids, labels, n_users_per_batch=8, n_per_user=6, seed=42, drop_last=True):
        self.uids = np.asarray(uids)
        self.labels = np.asarray(labels)
        self.U = int(n_users_per_batch)
        self.m = int(n_per_user)
        self.batch_size = self.U * self.m
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed)
        self.epoch = 0

        order = np.argsort(self.uids, kind="stable")
        self.by_user_pos, self.by_user_neg = {}, {}
        for u in np.unique(self.uids):
            idx = order[np.searchsorted(self.uids[order], u, "left"):
                        np.searchsorted(self.uids[order], u, "right")]
            self.by_user_pos[u] = idx[self.labels[idx] > 0.5]
            self.by_user_neg[u] = idx[self.labels[idx] <= 0.5]
        self.pairable_users = np.array(
            [u for u in self.by_user_pos
             if len(self.by_user_pos[u]) > 0 and len(self.by_user_neg[u]) > 0]
        )
        self.all_users = np.array(sorted(self.by_user_pos.keys()))
        self.n_rows = len(self.uids)
        logging.info(
            "UserGroupedBatchSampler: %d rows, %d users, %d users with both labels, "
            "batch=%dx%d=%d",
            self.n_rows, len(self.all_users), len(self.pairable_users),
            self.U, self.m, self.batch_size,
        )

    def _draw_user(self, u):
        """Up to m indices for user u, label-mixed where possible."""
        want_pos = (self.m + 1) // 2
        pos, neg = self.by_user_pos[u], self.by_user_neg[u]
        take_pos = min(want_pos, len(pos))
        take_neg = min(self.m - take_pos, len(neg))
        take_pos = min(len(pos), self.m - take_neg)   # backfill if few negatives
        out = []
        if take_pos:
            out.append(self.rng.choice(pos, take_pos, replace=False))
        if take_neg:
            out.append(self.rng.choice(neg, take_neg, replace=False))
        return np.concatenate(out) if out else np.array([], dtype=np.int64)

    def __iter__(self):
        self.epoch += 1
        n_batches = len(self)
        # prefer users that can actually form a pos/neg pair
        pool = self.pairable_users if len(self.pairable_users) >= self.U else self.all_users
        for _ in range(n_batches):
            users = self.rng.choice(pool, min(self.U, len(pool)), replace=len(pool) < self.U)
            batch = [self._draw_user(u) for u in users]
            batch = np.concatenate(batch) if batch else np.array([], dtype=np.int64)
            if len(batch) < self.batch_size:   # top up with random rows
                extra = self.rng.randint(0, self.n_rows, self.batch_size - len(batch))
                batch = np.concatenate([batch, extra])
            yield [int(i) for i in batch[: self.batch_size]]

    def __len__(self):
        return max(1, self.n_rows // self.batch_size)
