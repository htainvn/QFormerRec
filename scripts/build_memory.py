#!/usr/bin/env python
"""Stage 0b: build the offline collaborative memory index.

Inputs : a CoLLM-processed dataset directory (``{train,valid,test}_ood2.pkl``)
         and a pretrained MF checkpoint (``state_dict`` of
         ``minigpt4.models.rec_base_models.MatrixFactorization``).
Output : ``memory_index_{name}.pkl`` consumed by ``mini_gpt4rec_qformer``.

LEAKAGE RULES (non-negotiable, see spec 3.1)
  * every interaction-derived structure is computed from the TRAIN split only;
  * valid/test pickles are opened for two things only: the user/item id ranges
    (so the tables line up with MF) and the item title -> genre map, which is
    item metadata, not interactions;
  * we assert the splits are timestamp-ordered, which is what makes a test
    sample's memory train-only by construction.

Contents of the artifact:
  item_in_train      (I,)            bool: did MF ever see this item in training?
  hist_items         (U, k_hist)     ABLATION ONLY -- most recent *train* positives.
                                     The default history path is per-row and
                                     point-in-time (the dataset's `PitHistItems`,
                                     built from each row's own `his` column), so
                                     nothing here is needed for it. This table only
                                     serves `history_source: train_only`, the
                                     `-pit-history` ablation row.
  neighbors          (U, k_neighbor) cosine-similar users, -1 padded
  neighbor_sim       (U, k_neighbor) the similarities (attention prior)
  genres             (U, k_genre)    top genres/categories, -1 padded
  user_cluster       (U, k_cluster)  own KMeans cluster + 2 nearest centroids
  user_has_train     (U,)            bool: has >=1 train positive
  genre_proto_init   (G, d1)         mean MF item embedding per genre
  cluster_proto_init (K, d1)         KMeans centroids of MF user embeddings
  meta               dict
"""

import argparse
import hashlib
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# ML-1M's 18 canonical genres (movies.dat)
ML1M_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def sha256(path, limit=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
def load_splits(data_dir, need_titles):
    """Read the three splits, keeping only the columns we actually use.

    The Amazon-Book ``train_ood2.pkl`` is ~500 MB on disk and expands to several
    GB in memory because of the per-row ``his`` / ``his_title`` lists, so each
    frame is slimmed immediately after reading and the fat one is released.
    """
    paths = {s: os.path.join(data_dir, f"{s}_ood2.pkl") for s in ["train", "valid", "test"]}
    for s, p in paths.items():
        assert os.path.exists(p), f"missing {p}"
    keep = ["uid", "iid", "label", "timestamp"] + (["title"] if need_titles else [])

    def _read(p):
        df = pd.read_pickle(p)
        missing = [c for c in keep if c not in df.columns]
        assert not missing, f"{p} is missing columns {missing}"
        out = df[keep].copy()
        del df
        return out

    train, valid, test = (_read(paths[s]) for s in ["train", "valid", "test"])

    # ---- leakage assertion: the splits must be time-ordered
    assert train.timestamp.max() <= valid.timestamp.min(), (
        "train/valid timestamps overlap -- a train-only memory would no longer "
        f"be leak-free by construction (train.max={train.timestamp.max()}, "
        f"valid.min={valid.timestamp.min()})"
    )
    assert valid.timestamp.max() <= test.timestamp.min(), (
        f"valid/test timestamps overlap (valid.max={valid.timestamp.max()}, "
        f"test.min={test.timestamp.min()})"
    )
    print(
        f"[splits] train={len(train)} valid={len(valid)} test={len(test)}  "
        f"time-ordered OK  train.ts=[{train.timestamp.min()}, {train.timestamp.max()}]"
    )
    return train, valid, test, paths


def build_history(train_pos, n_users, k_hist):
    """ABLATION ONLY: top-k most recent liked items per user, TRAIN positives only.

    Not used by the default `history_source: pit` path, which reads each row's own
    point-in-time `his` column in the dataset. Kept so the `-pit-history` ablation
    can quantify what the point-in-time history is worth.
    """
    hist = np.full((n_users, k_hist), -1, dtype=np.int64)
    ordered = train_pos.sort_values("timestamp", ascending=False)
    for uid, grp in ordered.groupby("uid", sort=False):
        items = grp.iid.values[:k_hist]
        hist[uid, : len(items)] = items
    filled = (hist >= 0).sum(axis=1)
    print(f"[hist] users with >=1 history slot: {(filled > 0).sum()}/{n_users}, "
          f"mean filled slots={filled.mean():.2f}")
    return hist


def build_neighbors(R, n_users, k_neighbor):
    """(b) top-k cosine-similar users over the binary train user-item matrix."""
    nb = np.full((n_users, k_neighbor), -1, dtype=np.int64)
    sim_out = np.zeros((n_users, k_neighbor), dtype=np.float32)
    if k_neighbor == 0:
        return nb, sim_out
    Rn = normalize(R, norm="l2", axis=1)
    block = 512
    for start in range(0, n_users, block):
        end = min(start + block, n_users)
        S = (Rn[start:end] @ Rn.T).toarray()               # (b, U) cosine
        for r in range(end - start):
            u = start + r
            S[r, u] = 0.0                                  # zero the diagonal
            row = S[r]
            if not row.any():
                continue
            k = min(k_neighbor, int((row > 0).sum()))
            top = np.argpartition(-row, k - 1)[:k]
            top = top[np.argsort(-row[top])]
            nb[u, : len(top)] = top
            sim_out[u, : len(top)] = row[top]
    print(f"[neighbors] users with >=1 neighbour: {(nb[:, 0] >= 0).sum()}/{n_users}, "
          f"mean top1 sim={sim_out[nb[:, 0] >= 0, 0].mean() if (nb[:, 0] >= 0).any() else 0:.3f}")
    return nb, sim_out


def item_genres_from_movies_dat(movies_dat, iid2title, n_items):
    """ML-1M: parse movies.dat and map iid -> genre ids by exact title match."""
    title2genres = {}
    with open(movies_dat, "r", encoding="ISO-8859-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) < 3:
                continue
            title2genres[parts[1].strip()] = [
                ML1M_GENRES.index(g) for g in parts[2].split("|") if g in ML1M_GENRES
            ]
    item_g = [[] for _ in range(n_items)]
    hit = 0
    for iid, title in iid2title.items():
        gs = title2genres.get(str(title).strip())
        if gs:
            item_g[iid] = gs
            hit += 1
    print(f"[genre] matched {hit}/{len(iid2title)} item titles against movies.dat "
          f"({len(title2genres)} movies parsed)")
    assert hit > 0.8 * len(iid2title), (
        "fewer than 80% of item titles matched movies.dat -- check the file/encoding"
    )
    return item_g, len(ML1M_GENRES)


def item_genres_from_kmeans(item_emb, n_items, n_clusters, seed):
    """Fallback pseudo-genres: KMeans over MF *item* embeddings."""
    n_clusters = min(n_clusters, max(2, n_items - 1))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(item_emb)
    labels = km.labels_
    print(f"[genre] item-KMeans pseudo-genres: k={n_clusters}, "
          f"sizes min/med/max={np.bincount(labels).min()}/"
          f"{int(np.median(np.bincount(labels)))}/{np.bincount(labels).max()}")
    return [[int(labels[i])] for i in range(n_items)], n_clusters


def build_user_genres(train_pos, item_g, n_users, n_genres, k_genre):
    """(c) top-k genres per user, counted over TRAIN positives only."""
    genres = np.full((n_users, k_genre), -1, dtype=np.int64)
    if k_genre == 0:
        return genres
    for uid, grp in train_pos.groupby("uid", sort=False):
        counts = np.zeros(n_genres, dtype=np.int64)
        for iid in grp.iid.values:
            for g in item_g[iid]:
                counts[g] += 1
        if counts.sum() == 0:
            continue
        k = min(k_genre, int((counts > 0).sum()))
        top = np.argsort(-counts)[:k]
        genres[uid, :k] = top
    print(f"[genre] users with >=1 genre slot: {(genres[:, 0] >= 0).sum()}/{n_users}")
    return genres


def build_user_clusters(user_emb, n_users, n_clusters, k_cluster, seed):
    """(d) own KMeans cluster + the (k-1) next-nearest centroids."""
    n_clusters = min(n_clusters, max(2, n_users - 1))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(user_emb)
    centroids = km.cluster_centers_.astype(np.float32)
    # chunked squared distances: (U, K) without ever materialising (U, K, d1)
    d = np.empty((n_users, n_clusters), dtype=np.float32)
    c_sq = (centroids ** 2).sum(1)[None, :]
    for s in range(0, n_users, 4096):
        e = min(s + 4096, n_users)
        u = user_emb[s:e]
        d[s:e] = (u ** 2).sum(1)[:, None] + c_sq - 2.0 * (u @ centroids.T)
    order = np.argsort(d, axis=1)[:, :k_cluster]
    print(f"[cluster] user-KMeans: k={n_clusters}, k_cluster={k_cluster}")
    return order.astype(np.int64), centroids


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="build the collaborative memory index")
    ap.add_argument("--data_dir", required=True, help="dir with {train,valid,test}_ood2.pkl")
    ap.add_argument("--mf_ckpt", required=True, help="pretrained MF state_dict (.pth)")
    ap.add_argument("--out", required=True, help="output .pkl path")
    ap.add_argument("--dataset", default="ml1m", choices=["ml1m", "amazon_book"])
    ap.add_argument("--k_hist", type=int, default=10)
    ap.add_argument("--k_neighbor", type=int, default=8)
    ap.add_argument("--k_genre", type=int, default=3)
    ap.add_argument("--k_cluster", type=int, default=3)
    ap.add_argument("--n_user_clusters", type=int, default=None,
                    help="default: 32 for ml1m, 256 for amazon_book")
    ap.add_argument("--genre_source", default="metadata", choices=["metadata", "item_kmeans"])
    ap.add_argument("--movies_dat", default=None, help="ML-1M movies.dat (genre_source=metadata)")
    ap.add_argument("--n_item_clusters", type=int, default=32,
                    help="pseudo-genre count when genre_source=item_kmeans")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train, valid, test, paths = load_splits(
        args.data_dir, need_titles=(args.genre_source == "metadata")
    )

    # id space must match MF's tables, which were sized over all splits
    n_users = int(max(train.uid.max(), valid.uid.max(), test.uid.max())) + 1
    n_items = int(max(train.iid.max(), valid.iid.max(), test.iid.max())) + 1

    sd = torch.load(args.mf_ckpt, map_location="cpu")
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    user_emb = sd["user_embedding.weight"].float().numpy()
    item_emb = sd["item_embedding.weight"].float().numpy()
    d1 = user_emb.shape[1]
    assert user_emb.shape[0] == n_users and item_emb.shape[0] == n_items, (
        f"MF checkpoint is ({user_emb.shape[0]}, {item_emb.shape[0]}) but the data needs "
        f"({n_users}, {n_items}) -- was the MF trained on this split?"
    )
    print(f"[mf] {args.mf_ckpt}: users={n_users} items={n_items} d1={d1}")

    # ---- TRAIN POSITIVES ONLY from here on
    train_pos = train[train.label == 1][["uid", "iid", "timestamp"]].copy()
    print(f"[train] positives={len(train_pos)} users={train_pos.uid.nunique()} "
          f"items={train_pos.iid.nunique()}")

    hist_items = build_history(train_pos, n_users, args.k_hist)

    R = csr_matrix(
        (np.ones(len(train_pos), dtype=np.float32), (train_pos.uid.values, train_pos.iid.values)),
        shape=(n_users, n_items),
    )
    R.data[:] = 1.0  # binary, in case of duplicate (u, i) rows
    neighbors, neighbor_sim = build_neighbors(R, n_users, args.k_neighbor)

    # ---- genres / categories
    if args.genre_source == "metadata":
        assert args.dataset == "ml1m", (
            "genre_source=metadata is only implemented for ML-1M (movies.dat). The CoLLM "
            "Amazon-Book pickles carry no category field -- use --genre_source item_kmeans."
        )
        assert args.movies_dat and os.path.exists(args.movies_dat), (
            "--movies_dat is required for genre_source=metadata"
        )
        # item titles are metadata, not interactions: safe to read from all splits
        iid2title = {}
        for df in (train, valid, test):
            for iid, t in zip(df.iid.values, df.title.values):
                iid2title.setdefault(int(iid), t)
        item_g, n_genres = item_genres_from_movies_dat(args.movies_dat, iid2title, n_items)
    else:
        item_g, n_genres = item_genres_from_kmeans(
            item_emb, n_items, args.n_item_clusters, args.seed
        )

    genres = build_user_genres(train_pos, item_g, n_users, n_genres, args.k_genre)

    # genre prototypes: mean MF item embedding over the items in each genre
    genre_proto = np.zeros((n_genres, d1), dtype=np.float32)
    for g in range(n_genres):
        members = [i for i in range(n_items) if g in item_g[i]]
        if members:
            genre_proto[g] = item_emb[members].mean(axis=0)

    # ---- user clusters
    n_user_clusters = args.n_user_clusters or (32 if args.dataset == "ml1m" else 256)
    user_cluster, cluster_proto = build_user_clusters(
        user_emb, n_users, n_user_clusters, args.k_cluster, args.seed
    )

    user_has_train = np.zeros(n_users, dtype=bool)
    user_has_train[np.unique(train_pos.uid.values)] = True

    # Items MF actually saw. A history item outside this set has an untrained
    # (random) embedding row, so the model substitutes its learned `unk_item`
    # vector and flags the slot with a distinct type id. Derived from the train
    # split over BOTH labels: MF is trained on positives and negatives alike.
    item_in_train = np.zeros(n_items, dtype=bool)
    item_in_train[np.unique(train.iid.values)] = True
    print(f"[items] seen in train: {item_in_train.sum()}/{n_items} "
          f"({item_in_train.mean():.2%})")

    meta = {
        "dataset": args.dataset,
        "n_users": n_users,
        "n_items": n_items,
        "mf_embedding_size": d1,
        "mf_ckpt": os.path.abspath(args.mf_ckpt),
        "mf_ckpt_sha": sha256(args.mf_ckpt),
        "train_pkl_sha": sha256(paths["train"]),
        "hist_items_k": args.k_hist,   # width of the ABLATION-ONLY train-only table
        "k_neighbor": args.k_neighbor,
        "k_genre": args.k_genre,
        "k_cluster": args.k_cluster,
        "n_genres": n_genres,
        "n_user_clusters": int(cluster_proto.shape[0]),
        "genre_source": args.genre_source,
        "seed": args.seed,
        "train_only": True,
        "train_ts_max": int(train.timestamp.max()),
        "valid_ts_min": int(valid.timestamp.min()),
        "n_items_in_train": int(item_in_train.sum()),
        "hist_items_is_ablation_only": True,
    }

    out = {
        "item_in_train": item_in_train,
        "hist_items": hist_items,          # ablation only (history_source: train_only)
        "neighbors": neighbors,
        "neighbor_sim": neighbor_sim,
        "genres": genres,
        "user_cluster": user_cluster,
        "user_has_train": user_has_train,
        "genre_proto_init": genre_proto,
        "cluster_proto_init": cluster_proto,
        "meta": meta,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print("[done]", args.out)
    print(json.dumps(meta, indent=2))
    print("Slot budget is set by the MODEL config, not this artifact: "
          "1 (user) + k_hist (per-row point-in-time history) + "
          f"{args.k_neighbor} (neighbours) + {args.k_genre} (genres) + "
          f"{args.k_cluster} (clusters).")
    print("Next: verify the point-in-time history with scripts/check_pit_history.py")


if __name__ == "__main__":
    main()
