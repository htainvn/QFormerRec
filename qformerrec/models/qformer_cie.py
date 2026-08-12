"""Candidate-aware Q-Former CIE module for CoLLM.

Replaces CoLLM's CIE (MF embedding -> single MLP -> one soft token) with a
candidate-aware Q-Former that reads a per-user collaborative memory bank and
emits ``L`` preference tokens.

Everything here is dataset/LLM agnostic: the module consumes MF embeddings and
integer memory ids, and produces ``(B, L, d_llm)`` soft embeddings plus the
diagnostics/losses needed to keep the ``L`` tokens from collapsing.

Slot type ids (T = 6):
    0 = user itself, 1 = liked history item, 2 = neighbour user,
    3 = genre/category prototype, 4 = user-cluster prototype,
    5 = history item whose MF row was never trained (see `unk_item`)

History slots are **point-in-time and per-row**: they come from the sample's own
``his`` column, which is exactly what CoLLM renders into ``<ItemTitleList>``.
The other slot types are fitted objects (KNN graph, KMeans centroids, genre
means) and stay train-split-only, because fitting them across valid/test would
use other users' futures.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# slot type ids -- keep stable, they index `type_bias` and the logged heatmap
SLOT_USER = 0
SLOT_HIST = 1
SLOT_NEIGHBOR = 2
SLOT_GENRE = 3
SLOT_CLUSTER = 4
SLOT_HIST_UNK = 5
N_SLOT_TYPES = 6
SLOT_TYPE_NAMES = ["user", "hist", "neighbor", "genre", "cluster", "hist_unk"]

HIST_PAD = -1        # padding marker in the per-row history array

# ---- content ablations (the controls) ------------------------------------- #
# Each mode destroys the *information* in a tensor while leaving its shape and
# its per-row scale intact, so the only thing that changes between the real run
# and the control is what the numbers mean -- not how big they are, not how many
# slots there are, and not which ones are masked. That matters: a control that
# also changes the norm would confound "the content is useless" with "the norm
# moved", and the norm is exactly what LayerNorm/RMS-matching are tuned around.
ABLATION_MODES = ("none", "random", "shuffle", "mean")


PREF_TOKEN_FLAG = "<PrefTokens>"
# soft-token placeholders; resolved in order of appearance in the prompt string
SOFT_FLAGS = (PREF_TOKEN_FLAG, "<UserID>", "<TargetItemID>")
TITLE_LIST_FLAG = "<ItemTitleList>"


def soft_slot_plan(prompt, n_query, proj_token_num):
    """[(placeholder, n_tokens)] ordered by position in the prompt.

    The order matters: ``torch.nonzero`` walks the ``<unk>`` mask row-major, so
    the soft blocks must be concatenated in the same order they appear in the
    prompt or the LLM silently receives them scrambled.
    """
    counts = {
        PREF_TOKEN_FLAG: n_query,
        "<UserID>": proj_token_num,
        "<TargetItemID>": proj_token_num,
    }
    found = sorted((prompt.find(f), f) for f in SOFT_FLAGS if prompt.find(f) >= 0)
    return [(f, counts[f]) for _, f in found]


def split_title_list(titles_str):
    """Split a ``convert_title_list_v2`` string back into individual titles.

    That helper joins already-quoted titles with ``", "``, so splitting on the
    exact 3-char sequence ``", "`` is unambiguous even though titles themselves
    contain commas (e.g. ``"Shawshank Redemption, The (1994)"``).
    """
    if not titles_str or titles_str == "unkow":
        return []
    return titles_str.split('", "')


def centre_log_prior(prior, type_ids, mask, eps=1e-6):
    """Zero-mean the log attention prior WITHIN each slot type.

    The prior exists to order slots *inside* a group -- recency inside history,
    similarity inside neighbours -- but it was added to the same logit scale that
    decides competition *between* groups, on an absolute scale that differs wildly
    per type:

        user      1.0                      -> log-prior  0
        genre     1.0                      -> log-prior  0
        cluster   1.0                      -> log-prior  0
        history   1/(rank+1)               -> mean log-prior -log(k!)/k
                                              = -2.97 at k=50, -2.58 at 33 filled
        neighbour cosine sim, clamped 1e-3 -> as low as -6.91

    So an average history slot started ~3 logits below every prototype slot -- a
    19x handicap in odds -- and the handicap fell on the 50 slots that carry
    per-row information while the 6 always-present prototype slots were exempt.
    Measured consequences: 3 of the 4 query tokens put 0.89-0.98 of their mass on
    genre/cluster, neighbours were always weakest, and `type_bias` -- the parameter
    that exists precisely to arbitrate between types -- stayed at ~0.167 softmax,
    i.e. uniform, because it would have had to learn +3 just to reach parity.

    Centring per type leaves the within-group ordering untouched and removes the
    between-group shift, which is `type_bias`'s job. Masked slots are excluded from
    the mean and left alone; they are set to -inf by the attention mask anyway.
    """
    log_prior = torch.log(prior.clamp_min(eps))
    out = log_prior.clone()
    for t in range(N_SLOT_TYPES):
        sel = (type_ids == t) & mask                       # (B, N)
        if not bool(sel.any()):
            continue
        n = sel.sum(-1, keepdim=True).clamp_min(1)
        mean_t = (log_prior * sel).sum(-1, keepdim=True) / n
        out = torch.where(sel, log_prior - mean_t, out)
    return out


def ablate_content(x, mode, perm=None, eps=1e-12):
    """Return ``x`` with its information destroyed but its shape/scale kept.

    ``random``  -- keep each row's RMS *exactly*, randomise its direction. The
                   scale is detached and the noise carries no grad, so a *training*
                   run in this mode really trains with no information from ``x``.
                   The noise is renormalised to unit RMS before rescaling, rather
                   than relying on ``randn`` being unit-RMS on average: the sample
                   RMS of ``d`` standard normals deviates by ~``1/sqrt(2d)``, which
                   is ~1% at d_llm=4096 but ~18% at d_q=16. A control whose scale
                   drifts confounds "this content is useless" with "this tensor is
                   the wrong size", which is the one thing it must not do.
    ``shuffle`` -- permute along the batch axis. The strongest of the three: the
                   marginal distribution of vectors is exactly the real one, only
                   the pairing with the row is broken. Use this when the worry is
                   "maybe the model just needs vectors of the right shape".
    ``mean``    -- every row gets the batch mean, i.e. a constant. Isolates "does
                   the model use the *per-user variation*" from "does it use the
                   token at all".

    ``perm`` lets two tensors (e.g. the MF side and the title side of the same
    memory bank) be shuffled with the *same* permutation, so the control breaks
    the row pairing without also decorrelating the two sources from each other.
    """
    if mode == "none":
        return x
    if mode == "random":
        rms = x.detach().pow(2).mean(dim=-1, keepdim=True).sqrt()
        z = torch.randn_like(x)
        # an all-zero row (a padded slot) has rms 0 and stays zero either way
        z = z / z.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        return z * rms
    if mode == "shuffle":
        if perm is None:
            perm = torch.randperm(x.shape[0], device=x.device)
        return x[perm]
    if mode == "mean":
        # `.contiguous()`: the caller reshapes / index-assigns this downstream, and
        # an expanded view would silently materialise a copy at each of those.
        return x.mean(dim=0, keepdim=True).expand_as(x).contiguous()
    raise ValueError(f"unknown ablation mode {mode!r}, expected one of {ABLATION_MODES}")


def pca_reduce(X, k, covered=None, eps=1e-6):
    """Project the rows of ``X`` onto their top-``k`` principal directions.

    Fitted on the covered rows only: uncovered rows are all-zero in the artifact
    and would otherwise drag the mean. The output is rescaled to unit mean row
    norm -- the same convention ``build_memory.py`` applies to the full-width
    table -- so every downstream scale argument (the ``title_proj`` rescale, the
    LayerNorm after it) holds unchanged at any ``k``.

    Uses ``eigh`` on the (d_text, d_text) scatter matrix rather than
    ``torch.pca_lowrank``: lowrank draws a random projection, which would consume
    RNG inside ``MemoryEncoder.__init__`` and shift every module built after it,
    so ``title_pca_dim=0`` vs ``32`` at one seed would differ in more than the
    PCA. ``eigh`` is deterministic. Cost is one (d_text, d_text) gram + an eigh:
    ~67 MB and a few seconds at d_text=4096, paid once at model construction.

    Returns ``(reduced (I, k) float32, explained_variance_ratio)``.
    """
    X = X.float()
    idx = (torch.nonzero(covered).squeeze(-1) if covered is not None
           else torch.arange(X.shape[0]))
    obs = X[idx]
    mu = obs.mean(dim=0, keepdim=True)
    Xc = obs - mu
    k = int(min(k, Xc.shape[0], Xc.shape[1]))
    evals, evecs = torch.linalg.eigh(Xc.t() @ Xc)          # ascending
    V = evecs[:, -k:].flip(-1)                             # (d_text, k), descending
    proj = Xc @ V
    scale = proj.norm(dim=1).mean().clamp_min(eps)
    out = X.new_zeros(X.shape[0], k)
    out[idx] = proj / scale
    total = evals.clamp_min(0).sum().clamp_min(eps)
    return out, float((evals[-k:].clamp_min(0).sum() / total).item())


# --------------------------------------------------------------------------- #
# memory bank
# --------------------------------------------------------------------------- #
class MemoryEncoder(nn.Module):
    """Assemble the per-row collaborative memory bank.

    Two kinds of slot, with deliberately different provenance:

    * **history** (``history_source='pit'``, the default) -- the *sample's own*
      point-in-time history, passed in per row as ``PitHistItems`` and built by
      the dataset from the ``his`` column. This is exactly the list CoLLM renders
      into ``<ItemTitleList>``, so the Q-Former reads the same user behaviour the
      baseline does. Nothing is precomputed. ``history_source='train_only'``
      restores the old per-user train-split lookup for the ``-pit-history``
      ablation.
    * **neighbours / genres / clusters** -- fitted objects (KNN graph, genre
      means, KMeans centroids) from the offline artifact. These stay
      train-split-only: fitting them over valid/test would use *other users'*
      futures, which is real leakage.

    History/neighbour slots are live lookups into the MF tables, so gradients
    reach MF whenever it is unfrozen. History items whose MF row was never
    trained (0.8% on ML-1M, 10% on Amazon-Book) get a single learned
    ``unk_item`` vector and the distinct slot type ``SLOT_HIST_UNK``, so
    ``type_bias`` can learn to discount them -- feeding a randomly-initialised
    embedding row instead would inject pure noise.
    """

    def __init__(
        self,
        memory_index,
        d1,
        d_q,
        k_hist=10,
        k_neighbor=8,
        k_genre=3,
        k_cluster=3,
        dropout=0.1,
        use_slot_prior=True,
        history_source="pit",
        use_title=False,
        title_pca_dim=0,
        ablate="none",
        use_user_slot=True,
    ):
        super().__init__()
        assert history_source in ("pit", "train_only"), history_source
        assert ablate in ABLATION_MODES, f"memory.ablate={ablate!r} not in {ABLATION_MODES}"
        self.history_source = history_source
        # CONTROL: destroy the bank's content, keep its geometry. See the header
        # of `ablate_content`. Applied to the slot vectors *before* projection, so
        # `type_emb`/`rank_emb`/the slot mask/the attention prior are all still the
        # real ones -- only what the slots say is fake.
        self.ablate = ablate
        # CONTROL: mask slot 0. The user's own MF row is one slot out of 65 but
        # carries ~0.5 of the attention mass, so "the Q-Former just re-computes
        # slot 0" is not ruled out by the per-type heatmap alone. Rows that would
        # be left with no valid key at all keep it (the softmax needs one).
        self.use_user_slot = bool(use_user_slot)
        # allow ablations to switch a slot type off by asking for 0 of them
        self.k_hist = int(k_hist)
        if history_source == "train_only":
            # bounded by the offline artifact; the pit path has no such cap
            self.k_hist = min(self.k_hist, memory_index["hist_items"].shape[1])
        self.k_neighbor = min(int(k_neighbor), memory_index["neighbors"].shape[1])
        self.k_genre = min(int(k_genre), memory_index["genres"].shape[1])
        self.k_cluster = min(int(k_cluster), memory_index["user_cluster"].shape[1])
        self.use_slot_prior = use_slot_prior
        self.d1 = d1
        self.d_q = d_q

        def _buf(name, arr, dtype=torch.long):
            self.register_buffer(name, torch.as_tensor(np.asarray(arr)).to(dtype), persistent=False)

        _buf("hist_items", memory_index["hist_items"][:, : max(self.k_hist, 1)])
        _buf("neighbors", memory_index["neighbors"][:, : self.k_neighbor])
        _buf("neighbor_sim", memory_index["neighbor_sim"][:, : self.k_neighbor], torch.float)
        _buf("genres", memory_index["genres"][:, : self.k_genre])
        _buf("user_cluster", memory_index["user_cluster"][:, : self.k_cluster])
        # users with no train-split positives: their genre/cluster assignment is
        # meaningless (their MF row was never updated), so mask those slots too.
        _buf("user_has_train", memory_index["user_has_train"], torch.bool)
        # which item ids the MF actually saw in training -- train-derived, so it
        # lives in the artifact
        assert "item_in_train" in memory_index, (
            "memory index predates the point-in-time history change; rebuild it with "
            "scripts/build_memory.py so it carries `item_in_train`"
        )
        _buf("item_in_train", memory_index["item_in_train"], torch.bool)

        # a single learned stand-in for items with an untrained MF row
        self.unk_item = nn.Parameter(torch.randn(d1) * 0.02)

        # slot layout, in the order they are concatenated. History type ids are
        # filled in per row at forward time (known vs untrained item).
        self.n_slots = 1 + self.k_hist + self.k_neighbor + self.k_genre + self.k_cluster
        base_type_ids = (
            [SLOT_USER]
            + [SLOT_HIST] * self.k_hist
            + [SLOT_NEIGHBOR] * self.k_neighbor
            + [SLOT_GENRE] * self.k_genre
            + [SLOT_CLUSTER] * self.k_cluster
        )
        self.register_buffer(
            "base_type_ids", torch.tensor(base_type_ids, dtype=torch.long), persistent=False
        )

        # trainable prototypes, initialised from the offline artifact
        g_init = torch.as_tensor(np.asarray(memory_index["genre_proto_init"]), dtype=torch.float)
        c_init = torch.as_tensor(np.asarray(memory_index["cluster_proto_init"]), dtype=torch.float)
        assert g_init.shape[1] == d1 and c_init.shape[1] == d1, (
            "memory index was built with a different MF embedding size "
            f"({g_init.shape[1]}/{c_init.shape[1]}) than the model ({d1})"
        )
        self.genre_proto = nn.Embedding(g_init.shape[0], d1)
        self.cluster_proto = nn.Embedding(c_init.shape[0], d1)
        with torch.no_grad():
            self.genre_proto.weight.copy_(g_init)
            self.cluster_proto.weight.copy_(c_init)

        self.proj = nn.Linear(d1, d_q, bias=False)
        self.type_emb = nn.Embedding(N_SLOT_TYPES, d_q)
        self.rank_emb = nn.Embedding(max(self.k_hist, 1), d_q)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        nn.init.normal_(self.rank_emb.weight, std=0.02)

        # ---- title semantics on the history slots (optional second source)
        #
        # KEEP THIS BLOCK LAST. It draws from the RNG (`unk_title`, `title_proj`),
        # so constructing it earlier would shift every module built after it and
        # `use_title=True/False` at one seed would differ in far more than the
        # title path -- which is exactly what the `-title` ablation must isolate.
        #
        # Every other slot -- user, history, neighbours, genre and cluster
        # prototypes -- is a function of the same frozen MF table, so attention
        # over them is a convex combination inside one span and cannot beat the
        # point MF already fitted for that user. Measured: a 65-slot memory scored
        # the same as the 1-slot user-only ablation. `title_emb` is the one input
        # from outside that span, which is also why it covers cold items: an
        # unseen item has an untrained MF row but a readable title.
        #
        # A separate projection summed in, rather than concatenating into `proj`:
        # algebraically identical (a linear map on a concatenation is the sum of
        # two linear maps) but it leaves the shared `proj` path untouched for the
        # slot types that have no title, and `use_title=False` restores the old
        # artifact exactly.
        self.use_title = bool(use_title) and self.k_hist > 0
        if self.use_title:
            assert "title_emb" in memory_index, (
                "use_title=True but this memory index has no `title_emb` -- rebuild it "
                "with scripts/build_memory.py --llm_path <vicuna dir>"
            )
            # fp32 to match the rest of this block, which runs with autocast
            # disabled. Resident cost is (n_items x d_text x 4): ~53 MB on ML-1M,
            # ~560 MB on Amazon-Book. Non-persistent, so it never enters a
            # checkpoint -- it is rebuilt from the artifact on every load.
            t_init = torch.as_tensor(np.asarray(memory_index["title_emb"]), dtype=torch.float)
            assert t_init.shape[0] == self.item_in_train.shape[0], (
                f"title_emb covers {t_init.shape[0]} items but the artifact's item tables "
                f"have {self.item_in_train.shape[0]} -- mismatched memory index"
            )
            t_covered = torch.as_tensor(
                np.asarray(memory_index["title_covered"]), dtype=torch.bool
            )
            # CONTROL: a d_text=4096 title vector is a *fingerprint by construction*
            # -- one fixed vector per item id, i.e. a perfect ID embedding the model
            # never had to learn. Cutting it to 16-32 principal directions destroys
            # the capacity to identify an individual item while keeping the coarse
            # semantics (items of one genre stay clustered), so a run that improves
            # here was reading semantics and a run that does not was reading the id.
            self.title_pca_dim = int(title_pca_dim)
            self.title_pca_evr = None
            if self.title_pca_dim > 0:
                t_init, self.title_pca_evr = pca_reduce(t_init, self.title_pca_dim, t_covered)
            self.d_text = t_init.shape[1]
            self.register_buffer("title_emb", t_init, persistent=False)
            self.register_buffer("title_covered", t_covered, persistent=False)
            # mirrors `unk_item`: a learned stand-in for ids with no title at all
            self.unk_title = nn.Parameter(torch.randn(self.d_text) * 0.02)
            self.title_proj = nn.Linear(self.d_text, d_q, bias=False)
            # Standardising `title_emb` to unit row-norm equalises the two INPUTS,
            # but not the two projections: nn.Linear inits at 1/sqrt(fan_in), and
            # fan_in differs 16x (d_text 4096 vs d1 256), so for unit-norm inputs
            # ||W_title x|| = sqrt(d_q/3d_text) = 0.102 against ||W_mf x|| =
            # sqrt(d_q/3d1) = 0.408 -- the title branch would start 4x quieter and
            # sit under the MF branch through the LayerNorm that follows. Rescale
            # so both sources contribute equally at step 0 and the model is free to
            # reweight them from there.
            with torch.no_grad():
                self.title_proj.weight.mul_((self.d_text / d1) ** 0.5)
        else:
            self.d_text = 0
            self.title_pca_dim = 0
            self.title_pca_evr = None
        self.ln = nn.LayerNorm(d_q)
        self.drop = nn.Dropout(dropout)

    def forward(self, uid, user_emb, item_emb_fn, user_emb_fn, pit_hist=None):
        """
        Args:
            uid:          (B,) long user ids
            user_emb:     (B, d1) MF embedding of the batch users (slot 0)
            item_emb_fn:  callable (ids)->(..., d1) MF item lookup
            user_emb_fn:  callable (ids)->(..., d1) MF user lookup
            pit_hist:     (B, W) long, the row's own history, most-recent-first,
                          ``HIST_PAD`` padded. Required when history_source='pit'.
        Returns:
            mem      (B, N, d_q)
            mask     (B, N) bool, True = valid slot
            type_ids (B, N) long   (per row: history slots may be SLOT_HIST_UNK)
            prior    (B, N) float, per-slot attention prior (>0)
            stats    dict of scalars for the diagnostics log
        """
        B = uid.shape[0]
        vecs, masks, priors = [], [], []
        type_ids = self.base_type_ids.unsqueeze(0).expand(B, -1).clone()   # (B, N)
        stats = {}

        # --- slot 0: the user itself. Always valid, which guarantees at least
        # one unmasked key per row so the cross-attention softmax is finite.
        vecs.append(user_emb.unsqueeze(1))
        masks.append(torch.ones(B, 1, dtype=torch.bool, device=uid.device))
        priors.append(torch.ones(B, 1, device=uid.device))

        # --- history items: the sample's own point-in-time history
        hist_safe = hist_mask = None          # kept for the title lookup below
        if self.k_hist > 0:
            if self.history_source == "pit":
                assert pit_hist is not None, (
                    "history_source='pit' needs the per-row `PitHistItems` field; the "
                    "dataset did not provide it"
                )
                assert pit_hist.shape[1] >= self.k_hist, (
                    f"k_hist={self.k_hist} but the dataset only emits "
                    f"{pit_hist.shape[1]} history slots -- raise `pit_hist_width` in the "
                    "dataset config"
                )
                # column j is the j-th most recent, so slicing keeps recency order
                ids = pit_hist[:, : self.k_hist].long()
            else:
                ids = self.hist_items[uid][:, : self.k_hist]
            m = ids > HIST_PAD
            safe = ids.clamp(min=0)
            # items the MF never saw: one learned vector instead of a random row
            is_unk = m & (~self.item_in_train[safe])
            e = torch.where(is_unk.unsqueeze(-1), self.unk_item.view(1, 1, -1),
                            item_emb_fn(safe))
            vecs.append(e)
            masks.append(m)
            hist_safe, hist_mask = safe, m
            rank = torch.arange(self.k_hist, device=uid.device, dtype=torch.float)
            priors.append((1.0 / (rank + 1.0)).unsqueeze(0).expand(B, -1))
            type_ids[:, 1 : 1 + self.k_hist] = torch.where(
                is_unk, torch.full_like(ids, SLOT_HIST_UNK), torch.full_like(ids, SLOT_HIST)
            )
            n_valid = m.sum().clamp(min=1)
            stats["hist_unk_rate"] = float(is_unk.sum().float() / n_valid)
            stats["hist_slots_filled"] = float(m.float().sum(1).mean())

        # --- neighbour users
        if self.k_neighbor > 0:
            ids = self.neighbors[uid]
            m = ids >= 0
            e = user_emb_fn(ids.clamp(min=0))
            vecs.append(e)
            masks.append(m)
            priors.append(self.neighbor_sim[uid].clamp(min=1e-3))

        # --- genre prototypes
        if self.k_genre > 0:
            ids = self.genres[uid]
            m = (ids >= 0) & self.user_has_train[uid].unsqueeze(1)
            vecs.append(self.genre_proto(ids.clamp(min=0)))
            masks.append(m)
            priors.append(torch.ones(B, self.k_genre, device=uid.device))

        # --- cluster prototypes
        if self.k_cluster > 0:
            ids = self.user_cluster[uid]
            m = (ids >= 0) & self.user_has_train[uid].unsqueeze(1)
            vecs.append(self.cluster_proto(ids.clamp(min=0)))
            masks.append(m)
            priors.append(torch.ones(B, self.k_cluster, device=uid.device))

        raw = torch.cat(vecs, dim=1).float()               # (B, N, d1)
        mask = torch.cat(masks, dim=1)                     # (B, N)
        prior = torch.cat(priors, dim=1).float()           # (B, N)

        # CONTROL: drop the user's own slot. Kept for rows that have nothing else,
        # otherwise their cross-attention softmax would be over an all-masked key
        # set and come out NaN.
        if not self.use_user_slot:
            other = mask[:, 1:].any(dim=1) if mask.shape[1] > 1 else torch.zeros_like(mask[:, 0])
            mask[:, 0] = ~other
            stats["user_slot_kept"] = float(mask[:, 0].float().mean())

        raw = raw * mask.unsqueeze(-1)                     # zero out padded rows
        # CONTROL: ablate the slot *contents*. One permutation shared with the
        # title branch below, so `shuffle` breaks the row->memory pairing without
        # also tearing an item's MF row away from its own title.
        ablate_perm = None
        if self.ablate != "none":
            if self.ablate == "shuffle":
                ablate_perm = torch.randperm(B, device=uid.device)
            raw = ablate_content(raw, self.ablate, ablate_perm) * mask.unsqueeze(-1)
        mem = self.proj(raw) + self.type_emb(type_ids)      # type_ids is (B, N)

        # title semantics, added only on the history block. Same out-of-place shape
        # trick as the recency embedding below: build a (B, N, d_q) tensor that is
        # zero everywhere except slots [1 : 1+k_hist].
        if self.use_title and hist_safe is not None:
            t = self.title_emb[hist_safe]                              # (B, k_hist, d_text)
            t = torch.where(self.title_covered[hist_safe].unsqueeze(-1),
                            t, self.unk_title.view(1, 1, -1))
            t = t * hist_mask.unsqueeze(-1)
            if self.ablate != "none":
                t = ablate_content(t, self.ablate, ablate_perm) * hist_mask.unsqueeze(-1)
            t = self.title_proj(t)                                     # (B, k_hist, d_q)
            parts = [mem.new_zeros(B, 1, self.d_q), t]
            n_tail = self.n_slots - 1 - self.k_hist
            if n_tail > 0:
                parts.append(mem.new_zeros(B, n_tail, self.d_q))
            mem = mem + torch.cat(parts, dim=1)
        if self.k_hist > 0:
            # recency embedding on the history slots only, added out-of-place
            rank_ids = torch.arange(self.k_hist, device=uid.device)
            zeros_head = mem.new_zeros(1, self.d_q)
            n_tail = self.n_slots - 1 - self.k_hist
            parts = [zeros_head, self.rank_emb(rank_ids)]
            if n_tail > 0:
                parts.append(mem.new_zeros(n_tail, self.d_q))
            mem = mem + torch.cat(parts, dim=0).unsqueeze(0)
        mem = self.drop(self.ln(mem))

        if not self.use_slot_prior:
            prior = torch.ones_like(prior)
        return mem, mask, type_ids, prior, stats


# --------------------------------------------------------------------------- #
# candidate-aware queries
# --------------------------------------------------------------------------- #
class CandidateAwareQueryGenerator(nn.Module):
    """Base queries modulated by the candidate item via per-token FiLM.

    Per-token FiLM weights are structural anti-collapse: token k gets its own
    ``(gamma, beta)`` head, so two tokens cannot be driven to the same vector by
    the candidate. The FiLM output layer is zero-initialised so training starts
    exactly from the base queries.
    """

    def __init__(self, d1, d_q, n_query, use_candidate=True):
        super().__init__()
        self.n_query = n_query
        self.d_q = d_q
        self.use_candidate = use_candidate

        self.Q0 = nn.Parameter(torch.randn(n_query, d_q) * 0.02)
        self.cand_proj = nn.Linear(d1, d_q)
        self.cand_ln = nn.LayerNorm(d_q)
        self.film = nn.Linear(d_q, n_query * 2 * d_q)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.ln = nn.LayerNorm(d_q)

        # per-token preference over slot types; broadcast into the cross-attn
        # logits via type_ids. This is what lets token 1 learn "read history",
        # token 2 "read neighbours", ...
        self.type_bias = nn.Parameter(torch.zeros(n_query, N_SLOT_TYPES))

    def forward(self, item_emb):
        """item_emb: (B, d1) -> (queries (B, L, d_q), candidate code (B, d_q))"""
        B = item_emb.shape[0]
        c = self.cand_ln(self.cand_proj(item_emb.float()))
        q0 = self.Q0.unsqueeze(0).expand(B, -1, -1)
        if not self.use_candidate:
            return self.ln(q0), c
        gamma, beta = self.film(c).chunk(2, dim=-1)
        gamma = gamma.view(B, self.n_query, self.d_q)
        beta = beta.view(B, self.n_query, self.d_q)
        return self.ln(q0 * (1.0 + gamma) + beta), c

    def slot_bias(self, type_ids):
        """Broadcast ``type_bias`` (L, T) onto the memory slots.

        ``type_ids`` is per row -- (B, N) -- because a history slot's type depends
        on whether that particular item's MF row was ever trained.
        Returns (B, L, N); a 1-D (N,) input still works and returns (1, L, N).
        """
        if type_ids.dim() == 1:
            return self.type_bias[:, type_ids].unsqueeze(0)
        # (T, L)[(B, N)] -> (B, N, L) -> (B, L, N)
        return self.type_bias.t()[type_ids].permute(0, 2, 1)


# --------------------------------------------------------------------------- #
# attention / Q-Former blocks
# --------------------------------------------------------------------------- #
class BiasedMultiHeadAttention(nn.Module):
    """Multi-head attention with an additive per-(query, key) logit bias.

    Written out by hand instead of using ``nn.MultiheadAttention`` because we
    need (a) the additive bias and (b) the averaged attention map, and the
    ``average_attn_weights`` argument is not available on the torch version
    CoLLM pins.
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, kv, key_mask=None, attn_bias=None):
        B, Lq, _ = q.shape
        Lk = kv.shape[1]

        def split(x, L):
            return x.view(B, L, self.h, self.dh).transpose(1, 2)  # (B, h, L, dh)

        qh = split(self.q_proj(q), Lq)
        kh = split(self.k_proj(kv), Lk)
        vh = split(self.v_proj(kv), Lk)

        scores = qh @ kh.transpose(-1, -2) / math.sqrt(self.dh)   # (B, h, Lq, Lk)
        if attn_bias is not None:
            scores = scores + attn_bias.unsqueeze(1)
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (self.drop(attn) @ vh).transpose(1, 2).reshape(B, Lq, self.h * self.dh)
        return self.out_proj(out), attn.mean(dim=1)               # (B, Lq, d), (B, Lq, Lk)


class QFormerLayer(nn.Module):
    """Pre-LN layer: self-attention, then cross-attention (BLIP-2 ordering)."""

    def __init__(self, d_q, n_heads, dropout=0.1):
        super().__init__()
        self.ln_self = nn.LayerNorm(d_q)
        self.self_attn = BiasedMultiHeadAttention(d_q, n_heads, dropout)
        self.ln_cross = nn.LayerNorm(d_q)
        self.cross_attn = BiasedMultiHeadAttention(d_q, n_heads, dropout)
        self.ln_ffn = nn.LayerNorm(d_q)
        self.ffn = nn.Sequential(
            nn.Linear(d_q, 4 * d_q), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_q, d_q)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem, key_mask, attn_bias):
        h = self.ln_self(x)
        a, _ = self.self_attn(h, h)
        x = x + self.drop(a)

        h = self.ln_cross(x)
        a, attn_map = self.cross_attn(h, mem, key_mask=key_mask, attn_bias=attn_bias)
        x = x + self.drop(a)

        x = x + self.drop(self.ffn(self.ln_ffn(x)))
        return x, attn_map


class CollabQFormer(nn.Module):
    def __init__(self, d_q, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([QFormerLayer(d_q, n_heads, dropout) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d_q)

    def forward(self, queries, mem, key_mask, attn_bias):
        x = queries
        maps = []
        for layer in self.layers:
            x, m = layer(x, mem, key_mask, attn_bias)
            maps.append(m)
        A = torch.stack(maps, dim=0).mean(dim=0)   # (B, L, N) averaged over layers+heads
        return self.ln_out(x), A


# --------------------------------------------------------------------------- #
# projection to the LLM embedding space
# --------------------------------------------------------------------------- #
class LLMProjector(nn.Module):
    """d_q -> d_llm with RMS matching against the LLM embedding table.

    Soft tokens whose norm is far from the text-embedding norm are the single
    most common cause of "the soft prompt does nothing", so the output is
    LayerNorm'd (no affine) and rescaled by a *learnable* scalar initialised so
    that ``||z|| ~= mean ||E_row||``.
    """

    def __init__(self, d_q, d_llm, target_rms, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_q, 4 * d_q), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_q, d_llm)
        )
        self.ln = nn.LayerNorm(d_llm, elementwise_affine=False)
        self.scale = nn.Parameter(torch.tensor(float(target_rms)))

    def forward(self, z):
        return self.ln(self.net(z)) * self.scale


class NoNormLLMProjector(nn.Module):
    """`-norm-match` ablation: plain linear map, no LayerNorm, no RMS scaling."""

    def __init__(self, d_q, d_llm, **kwargs):
        super().__init__()
        self.net = nn.Linear(d_q, d_llm)

    def forward(self, z):
        return self.net(z)


class CFAuxHead(nn.Module):
    """Deep supervision: is the pooled Z already predictive of the candidate?"""

    def __init__(self, d_q):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d_q, 2 * d_q), nn.GELU(), nn.Linear(2 * d_q, 1))

    def forward(self, Z, c):
        return self.net(torch.cat([Z.mean(dim=1), c], dim=-1)).squeeze(-1)


def llm_target_rms(embedding_weight, d_llm):
    """mean row-norm of the LLM embedding table, expressed as an RMS per dim."""
    with torch.no_grad():
        E = embedding_weight.detach().float()
        return (E.norm(dim=-1).mean() / math.sqrt(d_llm)).item()


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def token_decorrelation_loss(Z):
    """`L_div`: different tokens must not be the same vector."""
    L = Z.shape[1]
    if L < 2:
        return Z.new_zeros(())
    Zn = F.normalize(Z.float(), dim=-1)
    S = Zn @ Zn.transpose(1, 2)                                    # (B, L, L)
    off = ~torch.eye(L, dtype=torch.bool, device=Z.device)
    return (S[:, off] ** 2).mean()


def mean_offdiag_cosine(Z):
    """Diagnostic: mean off-diagonal cosine between the L tokens."""
    L = Z.shape[1]
    if L < 2:
        return Z.new_zeros(())
    Zn = F.normalize(Z.float(), dim=-1)
    S = Zn @ Zn.transpose(1, 2)
    off = ~torch.eye(L, dtype=torch.bool, device=Z.device)
    return S[:, off].mean()


def attention_disagreement_loss(A, eps=1e-6):
    """`L_attn`: Bhattacharyya overlap of the per-token attention maps.

    ``eps`` inside the sqrt is not cosmetic: masked memory slots get exactly
    zero attention, and ``d/dx sqrt(x)`` is infinite at 0, so a plain
    ``A.sqrt()`` makes the whole backward pass NaN as soon as any slot is padded
    (i.e. on essentially every real batch).
    """
    L = A.shape[1]
    if L < 2:
        return A.new_zeros(())
    sq = (A.float().clamp_min(0) + eps).sqrt()
    BC = (sq.unsqueeze(2) * sq.unsqueeze(1)).sum(-1)               # (B, L, L)
    off = ~torch.eye(L, dtype=torch.bool, device=A.device)
    return BC[:, off].mean()


def variance_hinge_loss(Z, target_std=1.0, min_batch=8, eps=1e-6):
    """`L_var` (VICReg-style): a token must not be constant across users.

    Uses ``sqrt(var + eps)`` rather than ``std()`` for the same reason as above:
    the gradient of ``std`` is undefined when a token really is constant, which
    is exactly the collapse case this term exists to punish.
    """
    if Z.shape[0] < min_batch:
        return Z.new_zeros(())
    std = (Z.float().var(dim=0, unbiased=True) + eps).sqrt()       # (L, d_q)
    return F.relu(target_std - std).mean()


def within_user_rank_loss(scores, users, labels):
    """`L_rank`: pairwise BPR over pos/neg pairs of the *same* user in-batch.

    This is the direct lever on UAUC, which is exactly per-user ranking. Users
    without a pos/neg pair in the batch contribute nothing.
    """
    scores = scores.float()
    same_user = users.unsqueeze(1) == users.unsqueeze(0)
    pos = labels.reshape(-1) > 0.5
    pair = same_user & pos.unsqueeze(1) & (~pos).unsqueeze(0)
    n_pairs = int(pair.sum().item())
    if n_pairs == 0:
        return scores.new_zeros(()), 0
    diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    return F.softplus(-diff[pair]).mean(), n_pairs


def global_rank_loss(scores, labels):
    """`L_rank_global`: pairwise BPR over ALL pos/neg pairs in the batch.

    AUC is, by definition, the fraction of pos/neg pairs ordered correctly across
    the whole set -- and nothing else in this loss optimises it. ``L_bce`` touches
    global calibration only through absolute logit values, and
    ``within_user_rank_loss`` is invariant to a per-user shift of the scores, which
    is precisely the degree of freedom global AUC measures. Measured on ML-1M:
    turning the within-user term on moved UAUC +0.046 and AUC +0.014, i.e. almost
    all of it landed on the metric that term addresses.

    Same-user pairs are deliberately NOT masked out: AUC counts them too, so
    excluding them would optimise something slightly different from the metric. It
    barely matters either way -- on ML-1M valid there are ~27M cross-user pairs
    against ~59k within-user ones, so the two terms are near-disjoint in practice.

    Cost is one (n_pos, n_neg) matrix, at most 48x48 at the shipped batch size.
    """
    scores = scores.float()
    pos = labels.reshape(-1) > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return scores.new_zeros(()), 0
    diff = scores[pos].unsqueeze(1) - scores[~pos].unsqueeze(0)
    return F.softplus(-diff).mean(), n_pos * n_neg


def vocab_align_loss(z, embedding_weight, n_sample=8192, generator=None):
    """Optional `L_align`: push soft tokens toward directions the LLM has seen."""
    with torch.no_grad():
        V = embedding_weight.shape[0]
        idx = torch.randint(0, V, (min(n_sample, V),), device=z.device, generator=generator)
        Es = F.normalize(embedding_weight[idx].detach().float(), dim=-1)
    zn = F.normalize(z.float().reshape(-1, z.shape[-1]), dim=-1)
    return (1.0 - (zn @ Es.t()).max(dim=-1).values).mean()
