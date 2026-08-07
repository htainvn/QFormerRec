#!/usr/bin/env python
"""CPU smoke test for the Q-Former CIE.

Checks the parts that do not need Vicuna weights:
  1. shapes / masking / gradient flow through MemoryEncoder + Q-Former,
  2. FiLM zero-init really starts from the base queries,
  3. the anti-collapse losses behave (identical tokens -> large L_div, etc.),
  4. a row whose only valid slot is the user does not produce NaNs,
  5. the RMS matching puts the soft tokens on the LLM embedding scale,
  6. the <unk>-substitution splicing writes each soft block at the right
     position, in prompt order (the failure mode that silently scrambles
     <PrefTokens> / <UserID> / <TargetItemID>),
  7. the user-grouped sampler yields same-user pos/neg pairs.

Run:  python scripts/smoke_test.py [--memory_index path.pkl]
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qformerrec.datasets.samplers import UserGroupedBatchSampler  # noqa: E402
from qformerrec.models.qformer_cie import (  # noqa: E402
    HIST_PAD,
    N_SLOT_TYPES,
    PREF_TOKEN_FLAG,
    SLOT_HIST,
    SLOT_HIST_UNK,
    CFAuxHead,
    CandidateAwareQueryGenerator,
    CollabQFormer,
    LLMProjector,
    MemoryEncoder,
    attention_disagreement_loss,
    llm_target_rms,
    mean_offdiag_cosine,
    soft_slot_plan,
    split_title_list,
    token_decorrelation_loss,
    variance_hinge_loss,
    within_user_rank_loss,
)

OK, FAIL = "  ok  ", " FAIL "
_failures = []


def check(name, cond, extra=""):
    print(f"[{OK if cond else FAIL}] {name} {extra}")
    if not cond:
        _failures.append(name)


def fake_index(n_users=40, n_items=60, d1=32, k_i=10, k_n=8, k_g=3, k_c=3, n_genres=6, n_clu=5):
    rng = np.random.RandomState(0)
    # items 50..59 were never seen in training -> they must route to `unk_item`
    item_in_train = np.ones(n_items, dtype=bool)
    item_in_train[50:] = False
    hist = rng.randint(1, n_items, (n_users, k_i))
    hist[:, 5:] = -1                       # half the history padded
    hist[0, :] = -1                        # user 0: no history at all
    nb = rng.randint(1, n_users, (n_users, k_n))
    nb[0, :] = -1
    sim = rng.rand(n_users, k_n).astype(np.float32)
    sim[0, :] = 0
    genres = rng.randint(0, n_genres, (n_users, k_g))
    clu = rng.randint(0, n_clu, (n_users, k_c))
    has = np.ones(n_users, dtype=bool)
    has[0] = False                         # user 0 has no train interactions
    return {
        "hist_items": hist, "neighbors": nb, "neighbor_sim": sim, "genres": genres,
        "user_cluster": clu, "user_has_train": has, "item_in_train": item_in_train,
        "genre_proto_init": rng.randn(n_genres, d1).astype(np.float32),
        "cluster_proto_init": rng.randn(n_clu, d1).astype(np.float32),
        "meta": {"mf_embedding_size": d1},
    }


def build(mi, d1=32, d_q=16, L=4, n_users=40, n_items=60, d_llm=64, **kw):
    torch.manual_seed(0)
    mem_enc = MemoryEncoder(mi, d1=d1, d_q=d_q, **kw)
    qgen = CandidateAwareQueryGenerator(d1, d_q, L)
    qf = CollabQFormer(d_q, n_layers=2, n_heads=4, dropout=0.0)
    U = torch.nn.Embedding(n_users, d1)
    I = torch.nn.Embedding(n_items, d1)
    E = torch.randn(500, d_llm) * 0.02
    proj = LLMProjector(d_q, d_llm, llm_target_rms(E, d_llm))
    return mem_enc, qgen, qf, U, I, proj, E


def make_pit(B, width=10, n_items=60, seed=0, unk_frac=0.0):
    """Per-row point-in-time history: most-recent-first, HIST_PAD padded."""
    rng = np.random.RandomState(seed)
    out = np.full((B, width), HIST_PAD, dtype=np.int64)
    for b in range(B):
        n = rng.randint(0, width + 1)                 # row 0 often ends up empty
        hi = 50 if unk_frac > 0 else 50               # 50..59 are the untrained ids
        items = rng.randint(1, n_items if unk_frac > 0 else hi, n)
        out[b, :n] = items
    return torch.as_tensor(out)


def run(mem_enc, qgen, qf, U, I, proj, uid, iid, pit=None):
    user_emb = U(uid)
    if pit is None and mem_enc.history_source == "pit":
        pit = make_pit(uid.shape[0], width=max(mem_enc.k_hist, 1))
    mem, mask, tids, prior, stats = mem_enc(
        uid, user_emb, lambda x: I(x), lambda x: U(x), pit_hist=pit
    )
    q, c = qgen(I(iid))
    bias = qgen.slot_bias(tids) + torch.log(prior.clamp_min(1e-6)).unsqueeze(1)
    Z, A = qf(q, mem, mask, bias)
    return mem, mask, tids, Z, A, proj(Z), c


def test_core():
    print("\n=== 1-5: memory bank / Q-Former / losses ===")
    d1, d_q, L, d_llm = 32, 16, 4, 64
    mi = fake_index(d1=d1)
    mem_enc, qgen, qf, U, I, proj, E = build(mi, d1=d1, d_q=d_q, L=L, d_llm=d_llm)

    B = 32
    uid = torch.arange(B) % 40
    iid = torch.randint(1, 60, (B,))
    # row 0 gets an empty history, row 1 an all-untrained one
    pit = make_pit(B, width=10, seed=3, unk_frac=1.0)
    pit[0, :] = HIST_PAD
    pit[1, :] = torch.tensor([55, 56, 57, HIST_PAD, HIST_PAD, HIST_PAD,
                              HIST_PAD, HIST_PAD, HIST_PAD, HIST_PAD])
    mem, mask, tids, Z, A, z, c = run(mem_enc, qgen, qf, U, I, proj, uid, iid, pit=pit)

    N = 1 + 10 + 8 + 3 + 3
    check("slot budget = 25", mem_enc.n_slots == N, f"(got {mem_enc.n_slots})")
    check("mem shape (B,N,d_q)", tuple(mem.shape) == (B, N, d_q), str(tuple(mem.shape)))
    check("mask shape (B,N)", tuple(mask.shape) == (B, N))
    check("type_ids is per row (B,N)", tuple(tids.shape) == (B, N), str(tuple(tids.shape)))
    check("type_ids in [0,T)", int(tids.min()) == 0 and int(tids.max()) < N_SLOT_TYPES)
    check("non-history slot types are fixed",
          tids[:, 0].unique().tolist() == [0]
          and tids[:, 11:].unique().tolist() == [2, 3, 4])
    check("history slots use only SLOT_HIST / SLOT_HIST_UNK",
          set(tids[:, 1:11].unique().tolist()) <= {SLOT_HIST, SLOT_HIST_UNK})
    # untrained items must be flagged and routed to the learned unk_item vector
    check("untrained history items get SLOT_HIST_UNK",
          tids[1, 1:4].tolist() == [SLOT_HIST_UNK] * 3, str(tids[1, 1:11].tolist()))
    check("trained history items keep SLOT_HIST",
          all(t == SLOT_HIST for t, i in zip(tids[0 if False else 2, 1:11].tolist(),
                                             pit[2].tolist()) if 0 < i < 50))
    check("empty history row -> all history slots masked",
          int(mask[0, 1:11].sum()) == 0 and int(mask[0].sum()) > 0,
          f"(hist={int(mask[0, 1:11].sum())}, total={int(mask[0].sum())})")
    check("history mask follows HIST_PAD exactly",
          bool(torch.equal(mask[:, 1:11], pit > HIST_PAD)))
    check("mem_stats reports the unk rate",
          0.0 < float(mem_enc(uid, U(uid), lambda x: I(x), lambda x: U(x),
                              pit_hist=pit)[4]["hist_unk_rate"]) <= 1.0)
    check("Z shape (B,L,d_q)", tuple(Z.shape) == (B, L, d_q))
    check("A shape (B,L,N)", tuple(A.shape) == (B, L, N))
    check("z shape (B,L,d_llm)", tuple(z.shape) == (B, L, d_llm))
    check("no NaN/Inf anywhere", bool(torch.isfinite(Z).all() and torch.isfinite(A).all()
                                     and torch.isfinite(z).all()))

    # 4. the all-but-user-masked row (user 0) must be finite and its attention
    #    must sit entirely on the single valid slot
    row0 = (uid == 0).nonzero()[0, 0].item()
    check("user 0 has exactly 1 valid slot", int(mask[row0].sum()) == 1, str(int(mask[row0].sum())))
    check("user 0 attention is finite", bool(torch.isfinite(A[row0]).all()))
    check("user 0 attention mass on slot 0 == 1",
          bool(torch.allclose(A[row0, :, 0], torch.ones(L), atol=1e-5)),
          f"{A[row0, :, 0].tolist()}")
    check("attention over masked slots is 0",
          float(A[~mask.unsqueeze(1).expand(-1, L, -1)].abs().max()) == 0.0)
    check("attention rows sum to 1", bool(torch.allclose(A.sum(-1), torch.ones(B, L), atol=1e-4)))

    # padded slots must not leak their (clamped) embedding into `mem` pre-LN --
    # check the projection input was zeroed by re-running with a poisoned table
    with torch.no_grad():
        I2 = torch.nn.Embedding(60, d1)
        I2.weight.copy_(I.weight)
        I2.weight[0] = 1e4                      # index 0 = what pads clamp to
        mem_p = mem_enc(uid, U(uid), lambda x: I2(x), lambda x: U(x), pit_hist=pit)[0]
    check("padded slots are zeroed before projection",
          bool(torch.isfinite(mem_p).all() and float(mem_p.abs().max()) < 100.0),
          f"max|mem|={float(mem_p.abs().max()):.2f}")

    # 2. FiLM zero-init => queries are exactly LN(Q0), candidate-independent
    q_a, _ = qgen(I(torch.zeros(B, dtype=torch.long)))
    q_b, _ = qgen(I(torch.full((B,), 7, dtype=torch.long)))
    check("FiLM zero-init: queries independent of candidate at init",
          bool(torch.allclose(q_a, q_b, atol=1e-6)))
    check("type_bias zero-init -> uniform softmax",
          bool(torch.allclose(qgen.type_bias.softmax(-1),
                              torch.full((L, N_SLOT_TYPES), 1.0 / N_SLOT_TYPES), atol=1e-6)))

    # after a step on the FiLM weights the queries must differ per candidate
    with torch.no_grad():
        qgen.film.weight.normal_(0, 0.02)
    q_a, _ = qgen(I(torch.zeros(B, dtype=torch.long)))
    q_b, _ = qgen(I(torch.full((B,), 7, dtype=torch.long)))
    check("FiLM active: queries depend on the candidate", not torch.allclose(q_a, q_b, atol=1e-4))

    # 5. RMS matching
    emb_norm = E.norm(dim=-1).mean().item()
    z_norm = z.norm(dim=-1).mean().item()
    check("soft-token norm within 2x of the LLM embedding norm",
          0.5 <= z_norm / emb_norm <= 2.0, f"(z={z_norm:.4f} vs E={emb_norm:.4f})")

    # 3. losses
    Zi = torch.randn(B, 1, d_q).expand(-1, L, -1).contiguous()      # collapsed
    Zd = F.normalize(torch.randn(B, L, d_q), dim=-1) * 3
    check("L_div: collapsed >> diverse",
          float(token_decorrelation_loss(Zi)) > 0.9 > float(token_decorrelation_loss(Zd)),
          f"({float(token_decorrelation_loss(Zi)):.3f} vs {float(token_decorrelation_loss(Zd)):.3f})")
    check("token cosine: collapsed ~ 1.0",
          abs(float(mean_offdiag_cosine(Zi)) - 1.0) < 1e-4)
    Ai = torch.zeros(B, L, N); Ai[:, :, 3] = 1.0                    # all read slot 3
    Ad = torch.zeros(B, L, N)
    for l in range(L):
        Ad[:, l, l] = 1.0                                          # disjoint slots
    # the eps inside sqrt shifts the endpoints by ~N*eps / ~2*sqrt(eps)
    check("L_attn: same-slot ~1, disjoint ~0",
          abs(float(attention_disagreement_loss(Ai)) - 1.0) < 1e-3
          and float(attention_disagreement_loss(Ad)) < 1e-2,
          f"({float(attention_disagreement_loss(Ai)):.5f} vs "
          f"{float(attention_disagreement_loss(Ad)):.5f})")
    # and the NaN case that motivated the eps: masked slots have A == 0 exactly
    Am = torch.zeros(B, L, N, requires_grad=False)
    Am[:, :, :4] = 0.25
    Am = Am.detach().requires_grad_(True)
    attention_disagreement_loss(Am).backward()
    check("L_attn: gradient stays finite when slots have exactly 0 attention",
          bool(torch.isfinite(Am.grad).all()))
    Zc = torch.ones(B, L, d_q)                                     # constant across users
    Zc_g = Zc.clone().requires_grad_(True)
    variance_hinge_loss(Zc_g).backward()
    check("L_var: gradient stays finite for a perfectly collapsed token",
          bool(torch.isfinite(Zc_g.grad).all()))
    check("L_var: constant tokens -> 1.0, unit-std -> ~0",
          abs(float(variance_hinge_loss(Zc)) - 1.0) < 1e-2
          and float(variance_hinge_loss(torch.randn(64, L, d_q))) < 0.15,
          f"({float(variance_hinge_loss(Zc)):.3f}, "
          f"{float(variance_hinge_loss(torch.randn(64, L, d_q))):.3f})")

    # L_rank: two users, one pos + one neg each -> 4 pairs? no: 1 pair per user
    users = torch.tensor([1, 1, 2, 2])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    good = torch.tensor([5.0, -5.0, 5.0, -5.0])
    bad = torch.tensor([-5.0, 5.0, -5.0, 5.0])
    lg, ng = within_user_rank_loss(good, users, labels)
    lb, nb_ = within_user_rank_loss(bad, users, labels)
    check("L_rank: pair count = 2", ng == 2 and nb_ == 2, f"({ng}, {nb_})")
    check("L_rank: correct order << wrong order", float(lg) < 0.01 < 9.0 < float(lb),
          f"({float(lg):.4f} vs {float(lb):.4f})")
    lz, nz = within_user_rank_loss(good, torch.tensor([1, 2, 3, 4]), labels)
    check("L_rank: no same-user pair -> 0 loss, 0 pairs", nz == 0 and float(lz) == 0.0)

    # gradients reach MF, prototypes, FiLM and type_bias
    cf = CFAuxHead(d_q)
    mem_enc.zero_grad(); qgen.zero_grad(); qf.zero_grad(); U.zero_grad(); I.zero_grad()
    mem, mask, tids, Z, A, z, c = run(mem_enc, qgen, qf, U, I, proj, uid, iid)
    loss = (z.sum() + cf(Z, c).sum() + token_decorrelation_loss(Z)
            + attention_disagreement_loss(A))
    loss.backward()
    for name, p in [("MF user table", U.weight), ("MF item table", I.weight),
                    ("genre_proto", mem_enc.genre_proto.weight),
                    ("cluster_proto", mem_enc.cluster_proto.weight),
                    ("Q0", qgen.Q0), ("film", qgen.film.weight),
                    ("type_bias", qgen.type_bias), ("proj.scale", proj.scale)]:
        check(f"gradient reaches {name}",
              p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0)

    # ablations: dropping a slot type shrinks N and never breaks
    for kw, expect in [
        (dict(k_hist=0, k_neighbor=0, k_genre=0, k_cluster=0), 1),      # -memory
        (dict(k_hist=0), 15), (dict(k_neighbor=0), 17),
        (dict(k_genre=0), 22), (dict(k_cluster=0), 22),
    ]:
        me, qg, q_, U_, I_, pr, _ = build(mi, d1=d1, d_q=d_q, L=L, d_llm=d_llm, **kw)
        _, msk, _, Z_, A_, z_, _ = run(me, qg, q_, U_, I_, pr, uid, iid)
        check(f"ablation {kw}: N={expect}, finite",
              me.n_slots == expect and bool(torch.isfinite(z_).all()),
              f"(got N={me.n_slots})")

    for L_ in (1, 2, 8):
        me, qg, q_, U_, I_, pr, _ = build(mi, d1=d1, d_q=d_q, L=L_, d_llm=d_llm)
        _, _, _, Z_, A_, z_, _ = run(me, qg, q_, U_, I_, pr, uid, iid)
        check(f"L={L_} runs and losses are finite",
              tuple(z_.shape) == (B, L_, d_llm)
              and torch.isfinite(token_decorrelation_loss(Z_))
              and torch.isfinite(attention_disagreement_loss(A_)))


def test_splicing():
    """The <unk>-substitution splicing: right count, right order, right rows."""
    print("\n=== 6: prompt splicing index math ===")
    prompt = ("features <PrefTokens>, identity <UserID>, title <TargetItemTitle> "
              "with feature <TargetItemID>?")
    plan = soft_slot_plan(prompt, n_query=4, proj_token_num=1)
    check("plan is in prompt order", [f for f, _ in plan] ==
          [PREF_TOKEN_FLAG, "<UserID>", "<TargetItemID>"], str(plan))
    check("plan token counts", [n for _, n in plan] == [4, 1, 1])
    # a prompt that puts the IDs first must reorder
    p2 = "<TargetItemID> then <UserID> then <PrefTokens>"
    check("plan follows a reordered prompt",
          [f for f, _ in soft_slot_plan(p2, 4, 1)] ==
          ["<TargetItemID>", "<UserID>", PREF_TOKEN_FLAG])
    check("plan honours proj_token_num=2",
          [n for _, n in soft_slot_plan(prompt, 4, 2)] == [4, 2, 2])
    check("plan is empty for a title-only prompt", soft_slot_plan("no soft here", 4, 1) == [])

    # emulate the tokenizer: unk id 0, per-row lengths differ (left padding)
    B, L, d = 5, 4, 8
    n_soft = L + 2
    rng = np.random.RandomState(1)
    ids = torch.full((B, 12), 9, dtype=torch.long)
    for b in range(B):
        cols = sorted(rng.choice(np.arange(2 + b % 3, 12), n_soft, replace=False))
        for cidx in cols:
            ids[b, cidx] = 0
    embeds = torch.zeros(B, 12, d)
    pref = torch.arange(B * L * d, dtype=torch.float).reshape(B, L, d)
    user = torch.full((B, 1, d), -1.0)
    item = torch.full((B, 1, d), -2.0)
    merged = torch.cat([pref, user, item], dim=1)
    idx = torch.nonzero(ids == 0)
    check("unk count == B * n_soft", idx.shape[0] == B * n_soft)
    embeds[idx[:, 0], idx[:, 1]] = merged.reshape(-1, d)
    # every row's unk positions, read left to right, must equal [pref..., user, item]
    all_ok = True
    for b in range(B):
        cols = (ids[b] == 0).nonzero().flatten()
        got = embeds[b, cols]
        all_ok &= bool(torch.equal(got, merged[b]))
    check("soft blocks land in prompt order on every row", all_ok)

    check("title splitter keeps commas inside titles",
          split_title_list('"Shawshank Redemption, The (1994)", "Sting, The (1973)"')
          == ['"Shawshank Redemption, The (1994)', 'Sting, The (1973)"'])
    check("title splitter on empty", split_title_list("unkow") == []
          and split_title_list("") == [])


def test_sampler():
    print("\n=== 7: user-grouped batch sampler ===")
    rng = np.random.RandomState(0)
    n = 4000
    uids = rng.randint(1, 120, n)
    labels = rng.randint(0, 2, n)
    s = UserGroupedBatchSampler(uids, labels, n_users_per_batch=8, n_per_user=6, seed=1)
    batches = [b for _, b in zip(range(30), iter(s))]
    check("batch size is exactly U*m = 48", all(len(b) == 48 for b in batches))
    check("indices are in range", all(0 <= i < n for b in batches for i in b))
    pairs = []
    for b in batches:
        u, l = uids[b], labels[b]
        same = u[:, None] == u[None, :]
        p = same & (l[:, None] == 1) & (l[None, :] == 0)
        pairs.append(int(p.sum()))
    check("every batch has same-user pos/neg pairs (L_rank has signal)",
          min(pairs) > 0, f"min={min(pairs)} mean={np.mean(pairs):.1f}")
    # a plain random sampler for comparison
    rnd = []
    for _ in range(30):
        b = rng.choice(n, 48, replace=False)
        u, l = uids[b], labels[b]
        same = u[:, None] == u[None, :]
        rnd.append(int((same & (l[:, None] == 1) & (l[None, :] == 0)).sum()))
    print(f"        (random sampler for reference: mean pairs={np.mean(rnd):.1f})")
    check("grouped sampler beats random on pair count", np.mean(pairs) > 3 * np.mean(rnd))
    check("re-iterating reshuffles", batches[0] != [b for _, b in zip(range(1), iter(s))][0])


def test_real_index(path):
    print(f"\n=== 8: real memory index {path} ===")
    with open(path, "rb") as f:
        mi = pickle.load(f)
    meta = mi["meta"]
    print("        meta:", {k: meta[k] for k in
                            ["dataset", "n_users", "n_items", "mf_embedding_size", "n_slots",
                             "genre_source", "train_only"] if k in meta})
    d1 = meta["mf_embedding_size"]
    n_users = mi["hist_items"].shape[0]
    uid = torch.arange(min(64, n_users))
    U = torch.nn.Embedding(n_users, d1)
    I = torch.nn.Embedding(meta["n_items"], d1)
    # both history paths must work off the same artifact
    for src, k_hist in [("pit", 50), ("train_only", 10)]:
        me = MemoryEncoder(mi, d1=d1, d_q=16, k_hist=k_hist, history_source=src)
        pit = make_pit(uid.shape[0], width=k_hist, n_items=meta["n_items"], seed=7) \
            if src == "pit" else None
        mem, mask, tids, prior, stats = me(
            uid, U(uid), lambda x: I(x), lambda x: U(x), pit_hist=pit
        )
        check(f"real index [{src}]: mem finite and N={me.n_slots}",
              bool(torch.isfinite(mem).all()), f"N={me.n_slots} k_hist={me.k_hist}")
        check(f"real index [{src}]: every row has >=1 valid slot",
              bool((mask.sum(1) > 0).all()))
        check(f"real index [{src}]: type_ids per row within [0,T)",
              tuple(tids.shape) == (uid.shape[0], me.n_slots)
              and int(tids.max()) < N_SLOT_TYPES)
    check("real index: item_in_train present and plausible",
          "item_in_train" in mi and 0.5 < mi["item_in_train"].mean() <= 1.0,
          f"{mi['item_in_train'].mean():.2%} of items seen in train")
    check("real index: ids within table bounds",
          int(mi["hist_items"].max()) < meta["n_items"]
          and int(mi["neighbors"].max()) < meta["n_users"]
          and int(mi["genres"].max()) < meta["n_genres"]
          and int(mi["user_cluster"].max()) < meta["n_user_clusters"])
    check("real index: no self-neighbours",
          not any(u in set(mi["neighbors"][u][mi["neighbors"][u] >= 0])
                  for u in range(min(500, n_users))))
    check("real index: neighbour sims in [0,1]",
          float(mi["neighbor_sim"].min()) >= 0 and float(mi["neighbor_sim"].max()) <= 1.0001)
    filled = (mi["hist_items"] >= 0).sum(1)
    print(f"        history slots filled: mean={filled.mean():.2f} "
          f"users with none={(filled == 0).sum()}/{n_users}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory_index", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    test_core()
    test_splicing()
    test_sampler()
    if args.memory_index:
        test_real_index(args.memory_index)

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED ({len(_failures)}):")
        for f in _failures:
            print("  -", f)
        sys.exit(1)
    print("all smoke tests passed")


if __name__ == "__main__":
    main()
