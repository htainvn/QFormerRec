#!/usr/bin/env python
"""End-to-end integration test with a TINY randomly-initialised LLaMA.

Exercises the parts the CPU smoke test cannot: registry wiring, the real
``LlamaTokenizer`` splicing path, LoRA freezing, ``forward_v2`` / eval, the
optimizer param grouping, the dataset/builder and the task's evaluation loop --
without needing the 13 GB Vicuna weights.

Prereqs: a CoLLM checkout on ``COLLM_ROOT``, a ``memory_index`` pkl, an MF
checkpoint, and a CoLLM-format data dir. The tiny LLaMA is created on the fly.

    python scripts/integration_test.py \
        --data_dir ... --mf_ckpt ... --memory_index ... --work_dir /tmp/itest
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from qformerrec.compat import check_environment, install_import_shims  # noqa: E402

install_import_shims()

_failures = []


def check(name, cond, extra=""):
    print(f"[{'  ok  ' if cond else ' FAIL '}] {name} {extra}")
    if not cond:
        _failures.append(name)


def make_tiny_llama(path, tokenizer_repo="hf-internal-testing/llama-tokenizer"):
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

    if os.path.exists(os.path.join(path, "config.json")):
        return path
    os.makedirs(path, exist_ok=True)
    cfg = LlamaConfig(
        vocab_size=32000, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=2048,
    )
    LlamaForCausalLM(cfg).save_pretrained(path)
    LlamaTokenizer.from_pretrained(tokenizer_repo, use_fast=False).save_pretrained(path)
    print(f"built tiny LLaMA at {path}")
    return path


def build_cfg(args, llama_path, stage):
    from omegaconf import OmegaConf

    freeze_lora = stage == 2
    return OmegaConf.create({
        "model": {
            "arch": "mini_gpt4rec_qformer", "model_type": "pretrain_vicuna",
            "freeze_rec": stage == 2, "freeze_proj": False, "freeze_lora": freeze_lora,
            "max_txt_len": 1024, "proj_token_num": 1, "proj_drop": 0, "proj_mid_times": 2,
            "end_sym": "###", "prompt_path": os.path.join(ROOT, "prompts/qformer_movie.txt"),
            "prompt_template": "{}", "llama_model": llama_path,
            "ans_type": "v2", "rec_model": "MF", "n_titles_kept": 0, "diag_log_freq": 0,
            "lora_config": {"use_lora": True, "r": 8, "alpha": 16,
                            "target_modules": ["q_proj", "v_proj"], "dropout": 0.05},
            "rec_config": {"user_num": args.user_num, "item_num": args.item_num,
                           "embedding_size": 256, "pretrained_path": args.mf_ckpt},
            "qformer": {"d_q": 32, "n_query": 4, "n_layers": 2, "n_heads": 4, "dropout": 0.1,
                        "use_slot_prior": True, "use_candidate": True, "match_llm_norm": True,
                        "memory": {"history_source": "pit", "k_hist": 50, "k_neighbor": 8,
                                   "k_genre": 3, "k_cluster": 3,
                                   "memory_index_path": args.memory_index}},
            "loss": {"lambda_rank": 0.5, "lambda_cf": 0.2, "lambda_div": 0.1,
                     "lambda_attn": 0.05, "lambda_var": 0.05, "lambda_align": 0.01,
                     "rank_score": "yes_minus_no"},
        },
        "run": {"task": "rec_qformer", "runner": "rec_runner_qformer",
                "lr_sched": "linear_warmup_cosine_lr_scaled", "init_lr": 1e-3, "min_lr": 8e-5,
                "warmup_lr": 1e-5, "mode": "v2", "select_metric": "uauc", "grad_clip": 1.0,
                "lr_scale": {"qformer": 1.0, "proto": 1.0, "rec": 0.1, "lora": 0.1},
                "user_grouped_sampler": True, "users_per_batch": 4, "rows_per_user": 4,
                "strict_batch": False, "weight_decay": 1e-3, "max_epoch": 1,
                "iters_per_epoch": 3, "batch_size_train": 16, "batch_size_eval": 16,
                "num_workers": 0, "warmup_steps": 2, "seed": 42,
                "output_dir": os.path.join(args.work_dir, f"out_stage{stage}"),
                "amp": False, "resume_ckpt_path": None, "evaluate": False,
                "train_splits": ["train"], "valid_splits": ["valid"],
                "test_splits": ["test"], "device": "cpu", "world_size": 1,
                "dist_url": "env://", "distributed": False},
        "datasets": {"movie_ood_qf": {"path": args.data_dir, "data_type": "default",
                                      "pit_hist_width": 50,
                                      "build_info": {"storage": args.data_dir}}},
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--mf_ckpt", required=True)
    ap.add_argument("--memory_index", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--collm_root", default=os.environ.get("COLLM_ROOT"))
    args = ap.parse_args()
    assert args.collm_root, "set COLLM_ROOT or pass --collm_root"
    sys.path.insert(0, os.path.abspath(args.collm_root))
    os.makedirs(args.work_dir, exist_ok=True)

    import pandas as pd

    ids = [pd.read_pickle(os.path.join(args.data_dir, f"{s}_ood2.pkl"))[["uid", "iid"]]
           for s in ["train", "valid", "test"]]
    args.user_num = int(max(d.uid.max() for d in ids)) + 1
    args.item_num = int(max(d.iid.max() for d in ids)) + 1
    del ids

    from minigpt4.common.registry import registry  # noqa: F401
    from minigpt4.datasets.data_utils import prepare_sample

    import qformerrec.datasets.rec_datasets_qformer as dsmod  # noqa: F401
    import qformerrec.models.minigpt4rec_qformer as mmod  # noqa: F401
    import qformerrec.runners.runner_qformer as rmod  # noqa: F401
    import qformerrec.tasks.rec_qformer_task as tmod  # noqa: F401

    print("\n=== environment ===")
    check_environment()
    check("environment has no known-bad version combination", not check_environment())

    print("\n=== registry ===")
    for kind, name in [("model", "mini_gpt4rec_qformer"), ("task", "rec_qformer"),
                       ("runner", "rec_runner_qformer"), ("builder", "movie_ood_qf"),
                       ("builder", "amazon_ood_qf"),
                       ("lr_scheduler", "linear_warmup_cosine_lr_scaled")]:
        getter = getattr(registry, f"get_{kind}_class")
        check(f"{kind} '{name}' registered", getter(name) is not None)

    llama_path = make_tiny_llama(os.path.join(args.work_dir, "tinyllama"))

    print("\n=== dataset / builder ===")
    ds = dsmod.RecOODDataset(ann_paths=[os.path.join(args.data_dir, "test")])
    ds_warm = dsmod.RecOODDataset(ann_paths=[os.path.join(args.data_dir, "test=warm")])
    ds_cold = dsmod.RecOODDataset(ann_paths=[os.path.join(args.data_dir, "test=cold")])
    check("warm + cold partition the test split",
          len(ds_warm) + len(ds_cold) == len(ds),
          f"({len(ds_warm)} + {len(ds_cold)} == {len(ds)})")
    check("warm split is non-trivial", 0 < len(ds_warm) < len(ds))
    s0 = ds[0]
    check("sample has the fields the model needs",
          all(k in s0 for k in ["UserID", "TargetItemID", "TargetItemTitle",
                                "InteractedItemTitles", "label"]), str(sorted(s0)))
    check("target title is quoted", s0["TargetItemTitle"].startswith('"'))

    print("\n=== point-in-time history field ===")
    from qformerrec.models.qformer_cie import HIST_PAD
    check("dataset emits PitHistItems", "PitHistItems" in s0, str(sorted(s0)))
    check("PitHistItems width == pit_hist_width",
          s0["PitHistItems"].shape == (ds.pit_hist_width,), str(s0["PitHistItems"].shape))
    raw = ds.annotation["InteractedItemIDs"].iloc[0]
    expect = [int(x) for x in raw if int(x) != 0][-ds.pit_hist_width:][::-1]
    got = [int(x) for x in s0["PitHistItems"] if int(x) != HIST_PAD]
    check("PitHistItems is the row's own history, most-recent-first",
          got == expect, f"got[:4]={got[:4]} expect[:4]={expect[:4]}")
    check("index-0 padding is dropped", 0 not in got)
    check("padding is on the right (recency rank == column index)",
          all(x == HIST_PAD for x in s0["PitHistItems"][len(got):]))
    # the whole point of the change: on the TEST split the history must be
    # dominated by valid/test-period items, not train ones
    import pandas as _pd
    _tr = _pd.read_pickle(os.path.join(args.data_dir, "train_ood2.pkl"))
    _va = _pd.read_pickle(os.path.join(args.data_dir, "valid_ood2.pkl"))
    tr_pos = set(zip(_tr[_tr.label == 1].uid, _tr[_tr.label == 1].iid))
    va_pos = set(zip(_va[_va.label == 1].uid, _va[_va.label == 1].iid))
    prov = {"train": 0, "valid_or_later": 0}
    for r in range(0, len(ds), max(1, len(ds) // 800)):
        smp = ds[r]
        for i in smp["PitHistItems"][:10]:
            i = int(i)
            if i == HIST_PAD:
                continue
            u = int(smp["UserID"])
            prov["train" if (u, i) in tr_pos else "valid_or_later"] += 1
    tot = sum(prov.values())
    frac_train = prov["train"] / max(tot, 1)
    check("test-split history is NOT dominated by train items (pit took effect)",
          frac_train < 0.30,
          f"train={frac_train:.1%} valid_or_later={1 - frac_train:.1%} "
          f"(expect ~9% train for ML-1M test at k=10)")
    del _tr, _va

    # ---------------------------------------------------------------- stage 2
    for stage in (2, 3):
        print(f"\n=== stage {stage}: build / forward / backward / eval ===")
        cfg_full = build_cfg(args, llama_path, stage)
        if stage == 3:
            cfg_full.model.ckpt = os.path.join(args.work_dir, "stage2_ckpt.pth")
        model = mmod.MiniGPT4RecQFormer.from_config(cfg_full.model)
        model.set_mode("v2")
        model = model.float()

        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        has_lora = any("lora_" in n for n in trainable)
        check(f"stage {stage}: LoRA {'frozen' if stage == 2 else 'trainable'}",
              has_lora == (stage == 3), f"(lora trainable={has_lora})")
        check(f"stage {stage}: MF {'frozen' if stage == 2 else 'trainable'}",
              any(n.startswith("rec_encoder") for n in trainable) == (stage == 3))
        for comp in ["memory_encoder", "query_gen", "qformer", "pref_proj", "cf_head",
                     "llama_proj"]:
            check(f"stage {stage}: {comp} trainable",
                  any(n.startswith(comp) for n in trainable))
        # peft rewrites q_proj/v_proj *inside* llama_model, so LoRA tensors are
        # also reachable as `llama_model.*`; only the base weights must be frozen
        base_llm_trainable = [n for n in trainable
                              if n.startswith("llama_model") and "lora_" not in n]
        check(f"stage {stage}: base LLM weights frozen", not base_llm_trainable,
              f"{base_llm_trainable[:3]}")

        # a real batch through the real tokenizer
        from torch.utils.data import DataLoader

        from qformerrec.datasets.samplers import UserGroupedBatchSampler
        train_ds = dsmod.RecOODDataset(ann_paths=[os.path.join(args.data_dir, "train")])
        bs = UserGroupedBatchSampler(train_ds.annotation["UserID"].values,
                                     train_ds.annotation["label"].values,
                                     n_users_per_batch=4, n_per_user=4, seed=0)
        loader = DataLoader(train_ds, batch_sampler=bs, collate_fn=train_ds.collater)
        batch = prepare_sample(next(iter(loader)), cuda_enabled=False)
        check(f"stage {stage}: batch is 16 rows", batch["UserID"].shape[0] == 16)

        # splicing: exact <unk> accounting and the L + 2 soft tokens
        prompt = model.prompt_list[0]
        enc = model.encode_recdata_qformer(batch)
        embeds, atts, n_tok = model.recprompt_wrap_qformer(enc, batch, prompt)
        n_soft = model.n_query + 2 * model.proj_token_num
        check(f"stage {stage}: pref tokens are (B, L, d_llm)",
              tuple(enc["PrefTokens"].shape) == (16, 4, model.llama_model.config.hidden_size))
        check(f"stage {stage}: spliced embeds finite", bool(torch.isfinite(embeds).all()))
        check(f"stage {stage}: attention mask matches embeds",
              atts.shape[:2] == embeds.shape[:2])
        check(f"stage {stage}: mean prompt length is short (< 110 tokens)",
              float(n_tok) < 110, f"({float(n_tok):.1f} tokens, n_soft={n_soft})")
        # the soft embeddings must actually be present in the spliced sequence
        unk_id = model.llama_tokenizer.unk_token_id
        for glue in ("", "."):
            toks = model.llama_tokenizer(
                [prompt.replace("<PrefTokens>", glue.join(["<unk>"] * 4))
                       .replace("<UserID>", "<unk>").replace("<TargetItemID>", "<unk>")
                       .replace("<TargetItemTitle>", batch["TargetItemTitle"][0])],
                add_special_tokens=False)
            got = sum(1 for t in toks.input_ids[0] if t == unk_id)
            check(f"stage {stage}: <unk> count == L+2 with glue={glue!r}",
                  got == n_soft, f"(got {got}, want {n_soft})")
        # every row must carry the same number of soft slots, or the row-major
        # scatter silently misaligns
        counts = {int((r == unk_id).sum()) for r in
                  model.llama_tokenizer(
                      [prompt.replace("<PrefTokens>", "<unk>" * 4)
                             .replace("<UserID>", "<unk>").replace("<TargetItemID>", "<unk>")
                             .replace("<TargetItemTitle>", t)
                       for t in batch["TargetItemTitle"]],
                      return_tensors="pt", padding="longest",
                      add_special_tokens=False).input_ids}
        check(f"stage {stage}: soft-slot count is identical on every row",
              counts == {n_soft}, str(counts))

        # forward / backward
        out = model.forward_v2(batch)
        check(f"stage {stage}: forward loss is finite scalar",
              torch.isfinite(out["loss"]) and out["loss"].ndim == 0, f"loss={float(out['loss']):.4f}")
        out["loss"].backward()
        grads = {n: p for n, p in model.named_parameters() if p.requires_grad}
        no_grad = [n for n, p in grads.items() if p.grad is None]
        nan_grad = [n for n, p in grads.items() if p.grad is not None
                    and not torch.isfinite(p.grad).all()]
        check(f"stage {stage}: every trainable param got a gradient",
              not no_grad, f"missing={no_grad[:5]}")
        check(f"stage {stage}: no NaN/Inf gradients", not nan_grad, f"nan={nan_grad[:5]}")
        reached = sum(1 for p in grads.values() if p.grad is not None and float(p.grad.abs().sum()) > 0)
        check(f"stage {stage}: gradients are non-zero", reached > 0.5 * len(grads),
              f"({reached}/{len(grads)})")

        # eval path
        model.eval()
        ev = model.generate_for_samples(batch)
        check(f"stage {stage}: eval returns logits + loss",
              "logits" in ev and "loss" in ev and ev["logits"].shape[0] == 16)
        check(f"stage {stage}: eval extras present",
              "n_prompt_tokens" in ev and "token_cosine" in ev,
              f"cos={float(ev['token_cosine']):.4f}")

        # a full optimizer step through the runner's grouping
        if stage == 2:
            print(f"\n=== stage {stage}: optimizer grouping + 2 real steps ===")
            import minigpt4.tasks as tasks
            from minigpt4.common.config import Config

            # CPU-only shim: CoLLM's PrefetchLoader constructs a
            # torch.cuda.Stream and calls .cuda() on every batch, neither of
            # which exists without a GPU. Identity-wrap it for this test; on the
            # real A100 run the genuine PrefetchLoader is used.
            if not torch.cuda.is_available():
                class CPULoader:
                    """PrefetchLoader's interface without the CUDA stream."""

                    def __init__(self, loader):
                        self.loader = loader

                    def __iter__(self):
                        return iter(self.loader)

                    def __len__(self):
                        return len(self.loader)

                    def __next__(self):     # MultiIterLoader asserts this exists
                        pass

                    def __getattr__(self, name):
                        return self.loader.__getattribute__(name)

                rmod.PrefetchLoader = CPULoader

            class _A:  # minimal stand-in for argparse output
                options = None
                cfg_path = os.path.join(args.work_dir, "itest_cfg.yaml")

            from omegaconf import OmegaConf
            OmegaConf.save(cfg_full, _A.cfg_path)
            cfg = Config(_A())
            task = tasks.setup_task(cfg)
            datasets = task.build_datasets(cfg)
            model2 = mmod.MiniGPT4RecQFormer.from_config(cfg.model_cfg).float()
            runner = registry.get_runner_class("rec_runner_qformer")(
                cfg=cfg, job_id="itest", task=task, model=model2, datasets=datasets
            )
            opt = runner.optimizer
            by_group = {}
            for g in opt.param_groups:
                by_group.setdefault(g["group"], []).append(g["lr_scale"])
            check("optimizer has qformer + proj + proto groups",
                  {"qformer", "proj", "proto"} <= set(by_group), str(sorted(by_group)))
            check("llama_proj is its own lr group (so it can be scaled down)",
                  "proj" in by_group and all(s == 1.0 for s in by_group["proj"]),
                  f"proj scales={by_group.get('proj')}")
            check("proto group carries lr_scale 1.0", all(s == 1.0 for s in by_group["proto"]))
            check("stage 2 has no lora group (frozen)", "lora" not in by_group)
            check("stage 2 has no rec group (frozen)", "rec" not in by_group)

            sched = runner.lr_scheduler
            sched.step(cur_epoch=0, cur_step=2)
            lrs = {g["group"]: g["lr"] for g in opt.param_groups}
            check("scheduler applies lr_scale per group",
                  abs(lrs["qformer"] / max(lrs.get("proto", lrs["qformer"]), 1e-12) - 1.0) < 1e-9)

            runner.set_model_mode("v2")
            before = model2.query_gen.Q0.detach().clone()
            stats = runner.train_epoch(0)
            check("training epoch ran", "loss" in stats, str(stats))
            check("Q0 actually moved", not torch.allclose(before, model2.query_gen.Q0.detach()))
            check("no NaN parameters after the step",
                  all(torch.isfinite(p).all() for p in model2.parameters()))

            # evaluation through the task (the path that used to crash single-GPU)
            print("\n=== task evaluation (single process) ===")
            val = runner.eval_epoch(split_name="valid", cur_epoch=0, skip_reload=True)
            for k in ["auc", "uauc", "agg_metrics", "mean_prompt_tokens", "ms_per_sample",
                      "token_cosine_offdiag", "uauc_users_scored", "uauc_users_skipped"]:
                check(f"eval result has '{k}'", k in val, f"={val.get(k)}")
            check("agg_metrics == uauc (UAUC-based selection)",
                  val["agg_metrics"] == val["uauc"])
            check("auc/uauc are finite and in (0, 1)",
                  np.isfinite(val["auc"]) and np.isfinite(val["uauc"])
                  and 0 < val["auc"] < 1 and 0 < val["uauc"] < 1,
                  f"auc={val['auc']:.4f} uauc={val['uauc']:.4f}")
            # the regression this guards: CoLLM's uAUC_me returns nan on modern
            # sklearn because single-class users are no longer skipped
            from qformerrec.tasks.rec_qformer_task import uauc_score
            from minigpt4.tasks.rec_base_task import uAUC_me
            import warnings
            rng = np.random.RandomState(0)
            uu = np.array([1, 1, 2, 2, 3, 3, 4])       # user 2 = all-positive, user 4 = 1 row
            yy = np.array([1, 0, 1, 1, 0, 1, 1])
            ss = rng.randn(7)
            ours, st = uauc_score(uu, ss, yy)
            check("uauc_score skips single-class and single-row users",
                  st["n_scored"] == 2 and st["n_single_class"] == 1
                  and st["n_single_row"] == 1 and np.isfinite(ours), str(st))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                legacy = uAUC_me(uu, ss, yy)[0]
            print(f"        (CoLLM uAUC_me on this input: {legacy}; ours: {ours:.4f})")

            # diagnostics + checkpoint for stage 3
            # train_epoch already flushed (and reset) the diagnostics, so do one
            # more forward to repopulate them before checking the summary
            model2.train()
            model2.forward_v2(batch)["loss"].backward()
            summ = model2.log_qformer_diagnostics(output_dir=str(runner.output_dir), tag="itest")
            check("diagnostics summary produced", "token_cosine_offdiag" in summ
                  and "pref_token_norm" in summ and "hist_unk_rate" in summ,
                  str({k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in summ.items()}))
            check("diagnostics file written",
                  os.path.exists(os.path.join(runner.output_dir, "qformer_diagnostics.jsonl")))
            sd = {n: p for n, p in model2.state_dict().items()
                  if not n.startswith(("llama_model.", "llama_model_lora."))}
            torch.save({"model": sd}, os.path.join(args.work_dir, "stage2_ckpt.pth"))

    # ------------------------------------------------- checkpoint contents/size
    # CoLLM's save filter is defeated by peft's module aliasing and writes the whole
    # frozen backbone: a real stage-1 "LoRA" checkpoint came to 13.5 GB, 0.2% LoRA.
    print("\n=== checkpoint contains only trainable tensors ===")
    ck_dir = os.path.join(args.work_dir, "out_ckpt")
    os.makedirs(ck_dir, exist_ok=True)
    ck_model = mmod.MiniGPT4RecQFormer.from_config(build_cfg(args, llama_path, 2).model).float()
    runner_cls = registry.get_runner_class("rec_runner_qformer")

    class _Shim:                       # only what _save_checkpoint touches
        output_dir = ck_dir
        scaler = None
        model = ck_model
        optimizer = torch.optim.AdamW(
            [p for p in ck_model.parameters() if p.requires_grad], lr=1e-3)
        config = type("C", (), {"to_dict": staticmethod(lambda: {})})()

        @staticmethod
        def unwrap_dist_model(m):
            return m

    runner_cls._save_checkpoint(_Shim(), 0, is_best=True)
    ck_path = os.path.join(ck_dir, "checkpoint_best.pth")
    saved_ck = torch.load(ck_path, map_location="cpu", weights_only=False)["model"]
    leaked = [k for k in saved_ck if k.startswith("llama_model") and "lora_" not in k]
    check("no frozen LLM weights in the checkpoint", not leaked,
          f"{len(leaked)} leaked, e.g. {leaked[:1]}")
    n_ck = sum(v.numel() for v in saved_ck.values())
    n_tr = sum(p.numel() for p in ck_model.parameters() if p.requires_grad)
    check("checkpoint params ~= trainable params (aliases may duplicate)",
          n_tr <= n_ck <= 2 * n_tr, f"ckpt={n_ck:,} trainable={n_tr:,}")
    check("checkpoint file is small", os.path.getsize(ck_path) / 1e6 < 50,
          f"{os.path.getsize(ck_path)/1e6:.1f} MB")
    # the same filter applied CoLLM's way, for contrast
    grad_d = {k: v.requires_grad for k, v in ck_model.named_parameters()}
    collm_sd = ck_model.state_dict()
    for k in list(collm_sd):
        if k in grad_d and not grad_d[k]:
            del collm_sd[k]
    collm_leak = [k for k in collm_sd if k.startswith("llama_model") and "lora_" not in k]
    check("control: CoLLM's name-based filter DOES leak the backbone",
          len(collm_leak) > 0,
          f"{len(collm_leak)} frozen tensors, "
          f"{sum(collm_sd[k].numel() for k in collm_leak)/1e6:.1f}M params")
    # and it must still be loadable
    c_re = build_cfg(args, llama_path, 2)
    c_re.model.ckpt = ck_path
    reloaded = mmod.MiniGPT4RecQFormer.from_config(c_re.model).float()
    gr, g0 = dict(reloaded.named_parameters()), dict(ck_model.named_parameters())
    keys = [k for k in saved_ck if k in gr]
    dmax = max(float((gr[k].detach().float() - g0[k].detach().float()).abs().max())
               for k in keys)
    check("the slim checkpoint reloads exactly", dmax < 1e-5, f"max diff={dmax:.2e}")
    del ck_model, reloaded

    # ------------------------------------------- stage-1 -> stage-2 LoRA handoff
    # A silent failure here would look exactly like "stage 2 is worse than stage 1":
    # the LoRA would be at init while the prompt had already changed.
    print("\n=== stage 1 -> stage 2: does ckpt_lora actually land? ===")
    from omegaconf import OmegaConf as _OC
    c1b = _OC.create(_OC.to_container(build_cfg(args, llama_path, 2)))
    c1b.model.arch = "mini_gpt4rec_v2"
    c1b.model.prompt_path = os.path.join(ROOT, "prompts/tallrec_movie.txt")
    c1b.model.freeze_rec, c1b.model.freeze_proj, c1b.model.freeze_lora = True, True, False
    for k in ["qformer", "loss", "n_titles_kept", "diag_log_freq"]:
        c1b.model.pop(k, None)
    mm1 = registry.get_model_class("mini_gpt4rec_v2").from_config(c1b.model)
    with torch.no_grad():                       # pretend stage 1 trained
        for n, prm in mm1.named_parameters():
            if prm.requires_grad:
                prm.add_(torch.randn_like(prm) * 0.05)
    grad_dic = {k: v.requires_grad for k, v in mm1.named_parameters()}
    sd1 = mm1.state_dict()
    for k in list(sd1):                         # exactly the runner's filter
        if k in grad_dic and not grad_dic[k]:
            del sd1[k]
    ck1 = os.path.join(args.work_dir, "stage1_lora_probe.pth")
    torch.save({"model": sd1}, ck1)
    lora_keys = [k for k in sd1 if "lora_" in k]
    check("stage-1 checkpoint carries the LoRA tensors", len(lora_keys) > 0,
          f"{len(lora_keys)} lora keys of {len(sd1)} saved")
    del mm1

    c2b = _OC.create(_OC.to_container(build_cfg(args, llama_path, 2)))
    c2b.model.ckpt_lora = ck1
    loaded = mmod.MiniGPT4RecQFormer.from_config(c2b.model)
    freshm = mmod.MiniGPT4RecQFormer.from_config(build_cfg(args, llama_path, 2).model)
    gl, gfr = dict(loaded.named_parameters()), dict(freshm.named_parameters())
    common = [k for k in lora_keys if k in gl]
    # LoRA is fp16 under peft>=0.5, so compare at fp16 resolution, not 1e-6
    d_load = max(float((gl[k].detach().float() - sd1[k].float()).abs().max()) for k in common)
    d_fresh = max(float((gfr[k].detach().float() - sd1[k].float()).abs().max()) for k in common)
    check("stage-2 LoRA equals the stage-1 checkpoint (fp16 tolerance)",
          d_load < 1e-3, f"max|loaded-ckpt|={d_load:.2e}")
    check("control: a fresh init does NOT match (test is not vacuous)",
          d_fresh > 1e-2, f"max|fresh-ckpt|={d_fresh:.2e}")
    del loaded, freshm

    # ------------------------------------------------------------ lr schedule
    # The inherited warmup gated on the global step but ramped on the per-epoch
    # step, so with warmup_steps > iters_per_epoch it sawtoothed and then jumped
    # to full lr -- the rise-then-collapse signature. Assert it is monotonic.
    print("\n=== lr schedule: warmup must ramp monotonically ===")
    sched_cls = registry.get_lr_scheduler_class("linear_warmup_cosine_lr_scaled")
    dummy = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    dummy.param_groups[0]["lr_scale"] = 1.0
    sch = sched_cls(optimizer=dummy, max_epoch=120, iters_per_epoch=100,
                    min_lr=1e-5, init_lr=1e-4, warmup_steps=300, warmup_start_lr=1e-6)
    warm = [sch.lr_at(e, s_) for e in range(3) for s_ in range(100)]
    check("warmup is monotonically increasing (no per-epoch sawtooth)",
          all(b >= a for a, b in zip(warm, warm[1:])),
          f"start={warm[0]:.2e} end={warm[-1]:.2e}")
    check("warmup ends at init_lr, no jump afterwards",
          abs(sch.lr_at(3, 0) - 1e-4) / 1e-4 < 0.01
          and abs(warm[-1] - 1e-4) / 1e-4 < 0.02,
          f"lr[epoch3,step0]={sch.lr_at(3, 0):.3e} vs init_lr=1.0e-04")
    check("cosine decays after warmup and respects min_lr",
          sch.lr_at(60, 0) < sch.lr_at(3, 0) and sch.lr_at(119, 99) >= 1e-5 * 0.99,
          f"mid={sch.lr_at(60, 0):.2e} end={sch.lr_at(119, 99):.2e}")
    dummy.param_groups[0]["lr_scale"] = 0.1
    sch.step(60, 0)
    check("lr_scale is applied per group",
          abs(dummy.param_groups[0]["lr"] - 0.1 * sch.lr_at(60, 0)) < 1e-12)

    # ---------------------------------------------------- stage 1 (legacy arch)
    # Stage 1 runs CoLLM's `mini_gpt4rec_v2`, not our subclass, and nothing else
    # here covers it -- so a break in that path would only show up as a stage-1
    # run that never learns.
    print("\n=== stage 1: legacy mini_gpt4rec_v2 + TALLRec prompt ===")
    c1 = _OC.create(_OC.to_container(build_cfg(args, llama_path, 2)))
    c1.model.arch = "mini_gpt4rec_v2"
    c1.model.prompt_path = os.path.join(ROOT, "prompts/tallrec_movie.txt")
    c1.model.freeze_rec, c1.model.freeze_proj, c1.model.freeze_lora = True, True, False
    for k in ["qformer", "loss", "n_titles_kept", "diag_log_freq"]:
        c1.model.pop(k, None)
    c1.run.user_grouped_sampler = False
    c1.run.lr_scale = {"lora": 1.0, "qformer": 1.0, "proto": 1.0, "rec": 0.1}
    c1.run.output_dir = os.path.join(args.work_dir, "out_stage1")

    m1 = registry.get_model_class("mini_gpt4rec_v2").from_config(c1.model).float()
    m1.set_mode("v2")
    tr1 = {n for n, p in m1.named_parameters() if p.requires_grad}
    check("stage 1: only LoRA is trainable",
          tr1 and all("lora_" in n for n in tr1), f"{len(tr1)} tensors, e.g. {sorted(tr1)[:1]}")
    check("stage 1: MF and llama_proj frozen",
          not any(n.startswith(("rec_encoder", "llama_proj")) for n in tr1))

    # the TALLRec prompt must actually receive the history titles
    prompt1 = m1.prompt_list[0]
    check("stage 1 prompt is the text-only TALLRec one",
          "<ItemTitleList>" in prompt1 and "<PrefTokens>" not in prompt1
          and "<UserID>" not in prompt1, prompt1[:60])
    enc1, _ = m1.encode_recdata_v2(batch, ids_order=[0, 1, 2])
    sp1, at1 = m1.recprompt_wrap_v2(enc1, batch, None, prompt1)
    dec = m1.llama_tokenizer.decode(
        m1.llama_tokenizer(prompt1.replace("<ItemTitleList>", batch["InteractedItemTitles"][0])
                           .replace("<TargetItemTitle>", batch["TargetItemTitle"][0]),
                           add_special_tokens=False).input_ids)
    check("stage 1: history titles are spliced into the prompt text",
          batch["InteractedItemTitles"][0].split('", "')[0].strip('"')[:12] in dec,
          f"...{dec[60:130]}...")

    before1 = {n: p.detach().clone() for n, p in m1.named_parameters() if p.requires_grad}
    out1 = m1.forward_v2(batch)
    check("stage 1: forward loss is finite", bool(torch.isfinite(out1["loss"])),
          f"loss={float(out1['loss']):.4f}")
    out1["loss"].backward()
    g1 = [n for n, p in m1.named_parameters()
          if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0)]
    # lora_A gets zero grad at init because lora_B starts at zero -- expected
    check("stage 1: LoRA receives gradients (lora_B at minimum)",
          len(g1) < len(before1), f"{len(before1) - len(g1)}/{len(before1)} nonzero")
    opt1 = torch.optim.AdamW([p for p in before1.values() if True] and
                             [p for n, p in m1.named_parameters() if p.requires_grad], lr=1e-3)
    opt1.step()
    moved = [n for n, p in m1.named_parameters()
             if p.requires_grad and not torch.equal(p.detach(), before1[n])]
    check("stage 1: an optimizer step actually moves LoRA weights",
          len(moved) > 0, f"{len(moved)}/{len(before1)} tensors changed")
    m1.eval()
    ev1 = m1.generate_for_samples(batch)
    check("stage 1: eval returns finite logits",
          bool(torch.isfinite(ev1["logits"]).all()) and ev1["logits"].shape[0] == 16)
    check("stage 1: eval has no n_prompt_tokens (why the runsheet shows prompt_tokens=nan)",
          "n_prompt_tokens" not in ev1)
    del m1

    # ------------------------------------------------- fp16/GradScaler contract
    # The gap that let `ValueError: Attempting to unscale FP16 gradients` reach a
    # real run: every test here used amp: False, so the GradScaler path with an
    # fp16-loaded backbone was never exercised. GradScaler needs CUDA, so assert
    # the precondition it enforces -- no trainable tensor may have an fp16 grad.
    print("\n=== fp16 / GradScaler contract ===")
    cfg_amp = build_cfg(args, llama_path, 2)
    cfg_amp.run.amp = True
    m_amp = mmod.MiniGPT4RecQFormer.from_config(cfg_amp.model)   # NOT .float()
    m_amp.set_mode("v2")
    pre = {n: p.dtype for n, p in m_amp.named_parameters() if p.requires_grad}
    check("stage 2 trainable params are fp32 before any cast (fp16 backbone only)",
          all(d == torch.float32 for d in pre.values()),
          f"dtypes={sorted({str(d) for d in pre.values()})}")

    # stage 3 unfreezes LoRA, which peft>=0.5 creates in the base fp16 dtype
    cfg3 = build_cfg(args, llama_path, 3)
    cfg3.model.ckpt = os.path.join(args.work_dir, "stage2_ckpt.pth")
    m3 = mmod.MiniGPT4RecQFormer.from_config(cfg3.model)
    lora_dtypes = {str(p.dtype) for n, p in m3.named_parameters()
                   if p.requires_grad and "lora_" in n}
    print(f"        LoRA dtypes as peft created them: {sorted(lora_dtypes) or 'none'}")
    # the runner's cast is what makes GradScaler viable
    runner_cls = registry.get_runner_class("rec_runner_qformer")
    runner_cls._cast_trainable_to_fp32(
        type("C", (), {"config": type("R", (), {"run_cfg": {"fp32_trainable": True}})()})(),
        m3,
    )
    after = {str(p.dtype) for n, p in m3.named_parameters() if p.requires_grad}
    check("after the runner cast, every trainable param is fp32",
          after == {"torch.float32"}, f"dtypes={sorted(after)}")
    check("frozen backbone stays fp16 (memory unchanged)",
          any(p.dtype == torch.float16 for n, p in m3.named_parameters()
              if not p.requires_grad),
          "at least one frozen fp16 tensor remains")
    m3 = m3.float()
    m3.set_mode("v2")
    out3 = m3.forward_v2(batch)
    out3["loss"].backward()
    bad = [n for n, p in m3.named_parameters()
           if p.requires_grad and p.grad is not None and p.grad.dtype == torch.float16]
    check("no trainable tensor produces an fp16 gradient (GradScaler precondition)",
          not bad, f"offenders={bad[:3]}")
    del m_amp, m3

    # ------------------------------------------------------ ablation configs
    print("\n=== every ablation row from the spec builds + trains one step ===")
    from omegaconf import OmegaConf

    ablations = {
        "full": {},
        "-memory": {"qformer.memory.k_hist": 0, "qformer.memory.k_neighbor": 0,
                    "qformer.memory.k_genre": 0, "qformer.memory.k_cluster": 0},
        "-hist": {"qformer.memory.k_hist": 0},
        "-pit-history": {"qformer.memory.history_source": "train_only",
                         "qformer.memory.k_hist": 10},
        "k_hist=10": {"qformer.memory.k_hist": 10},
        "k_hist=20": {"qformer.memory.k_hist": 20},
        "-neighbor": {"qformer.memory.k_neighbor": 0},
        "-genre": {"qformer.memory.k_genre": 0},
        "-cluster": {"qformer.memory.k_cluster": 0},
        "-candidate": {"qformer.use_candidate": False},
        "L=1": {"qformer.n_query": 1},
        "L=8": {"qformer.n_query": 8},
        "-anticollapse": {"loss.lambda_div": 0.0, "loss.lambda_attn": 0.0,
                          "loss.lambda_var": 0.0},
        "-rank": {"loss.lambda_rank": 0.0},
        "-norm-match": {"qformer.match_llm_norm": False},
        "-slot-prior": {"qformer.use_slot_prior": False},
        "+titles": {"n_titles_kept": 3,
                    "prompt_path": os.path.join(ROOT, "prompts/qformer_movie_titles.txt")},
        "terse-prompt": {"prompt_path": os.path.join(ROOT, "prompts/qformer_movie_terse.txt")},
        "n_layers=1": {"qformer.n_layers": 1},
        "n_layers=3": {"qformer.n_layers": 3},
        "dot-glue": {"soft_token_glue": "."},
    }
    for name, overrides in ablations.items():
        mcfg = OmegaConf.create(OmegaConf.to_container(build_cfg(args, llama_path, 2).model))
        for k, v in overrides.items():
            OmegaConf.update(mcfg, k, v, merge=True)
        try:
            m = mmod.MiniGPT4RecQFormer.from_config(mcfg).float()
            m.set_mode("v2")
            out = m.forward_v2(batch)
            out["loss"].backward()
            m.eval()
            ev = m.generate_for_samples(batch)
            n_slots = m.memory_encoder.n_slots
            ok = (torch.isfinite(out["loss"]) and torch.isfinite(ev["logits"]).all()
                  and not any(p.grad is not None and not torch.isfinite(p.grad).all()
                              for p in m.parameters()))
            check(f"ablation '{name}'", bool(ok),
                  f"L={m.n_query} slots={n_slots} "
                  f"hist={m.memory_encoder.history_source}/{m.memory_encoder.k_hist} "
                  f"loss={float(out['loss']):.4f} "
                  f"tokens={float(ev['n_prompt_tokens']):.1f}")
        except Exception as e:  # noqa: BLE001
            check(f"ablation '{name}'", False, f"{type(e).__name__}: {str(e)[:160]}")
        del m

    print("\n=== k_hist > pit_hist_width must fail loudly ===")
    mcfg = OmegaConf.create(OmegaConf.to_container(build_cfg(args, llama_path, 2).model))
    OmegaConf.update(mcfg, "qformer.memory.k_hist", 80, merge=True)
    try:
        m = mmod.MiniGPT4RecQFormer.from_config(mcfg).float()
        m.set_mode("v2")
        m.forward_v2(batch)
        check("k_hist beyond the dataset width raises", False, "it silently accepted 80")
    except AssertionError as e:
        check("k_hist beyond the dataset width raises", "pit_hist_width" in str(e),
              str(e)[:120])

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED ({len(_failures)}):")
        for f in _failures:
            print("  -", f)
        sys.exit(1)
    print("integration test passed")


if __name__ == "__main__":
    main()
