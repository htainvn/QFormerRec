"""Runner for the Q-Former runs.

Adds three things CoLLM's ``RecRunnerBase`` cannot express:

1. **Per-group learning rates.** CoLLM's schedulers overwrite ``lr`` on *every*
   param group, so per-group ``lr`` set on the optimiser is silently destroyed.
   Groups here carry an ``lr_scale`` and the registered
   ``linear_warmup_cosine_lr_scaled`` scheduler multiplies by it, giving the
   stage-2/3 table (Q-Former 1x, prototypes 1x, MF 0.1x, LoRA 0.1x).
2. **The user-grouped batch sampler**, needed by ``L_rank``.
3. **Q-Former diagnostics** written once per epoch to the output dir.
"""

import logging
import math
import os

import torch
import webdataset as wds
from minigpt4.common.dist_utils import get_rank, get_world_size
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import ChainDataset
from minigpt4.datasets.datasets.dataloader_utils import IterLoader, MultiIterLoader, PrefetchLoader
from minigpt4.runners.runner_base_rec import RecRunnerBase
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from qformerrec.compat import attach_file_log, enable_live_output
from qformerrec.datasets.samplers import UserGroupedBatchSampler

PROTO_KEYS = ("memory_encoder.genre_proto", "memory_encoder.cluster_proto")


class _CPULoader:
    """PrefetchLoader's interface without the CUDA stream (CPU debug runs)."""

    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def __next__(self):        # MultiIterLoader asserts this attribute exists
        pass

    def __getattr__(self, name):
        return self.loader.__getattribute__(name)


@registry.register_lr_scheduler("linear_warmup_cosine_lr_scaled")
class LinearWarmupCosineLRSchedulerScaled:
    """CoLLM's ``linear_warmup_cosine_lr``, times ``lr_scale``, with the warmup fixed.

    CoLLM's warmup gates on the *global* step (``total < warmup_steps``) but ramps
    on the *within-epoch* step (``cur_step``). Whenever ``warmup_steps`` exceeds
    ``iters_per_epoch`` -- which is the shipped stage-1 setting, 300 vs 100 -- the
    ramp therefore restarts every epoch and the lr sawtooths, then jumps straight
    to the full ``init_lr`` the moment the global step passes ``warmup_steps``:

        epoch 0   1e-05 -> 3.4e-04      (ramp)
        epoch 1   1e-05 -> 3.4e-04      (ramp again)
        epoch 2   1e-05 -> 3.4e-04      (and again)
        epoch 3   1.0e-03              <- jump to full lr, no warmup benefit at all

    That produces the classic rise-then-collapse curve: the model improves while
    the lr is accidentally small, peaks at the end of "warmup", then degrades once
    the real lr lands. Ramping on ``total`` gives the monotonic warmup that was
    intended.
    """

    def __init__(self, optimizer, max_epoch, iters_per_epoch, min_lr, init_lr,
                 warmup_steps=0, warmup_start_lr=-1, **kwargs):
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def lr_at(self, cur_epoch, cur_step):
        total = cur_epoch * self.iters_per_epoch + cur_step
        if total < self.warmup_steps:
            frac = total / max(self.warmup_steps, 1)          # global, not per-epoch
            return self.warmup_start_lr + (self.init_lr - self.warmup_start_lr) * frac
        return (self.init_lr - self.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * total / (self.max_epoch * self.iters_per_epoch))
        ) + self.min_lr

    def step(self, cur_epoch, cur_step):
        lr = self.lr_at(cur_epoch, cur_step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr * pg.get("lr_scale", 1.0)


@registry.register_runner("rec_runner_qformer")
class RecRunnerQFormer(RecRunnerBase):
    def setup_output_dir(self):
        """As the base, plus a real log file next to the checkpoint.

        CoLLM's ``setup_logger`` installs only a StreamHandler, so a disconnected
        Colab session loses every log line. ``train.log`` lands in the run dir and
        is therefore picked up by the S3/Drive sync alongside the checkpoint and
        ``qformer_diagnostics.jsonl``.
        """
        super().setup_output_dir()
        enable_live_output()
        attach_file_log(self.output_dir)

    @property
    def model(self):
        """Same as ``RunnerBase.model`` but safe when no move is needed.

        The base property only assigns ``_wrapped_model`` inside the
        ``if self._model.device != self.device`` branch, so it returns ``None``
        whenever the model already sits on the target device -- which is every
        CPU debug run.
        """
        if self._wrapped_model is None:
            self._model = self._model.to(self.device)
            self._cast_trainable_to_fp32(self._model)
            if self.use_distributed:
                self._wrapped_model = DDP(self._model, device_ids=[self.config.run_cfg.gpu])
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    def _cast_trainable_to_fp32(self, model):
        """Keep trainable parameters in fp32 while the frozen backbone stays fp16.

        Vicuna is loaded with ``torch_dtype=float16``, and peft >= 0.5 casts new
        LoRA adapters to the *base layer's* dtype -- so on a modern peft the only
        trainable parameters in stage 1/3 come out fp16. ``GradScaler`` refuses
        those outright (``ValueError: Attempting to unscale FP16 gradients``), from
        ``scaler.step()`` as much as from an explicit ``scaler.unscale_()``, so with
        ``amp: True`` training cannot start at all.

        CoLLM never hit this because it pins peft 0.4.0, which left the adapters in
        fp32. Casting them back is therefore not a deviation -- it restores the
        precision CoLLM actually trained with, and it is the standard
        mixed-precision LoRA arrangement (frozen fp16 base, fp32 adapters). peft
        casts activations to the adapter dtype in its forward, so this is safe.

        Cost is negligible: ~4.2M LoRA params, i.e. ~17 MB extra.
        """
        if not bool(self.config.run_cfg.get("fp32_trainable", True)):
            logging.warning("fp32_trainable=False: leaving trainable params in their "
                            "loaded dtype. With amp: True this will fail in GradScaler.")
            return
        casted, n = [], 0
        for name, p in model.named_parameters():
            if p.requires_grad and p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.float()
                casted.append(name)
                n += p.numel()
        if casted:
            msg = (f"cast {len(casted)} trainable tensors ({n/1e6:.1f}M params) to fp32 "
                   f"(were fp16/bf16); frozen backbone left untouched. "
                   f"e.g. {casted[0]}")
            logging.info(msg)
            print("[runner]", msg)

    # ---------------------------------------------------------------- optim
    @property
    def optimizer(self):
        if self._optimizer is not None:
            return self._optimizer

        cfg = self.config.run_cfg
        scales = cfg.get("lr_scale", {}) or {}
        scale = {
            "qformer": float(scales.get("qformer", 1.0)),
            "proto": float(scales.get("proto", 1.0)),
            "rec": float(scales.get("rec", 0.1)),
            "lora": float(scales.get("lora", 0.1)),
        }
        wd = float(cfg.weight_decay)
        buckets = {k: {"wd": [], "no_wd": []} for k in scale}

        n_param, listing = 0, []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("rec_encoder"):
                g = "rec"
            elif "lora_" in n or n.startswith("llama_model_lora"):
                g = "lora"
            elif any(n.startswith(k) for k in PROTO_KEYS):
                g = "proto"
            else:
                g = "qformer"
            no_wd = p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n or "norm" in n.lower()
            buckets[g]["no_wd" if no_wd else "wd"].append(p)
            n_param += p.numel()
            listing.append((g, n, tuple(p.shape)))

        optim_params = []
        for g, b in buckets.items():
            for key, w in (("wd", wd), ("no_wd", 0.0)):
                if b[key]:
                    optim_params.append(
                        {"params": b[key], "weight_decay": w, "lr_scale": scale[g], "group": g}
                    )
        counts = {g: sum(p.numel() for k in b for p in b[k]) for g, b in buckets.items()}
        logging.info("trainable parameters: %d  by group: %s", n_param, counts)
        print(f"trainable parameters: {n_param}  by group: {counts}")
        for g, n, s in listing:
            print(f"  [{g}] {n} {s}")
        print("lr_scale:", scale, " init_lr:", float(cfg.init_lr))
        self._num_trainable_para = n_param > 0

        beta2 = cfg.get("beta2", 0.999)
        self._optimizer = torch.optim.AdamW(
            optim_params, lr=float(cfg.init_lr), weight_decay=wd, betas=(0.9, beta2)
        )
        return self._optimizer

    # ------------------------------------------------------------- loaders
    def create_loaders(self, datasets, num_workers, batch_sizes, is_trains, collate_fns,
                       dataset_ratios=None):
        run = self.config.run_cfg
        use_grouped = bool(run.get("user_grouped_sampler", True))
        n_users_per_batch = int(run.get("users_per_batch", 8))
        n_per_user = int(run.get("rows_per_user", 6))

        def _create_loader(dataset, nw, bsz, is_train, collate_fn):
            if isinstance(dataset, ChainDataset) or isinstance(dataset, wds.DataPipeline):
                return iter(DataLoader(dataset, batch_size=bsz, num_workers=nw, pin_memory=True))

            batch_sampler = None
            if is_train and use_grouped:
                ann = dataset.annotation
                batch_sampler = UserGroupedBatchSampler(
                    ann["UserID"].values, ann["label"].values,
                    n_users_per_batch=n_users_per_batch, n_per_user=n_per_user,
                    seed=int(run.get("seed", 42)) + get_rank(),
                )
                assert batch_sampler.batch_size >= 32 or not run.get("strict_batch", True), (
                    f"user-grouped batch is {batch_sampler.batch_size} (<32). L_var/L_div/L_rank "
                    "are computed inside a micro-batch; set strict_batch: False to override."
                )
                loader = DataLoader(
                    dataset, batch_sampler=batch_sampler, num_workers=nw,
                    pin_memory=True, collate_fn=collate_fn,
                )
            else:
                sampler = None
                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset, shuffle=is_train, num_replicas=get_world_size(), rank=get_rank()
                    )
                    if not self.use_dist_eval_sampler:
                        sampler = sampler if is_train else None
                loader = DataLoader(
                    dataset, batch_size=bsz, num_workers=nw, pin_memory=True, sampler=sampler,
                    shuffle=sampler is None and is_train, collate_fn=collate_fn,
                    drop_last=bool(is_train),
                )
            # CoLLM's PrefetchLoader needs a CUDA stream and calls .cuda() on
            # every batch; skip it on CPU so a CPU debug run of the real entry
            # point works. On GPU the genuine prefetcher is used.
            if torch.cuda.is_available():
                loader = PrefetchLoader(loader)
            else:
                loader = _CPULoader(loader)
            if is_train:
                loader = IterLoader(loader, use_distributed=self.use_distributed)
            return loader

        loaders = []
        for dataset, bsz, is_train, collate_fn in zip(datasets, batch_sizes, is_trains, collate_fns):
            if isinstance(dataset, (list, tuple)):
                if hasattr(dataset[0], "sample_ratio") and dataset_ratios is None:
                    dataset_ratios = [d.sample_ratio for d in dataset]
                loaders.append(
                    MultiIterLoader(
                        loaders=[
                            _create_loader(d, num_workers, bsz, is_train, collate_fn[i])
                            for i, d in enumerate(dataset)
                        ],
                        ratios=dataset_ratios,
                    )
                )
            else:
                loaders.append(_create_loader(dataset, num_workers, bsz, is_train, collate_fn))
        return loaders

    # --------------------------------------------------------- diagnostics
    def train_epoch(self, epoch):
        stats = super().train_epoch(epoch)
        model = self.unwrap_dist_model(self.model)
        if hasattr(model, "log_qformer_diagnostics"):
            model.log_qformer_diagnostics(output_dir=str(self.output_dir), tag=f"epoch{epoch}")
        return stats

    def log_stats(self, stats, split_name):
        super().log_stats(stats, split_name)
        if isinstance(stats, dict) and "uauc" in stats:
            logging.info(
                "[%s] auc=%.6f uauc=%.6f prompt_tokens=%s",
                split_name, stats.get("auc", float("nan")), stats["uauc"],
                stats.get("mean_prompt_tokens"),
            )
