"""Task for the Q-Former runs.

Keeps CoLLM's metric definitions so AUC/UAUC stay comparable, and fixes three
things that break on a modern single-GPU setup:

1. ``RecBaseTask.evaluation``'s non-distributed branch calls
   ``results_logits_.cput()`` (typo) and assigns the 3-tuple from ``uAUC_me`` to
   a scalar, so it crashes as soon as ``world_size == 1`` -- which is exactly the
   single-A100 setup. Here the single-process path is the primary one.
2. ``uAUC_me`` returns NaN on scikit-learn >= ~1.3 (see ``uauc_score`` below).
   This is the dangerous one: it silently NaNs the primary metric.
3. Checkpoint selection. CoLLM's runner keys ``agg_metrics`` on AUC; the spec
   selects on validation UAUC, so ``agg_metrics`` is UAUC by default
   (``select_metric`` in the run config switches it back).

It also records what the efficiency table needs: mean prompt token count and
wall-clock inference time per split.
"""

import logging
import time

import numpy as np
import torch
import torch.distributed as dist
from minigpt4.common.dist_utils import is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample
from minigpt4.tasks.rec_base_task import RecBaseTask, uAUC_me  # noqa: F401
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


@registry.register_task("rec_qformer")
class RecQFormerTask(RecBaseTask):
    def __init__(self, select_metric="uauc", grad_clip=1.0, empty_cache_freq=0):
        super().__init__()
        self.select_metric = select_metric
        self.grad_clip = grad_clip
        self.empty_cache_freq = empty_cache_freq

    @classmethod
    def setup_task(cls, **kwargs):
        cfg = kwargs.get("cfg", None)
        kw = {}
        if cfg is not None:
            kw = {
                "select_metric": cfg.run_cfg.get("select_metric", "uauc"),
                "grad_clip": float(cfg.run_cfg.get("grad_clip", 1.0)),
                "empty_cache_freq": int(cfg.run_cfg.get("empty_cache_freq", 0)),
            }
        logging.info("RecQFormerTask: %s", kw)
        return cls(**kw)

    def valid_step(self, model, samples):
        return model.generate_for_samples(samples)

    # ------------------------------------------------------------------ #
    # training loop: CoLLM's, plus gradient clipping (the spec asks for 1.0,
    # and CoLLM's loop has none). With fp16 + GradScaler the gradients must be
    # unscaled before clipping, which is why the loop is spelled out here.
    # ------------------------------------------------------------------ #
    def _train_inner_loop(self, epoch, iters_per_epoch, model, data_loader, optimizer,
                          lr_scheduler, scaler=None, start_iters=None, log_freq=50,
                          cuda_enabled=False, accum_grad_iters=1, grad_clip=None):
        grad_clip = self.grad_clip if grad_clip is None else grad_clip
        use_amp = scaler is not None
        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        logging.info("Start training epoch %s, %s iters per inner epoch.", epoch, iters_per_epoch)
        header = "Train: data epoch: [{}]".format(epoch)
        inner_epoch = epoch if start_iters is None else start_iters // iters_per_epoch
        if start_iters is not None:
            header += "; inner epoch [{}]".format(inner_epoch)

        params = [p for g in optimizer.param_groups for p in g["params"]]
        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {"epoch": inner_epoch, "num_iters_per_epoch": iters_per_epoch, "iters": i}
            )
            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    if grad_clip and grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            # CoLLM empties the cache every step, which is a real slowdown; only
            # do it if the config asks for it (OOM mitigation).
            if self.empty_cache_freq and (i + 1) % self.empty_cache_freq == 0:
                torch.cuda.empty_cache()

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {k: "{:.3f}".format(m.global_avg) for k, m in metric_logger.meters.items()}

    def evaluation(self, model, data_loaders, cuda_enabled=True):
        # the runner always passes the default True; honour the actual hardware
        # so a CPU debug run works
        cuda_enabled = cuda_enabled and torch.cuda.is_available()
        model = model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("acc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = "Evaluation"

        loaders = getattr(data_loaders, "loaders", [data_loaders])
        print_freq = max(1, len(loaders[0]) // 5)

        results = {}
        for data_loader in loaders:
            logits_all, labels_all, users_all = [], [], []
            prompt_tokens, token_cos = [], []
            mem_stat_sums, mem_stat_n = {}, 0
            n_samples = 0
            if cuda_enabled and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            for samples in metric_logger.log_every(data_loader, print_freq, header):
                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                out = self.valid_step(model=model, samples=samples)
                logits = out["logits"].detach().float()
                logits_all.extend(logits.cpu().numpy())
                labels_all.extend(samples["label"].detach().cpu().numpy())
                users_all.extend(samples["UserID"].detach().cpu().numpy())
                if "n_prompt_tokens" in out:
                    prompt_tokens.append((float(out["n_prompt_tokens"]), logits.shape[0]))
                if "token_cosine" in out:
                    token_cos.append(float(out["token_cosine"]))
                for k in ("hist_unk_rate", "hist_slots_filled"):
                    if k in out:
                        mem_stat_sums[k] = mem_stat_sums.get(k, 0.0) + float(out[k])
                if mem_stat_sums:
                    mem_stat_n += 1
                n_samples += logits.shape[0]

                acc = ((logits > 0).float() == samples["label"].float()).float().mean()
                metric_logger.update(acc=acc.item())
                metric_logger.update(loss=out["loss"].item())
            if cuda_enabled and torch.cuda.is_available():
                torch.cuda.synchronize()
            wall = time.time() - t0

            logits_all = np.asarray(logits_all, dtype=np.float64)
            labels_all = np.asarray(labels_all)
            users_all = np.asarray(users_all)

            if is_dist_avail_and_initialized() and dist.get_world_size() > 1:
                logits_all, labels_all, users_all = self._all_gather(
                    logits_all, labels_all, users_all, model
                )

            auc = roc_auc_score(labels_all, logits_all)
            uauc, uauc_stats = uauc_score(users_all, logits_all, labels_all)
            metric_logger.synchronize_between_processes()

            mean_prompt_tokens = float("nan")
            if prompt_tokens:
                tot = sum(v * n for v, n in prompt_tokens)
                mean_prompt_tokens = tot / sum(n for _, n in prompt_tokens)

            results = {
                "auc": auc,
                "uauc": uauc,
                "acc": metric_logger.meters["acc"].global_avg,
                "loss": metric_logger.meters["loss"].global_avg,
                "n_samples": int(n_samples),
                "eval_seconds": round(wall, 2),
                "ms_per_sample": round(1000.0 * wall / max(n_samples, 1), 3),
                "mean_prompt_tokens": round(mean_prompt_tokens, 2),
                "uauc_users_scored": uauc_stats["n_scored"],
                "uauc_users_skipped": uauc_stats["n_single_row"] + uauc_stats["n_single_class"],
            }
            if token_cos:
                results["token_cosine_offdiag"] = round(float(np.mean(token_cos)), 4)
            for k, v in mem_stat_sums.items():
                results[k] = round(v / max(mem_stat_n, 1), 4)
            results["agg_metrics"] = uauc if self.select_metric == "uauc" else auc

            logging.info("UAUC users: %s", uauc_stats)
            extra = "" if not token_cos else f" ***token_cos: {results['token_cosine_offdiag']:.4f}"
            if "hist_unk_rate" in results:
                extra += (f" ***hist_slots: {results['hist_slots_filled']:.2f}"
                          f" ***hist_unk: {results['hist_unk_rate']:.2%}")
            logging.info(
                "Averaged stats: %s ***auc: %.6f ***uauc: %.6f ***prompt_tokens: %.2f "
                "***eval_s: %.1f (%.2f ms/sample, n=%d)%s",
                metric_logger.global_avg(), auc, uauc, mean_prompt_tokens, wall,
                results["ms_per_sample"], n_samples, extra,
            )
        return results

    @staticmethod
    def _all_gather(logits, labels, users, model):
        dev = next(model.parameters()).device
        out = []
        for arr, dtype in ((logits, torch.float64), (labels, torch.float64), (users, torch.long)):
            t = torch.as_tensor(arr, dtype=dtype, device=dev).contiguous()
            buf = [torch.empty_like(t) for _ in range(dist.get_world_size())]
            dist.all_gather(buf, t)
            out.append(torch.cat(buf, dim=0).cpu().numpy())
        dist.barrier()
        return out[0], out[1], out[2]
