#!/usr/bin/env python
"""Training / evaluation entry point for CoLLM-QFormer.

Same skeleton as CoLLM's ``train_collm_mf_din.py``; the only differences are the
sys.path bootstrap into the CoLLM checkout and the ``qformerrec`` imports that
register the new model / task / runner / builders.

    export COLLM_ROOT=/content/CoLLM
    python train_qformer.py --cfg-path train_configs/stage2_qformer_ml1m.yaml

Runs single-process by default (no torchrun): ``init_distributed_mode`` sets
``distributed=False`` when RANK/WORLD_SIZE are absent, which is what we want on
one A100.
"""

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

HERE = os.path.dirname(os.path.abspath(__file__))
COLLM_ROOT = os.environ.get("COLLM_ROOT", os.path.join(HERE, "..", "CoLLM"))
sys.path.insert(0, os.path.abspath(COLLM_ROOT))
sys.path.insert(0, HERE)

# must run BEFORE minigpt4 is imported: it stubs `decord`, which CoLLM imports at
# package level but which has no wheel for Python >= 3.11
from qformerrec.compat import (  # noqa: E402
    attach_file_log,
    check_environment,
    enable_live_output,
    install_import_shims,
)

# before anything prints: Colab's `!python` gives stdout an 8 KB block buffer, so
# without this a long run looks frozen until the buffer fills
enable_live_output()
install_import_shims()

import minigpt4.tasks as tasks  # noqa: E402
from minigpt4.common.config import Config  # noqa: E402
from minigpt4.common.dist_utils import get_rank, init_distributed_mode  # noqa: E402
from minigpt4.common.logger import setup_logger  # noqa: E402
from minigpt4.common.registry import registry  # noqa: E402
from minigpt4.common.utils import now  # noqa: E402
from minigpt4.datasets.builders import *  # noqa: E402,F401,F403
from minigpt4.models import *  # noqa: E402,F401,F403
from minigpt4.processors import *  # noqa: E402,F401,F403
from minigpt4.runners import *  # noqa: E402,F401,F403
from minigpt4.tasks import *  # noqa: E402,F401,F403

# registration of the new pieces
import qformerrec.datasets.rec_datasets_qformer  # noqa: E402,F401
import qformerrec.models.minigpt4rec_qformer  # noqa: E402,F401
import qformerrec.runners.runner_qformer  # noqa: E402,F401
import qformerrec.tasks.rec_qformer_task  # noqa: E402,F401


def parse_args():
    p = argparse.ArgumentParser(description="CoLLM-QFormer training")
    p.add_argument("--cfg-path", required=True, help="path to configuration file.")
    p.add_argument("--options", nargs="+",
                   help="override config settings in key=value form")
    return p.parse_args()


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    return registry.get_runner_class(cfg.run_cfg.get("runner", "rec_runner_qformer"))


def main():
    # strict: refuse to start on a known-bad stack rather than train for hours and
    # report loss=nan (which is exactly what transformers 5.x does here).
    check_environment(strict=os.environ.get("QFORMERREC_ALLOW_BAD_ENV") != "1")
    job_id = now()
    cfg = Config(parse_args())
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)

    # size the MF tables over all splits, exactly as CoLLM does
    data_dir = None
    for name in cfg.datasets_cfg:
        data_dir = cfg.datasets_cfg[name].path
        break
    assert data_dir is not None, "no dataset block in the config"
    print("data dir:", data_dir)
    ids = [pd.read_pickle(os.path.join(data_dir, f"{s}_ood2.pkl"))[["uid", "iid"]]
           for s in ["train", "valid", "test"]]
    user_num = int(max(d.uid.max() for d in ids)) + 1
    item_num = int(max(d.iid.max() for d in ids)) + 1
    del ids
    cfg.model_cfg.rec_config.user_num = user_num
    cfg.model_cfg.rec_config.item_num = item_num
    print(f"user_num={user_num} item_num={item_num}")
    cfg.pretty_print()

    model = task.build_model(cfg)
    print("trainable parameter groups:", model.trainable_summary()
          if hasattr(model, "trainable_summary") else "n/a")
    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
