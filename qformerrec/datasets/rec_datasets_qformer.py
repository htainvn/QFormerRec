"""Dataset + builders for the Q-Former runs.

Identical row semantics to CoLLM's ``MoiveOOData`` / ``AmazonOOData`` (same
columns, same 10-item history cap, same quoting of titles), with one fix:

CoLLM's ML-1M warm/cold builder points at ``test_warm_cold_ood2.pkl`` and filters
on a ``warm`` column. Neither exists in the released ML-1M pickles (they carry a
``not_cold`` flag on ``test_ood2.pkl``, which is what the Amazon path uses), so
that split raises ``FileNotFoundError``. Here both datasets take the warm/cold
split from ``not_cold`` on the ordinary test file.

On top of that it emits one extra field the Q-Former needs, ``PitHistItems``: the
row's own **point-in-time** history, most-recent-first and ``HIST_PAD`` padded to
a fixed width. That is the same ``his`` column CoLLM renders into
``<ItemTitleList>``, so the memory bank reads the same user behaviour the
baseline does, and it needs no offline precomputation.

Two details:

* CoLLM caps history at 10 (``self.max_lenght = min(max_length_, 10)``) *only*
  because it is rendered as text. Memory slots cost zero prompt tokens, so the
  width here is a config value (``pit_hist_width``, default 50). ML-1M's mean
  history is ~38 (train) / ~75 (test).
* Index 0 is the padding item. Verified on the released pickles: every row's
  ``his`` starts with 0 and 0 never appears at a later position, so dropping it
  is unambiguous.
"""

import logging
import os

import numpy as np
import pandas as pd
from minigpt4.common.registry import registry
from minigpt4.datasets.builders.rec_base_dataset_builder import RecBaseDatasetBuilder
from minigpt4.datasets.datasets.rec_base_dataset import RecBaseDataset
from minigpt4.datasets.datasets.rec_datasets import convert_title_list_v2

from qformerrec.models.qformer_cie import HIST_PAD

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(HERE, "..", "configs", "datasets")


class RecOODDataset(RecBaseDataset):
    """CoLLM OOD split reader. ``ann_paths[0]`` may carry ``=warm`` / ``=cold``."""

    MAX_HIS = 10  # CoLLM's cap, for the *rendered* history titles only

    def __init__(self, text_processor=None, ann_paths=None, pit_hist_width=50):
        super().__init__()
        self.pit_hist_width = int(pit_hist_width)
        parts = ann_paths[0].split("=")
        self.annotation = pd.read_pickle(parts[0] + "_ood2.pkl").reset_index(drop=True)

        if len(parts) > 1:
            assert "not_cold" in self.annotation.columns, (
                f"{parts[0]} has no 'not_cold' column, cannot build a warm/cold split"
            )
            keep = 1 if "warm" in parts[1:] else 0
            self.annotation = self.annotation[self.annotation["not_cold"] == keep].copy()
            self.annotation = self.annotation.reset_index(drop=True)

        self.use_his = "his" in self.annotation.columns or "sessionItems" in self.annotation.columns
        assert self.use_his, "expected a 'his' column in the CoLLM OOD pickles"
        used = ["uid", "iid", "title", "his", "his_title", "label"]
        renamed = ["UserID", "TargetItemID", "TargetItemTitle", "InteractedItemIDs",
                   "InteractedItemTitles", "label"]
        self.prompt_flag = "not_cold" in self.annotation.columns
        if self.prompt_flag:
            used.append("not_cold")
            renamed.append("prompt_flag")
        self.annotation = self.annotation[used]
        self.annotation.columns = renamed
        self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
        self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)

        self.user_num = self.annotation["UserID"].max() + 1
        self.item_num = self.annotation["TargetItemID"].max() + 1
        self.text_processor = text_processor
        self.max_lenght = min(
            max(len(x) for x in self.annotation["InteractedItemIDs"].values), self.MAX_HIS
        )
        hl = np.array([len(self._pit_history(h)) for h in
                       self.annotation["InteractedItemIDs"].values])
        print(f"data path: {parts[0]} data size: {self.annotation.shape} "
              f"rendered-title cap: {self.max_lenght} pos_rate: "
              f"{self.annotation['label'].mean():.4f} | point-in-time history: "
              f"mean={hl.mean():.1f} median={np.median(hl):.0f} max={hl.max()} "
              f"empty={np.mean(hl == 0):.2%} -> using width {self.pit_hist_width}")
        logging.info("loaded %s rows=%d pit_hist mean=%.1f width=%d",
                     ann_paths[0], len(self.annotation), hl.mean(), self.pit_hist_width)

    @staticmethod
    def _pit_history(his):
        """The row's own history, oldest-first, with the index-0 padding dropped."""
        return [int(x) for x in his if int(x) != 0]

    def _pit_hist_slots(self, his):
        """(pit_hist_width,) int64, column j = j-th MOST RECENT item, -1 padded.

        Most-recent-first so that column index == recency rank, which is what the
        recency embedding and the 1/(rank+1) attention prior both assume, and so
        that the model can slice ``[:, :k_hist]`` for any k_hist <= width.
        """
        used = self._pit_history(his)[-self.pit_hist_width:][::-1]
        out = np.full(self.pit_hist_width, HIST_PAD, dtype=np.int64)
        out[: len(used)] = used
        return out

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]
        a = ann["InteractedItemIDs"]
        interacted_num = len(a)
        if a[0] == 0:            # leading 0 is the padding item, not a real one
            interacted_num -= 1
        if len(a) < self.max_lenght:
            b = [0] * (self.max_lenght - len(a)) + list(a)
        elif len(a) > self.max_lenght:
            b = a[-self.max_lenght:]
            interacted_num = self.max_lenght
        else:
            b = a
        sample = {
            "UserID": ann["UserID"],
            # the Q-Former's history slots: this row's own point-in-time history
            "PitHistItems": self._pit_hist_slots(a),
            "InteractedItemIDs_pad": np.array(b),
            "InteractedItemTitles": convert_title_list_v2(
                ann["InteractedItemTitles"][-interacted_num:] if interacted_num > 0 else []
            ),
            "TargetItemID": ann["TargetItemID"],
            "TargetItemTitle": '"' + str(ann["TargetItemTitle"]).strip(" ") + '"',
            "InteractedNum": interacted_num,
            "label": ann["label"],
        }
        if self.prompt_flag:
            sample["prompt_flag"] = ann["prompt_flag"]
        return sample


class _QFormerBuilder(RecBaseDatasetBuilder):
    train_dataset_cls = RecOODDataset
    CONFIG_SUBDIR = "movielens"

    @classmethod
    def default_config_path(cls, type="default"):
        # kept inside this package so no file has to be added to the CoLLM tree
        return os.path.abspath(os.path.join(CONFIG_DIR, cls.CONFIG_SUBDIR, "default.yaml"))

    def build_datasets(self, evaluate_only=False):
        logging.info("Building datasets...")
        self.build_processors()
        storage_path = self.config.build_info.storage
        assert os.path.exists(storage_path), f"storage path {storage_path} does not exist"

        # width of the emitted point-in-time history; must be >= the model's k_hist
        width = int(self.config.get("pit_hist_width", 50))
        cls = self.train_dataset_cls
        tp = self.text_processors["train"]

        def _ds(name):
            return cls(text_processor=tp, ann_paths=[os.path.join(storage_path, name)],
                       pit_hist_width=width)

        datasets = {"train": _ds("train"), "valid": _ds("valid_small"), "test": _ds("test")}
        if evaluate_only:
            datasets["test_warm"] = _ds("test=warm")
            datasets["test_cold"] = _ds("test=cold")
        return datasets


@registry.register_builder("movie_ood_qf")
class MovieOODQFormerBuilder(_QFormerBuilder):
    CONFIG_SUBDIR = "movielens"
    DATASET_CONFIG_DICT = {"default": "configs/datasets/movielens/default.yaml"}


@registry.register_builder("amazon_ood_qf")
class AmazonOODQFormerBuilder(_QFormerBuilder):
    CONFIG_SUBDIR = "amazon"
    DATASET_CONFIG_DICT = {"default": "configs/datasets/amazon/default.yaml"}
