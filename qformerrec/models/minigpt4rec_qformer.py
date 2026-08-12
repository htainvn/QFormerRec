"""CoLLM with a candidate-aware Q-Former CIE (``mini_gpt4rec_qformer``).

Subclasses CoLLM's ``MiniGPT4Rec_v2`` and reuses everything that already works:
MF, the LoRA wiring, the ``<UserID>`` / ``<TargetItemID>`` MLP mapping, the
Yes/No answer-logit read-out, and the AUC/UAUC evaluator.

What is new:
  * a per-user collaborative memory bank (25 slots, 5 slot types),
  * candidate-conditioned queries (per-token FiLM) reading that bank,
  * ``L`` preference tokens spliced into the prompt as ``<PrefTokens>``,
    replacing CoLLM's ``<ItemTitleList>`` history titles,
  * within-user ranking + anti-collapse auxiliary losses.
"""

import json
import logging
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt4.common.registry import registry
from minigpt4.models.minigpt4rec_v2 import MiniGPT4Rec_v2

from qformerrec.models.qformer_cie import (
    ABLATION_MODES,
    PREF_TOKEN_FLAG,
    SLOT_TYPE_NAMES,
    TITLE_LIST_FLAG,
    CFAuxHead,
    CandidateAwareQueryGenerator,
    CollabQFormer,
    LLMProjector,
    MemoryEncoder,
    NoNormLLMProjector,
    ablate_content,
    attention_disagreement_loss,
    centre_log_prior,
    global_rank_loss,
    llm_target_rms,
    mean_offdiag_cosine,
    soft_slot_plan,
    split_title_list,
    token_decorrelation_loss,
    variance_hinge_loss,
    vocab_align_loss,
    within_user_rank_loss,
)

def _as_bool(v, name):
    """Strict bool coercion for config flags that gate a control.

    ``bool("False")`` is ``True``, so a flag that arrives as a string -- which is
    what a stray quote in an ``--options`` override produces -- would silently turn
    a control OFF and the run would look like a control that found nothing. Fail
    instead.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    raise ValueError(
        f"{name}={v!r} is not a boolean. Pass it unquoted, e.g. "
        f"--options {name}=False"
    )


def _to_dict(node):
    """omegaconf DictConfig (or None) -> plain dict."""
    if node is None:
        return {}
    if isinstance(node, dict):
        return dict(node)
    from omegaconf import OmegaConf

    return OmegaConf.to_container(node, resolve=True) or {}


@registry.register_model("mini_gpt4rec_qformer")
class MiniGPT4RecQFormer(MiniGPT4Rec_v2):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4rec.yaml",
    }

    def __init__(
        self,
        qformer_cfg=None,
        loss_cfg=None,
        n_titles_kept=0,
        diag_log_freq=100,
        soft_token_glue="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        print("running MiniGPT4RecQFormer ...... ")

        qf = _to_dict(qformer_cfg)
        ls = _to_dict(loss_cfg)
        self.loss_cfg = {
            "lambda_rank": float(ls.get("lambda_rank", 0.5)),
            # AUC's direct surrogate. 0.0 keeps every earlier run bit-identical.
            # `lambda_rank` optimises within-user order (UAUC); this one optimises
            # order across all pairs, which is what AUC measures.
            "lambda_rank_global": float(ls.get("lambda_rank_global", 0.0)),
            "lambda_cf": float(ls.get("lambda_cf", 0.2)),
            "lambda_div": float(ls.get("lambda_div", 0.1)),
            "lambda_attn": float(ls.get("lambda_attn", 0.05)),
            "lambda_var": float(ls.get("lambda_var", 0.05)),
            "lambda_align": float(ls.get("lambda_align", 0.0)),
            "rank_score": str(ls.get("rank_score", "yes_minus_no")),
        }
        self.n_titles_kept = int(n_titles_kept)
        self.diag_log_freq = int(diag_log_freq)
        # CoLLM glues repeated <unk> placeholders with "." . Measured on the
        # LLaMA tokenizer, consecutive "<unk><unk>" already tokenise to two
        # separate unk ids, so the default here is no glue -- it saves L-1
        # prompt tokens. Set to "." for byte-exact parity with CoLLM's spacing.
        self.soft_token_glue = str(soft_token_glue)

        d1 = self.rec_encoder.config.embedding_size
        d_llm = self.llama_model.config.hidden_size
        d_q = int(qf.get("d_q", 128))
        self.n_query = int(qf.get("n_query", 4))

        mem_cfg = _to_dict(qf.get("memory"))
        index_path = mem_cfg.get("memory_index_path")
        assert index_path and os.path.exists(index_path), (
            f"memory_index_path not found: {index_path!r}. "
            "Build it first with scripts/build_memory.py"
        )
        with open(index_path, "rb") as f:
            memory_index = pickle.load(f)
        self._check_memory_index(memory_index, d1)

        self.memory_encoder = MemoryEncoder(
            memory_index,
            d1=d1,
            d_q=d_q,
            k_hist=mem_cfg.get("k_hist", 50),
            history_source=mem_cfg.get("history_source", "pit"),
            k_neighbor=mem_cfg.get("k_neighbor", 8),
            k_genre=mem_cfg.get("k_genre", 3),
            k_cluster=mem_cfg.get("k_cluster", 3),
            dropout=float(qf.get("dropout", 0.1)),
            use_slot_prior=bool(qf.get("use_slot_prior", True)),
            use_title=_as_bool(mem_cfg.get("use_title", False), "model.qformer.memory.use_title"),
            title_pca_dim=int(mem_cfg.get("title_pca_dim", 0)),
            ablate=str(mem_cfg.get("ablate", "none")),
            use_user_slot=_as_bool(
                mem_cfg.get("user_slot", True), "model.qformer.memory.user_slot"
            ),
        )
        # CONTROL, one level above the memory bank: ablate the L soft tokens
        # themselves, i.e. what actually reaches the LLM. Together with
        # `memory.ablate` this separates the two failure modes that look identical
        # from the outside -- "the memory says nothing useful but the LLM does read
        # the tokens" (memory ablation moves the metric, pref ablation moves it
        # more) from "the LLM ignores the soft tokens entirely" (neither moves it).
        # The attention prior orders slots inside a type; it was also shifting mass
        # BETWEEN types, handicapping history ~3 logits against the prototypes.
        # See `centre_log_prior`. False restores the pre-2026-08-13 behaviour.
        self.prior_centred = _as_bool(
            qf.get("prior_centred", True), "model.qformer.prior_centred"
        )
        self.ablate_pref = str(qf.get("ablate_pref", "none"))
        assert self.ablate_pref in ABLATION_MODES, (
            f"qformer.ablate_pref={self.ablate_pref!r} not in {ABLATION_MODES}"
        )
        # CONTROL: the two CIE tokens, separately. `<UserID>` is the only per-user
        # ID embedding in the prompt and therefore the only surface on which the
        # model can memorise a user rather than generalise about them -- 256 free
        # dimensions per user against ~46 training rows once stage 3 unfreezes MF.
        # Splitting it from `<TargetItemID>` matters because they share `llama_proj`
        # and so cannot be told apart by any weight-level inspection.
        self.ablate_user_id = str(qf.get("ablate_user_id", "none"))
        self.ablate_item_id = str(qf.get("ablate_item_id", "none"))
        for n, v in (("ablate_user_id", self.ablate_user_id),
                     ("ablate_item_id", self.ablate_item_id)):
            assert v in ABLATION_MODES, f"qformer.{n}={v!r} not in {ABLATION_MODES}"
        self.query_gen = CandidateAwareQueryGenerator(
            d1=d1, d_q=d_q, n_query=self.n_query,
            use_candidate=bool(qf.get("use_candidate", True)),
        )
        self.qformer = CollabQFormer(
            d_q=d_q,
            n_layers=int(qf.get("n_layers", 2)),
            n_heads=int(qf.get("n_heads", 4)),
            dropout=float(qf.get("dropout", 0.1)),
        )

        self.match_llm_norm = bool(qf.get("match_llm_norm", True))
        emb_w = self.llama_model.get_input_embeddings().weight
        self.llm_emb_rms = llm_target_rms(emb_w, d_llm)
        self.llm_emb_norm_mean = float(emb_w.detach().float().norm(dim=-1).mean().item())
        proj_cls = LLMProjector if self.match_llm_norm else NoNormLLMProjector
        self.pref_proj = proj_cls(
            d_q, d_llm, target_rms=self.llm_emb_rms, dropout=float(qf.get("dropout", 0.1))
        )
        me = self.memory_encoder
        banner = (
            f"[qformer] d_q={d_q} L={self.n_query} slots={me.n_slots} "
            f"layers={len(self.qformer.layers)} llm_emb_norm={self.llm_emb_norm_mean:.4f} "
            f"target_rms={self.llm_emb_rms:.6f} match_llm_norm={self.match_llm_norm} "
            f"use_title={me.use_title} prior_centred={self.prior_centred}"
        )
        if me.use_title:
            banner += f" d_text={me.d_text}"
            if me.title_pca_dim > 0:
                banner += f" title_pca={me.title_pca_dim} (evr={me.title_pca_evr:.3f})"
        # A control run must be identifiable from its log alone -- these three
        # lines are the difference between a result and an unexplained number.
        if (me.ablate != "none" or self.ablate_pref != "none" or not me.use_user_slot
                or self.ablate_user_id != "none" or self.ablate_item_id != "none"):
            banner += (f"  CONTROL: memory.ablate={me.ablate} ablate_pref={self.ablate_pref} "
                       f"user_slot={me.use_user_slot} ablate_user_id={self.ablate_user_id} "
                       f"ablate_item_id={self.ablate_item_id}")
        print(banner)
        logging.info(banner)

        # Make the handicap a number in every log, not a thing to rediscover.
        with torch.no_grad():
            k = max(me.k_hist, 1)
            hist_lp = float(-torch.arange(1, k + 1).float().log().mean())
            nb_lp = (float(me.neighbor_sim[me.neighbor_sim > 0].clamp_min(1e-3).log().mean())
                     if me.k_neighbor > 0 and bool((me.neighbor_sim > 0).any()) else 0.0)
        pl = (f"[prior] mean log-prior by type: user/genre/cluster=0.00  "
              f"hist={hist_lp:.2f}  neighbor={nb_lp:.2f}  -> centred={self.prior_centred}")
        print(pl)
        logging.info(pl)
        if not self.prior_centred and min(hist_lp, nb_lp) < -1.0:
            w = (f"WARNING: uncentred prior puts history {-hist_lp:.1f} and neighbours "
                 f"{-nb_lp:.1f} logits below every prototype slot before any learning. "
                 "type_bias has to overcome that just to reach parity, and measured it "
                 "never does (softmax stays ~0.167). Set qformer.prior_centred=True.")
            print(w)
            logging.warning(w)

        self.cf_head = CFAuxHead(d_q)

        self.pos_ans_id = None
        self.neg_ans_id = None
        self._step = 0
        self._diag = {}
        self._reset_diag()

    # ------------------------------------------------------------------ #
    # setup helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_memory_index(mi, d1):
        required = [
            "hist_items", "neighbors", "neighbor_sim", "genres", "user_cluster",
            "user_has_train", "genre_proto_init", "cluster_proto_init", "meta",
        ]
        missing = [k for k in required if k not in mi]
        assert not missing, f"memory index is missing keys: {missing}"
        n_users = mi["hist_items"].shape[0]
        for k in ["neighbors", "neighbor_sim", "genres", "user_cluster", "user_has_train"]:
            assert mi[k].shape[0] == n_users, f"{k} has {mi[k].shape[0]} rows, expected {n_users}"
        assert mi["meta"]["mf_embedding_size"] == d1, (
            f"memory index built for d1={mi['meta']['mf_embedding_size']}, model uses {d1}"
        )
        logging.info("memory index meta: %s", json.dumps(mi["meta"], default=str))

    def set_answer_type(self, mode):
        super().set_answer_type(mode)
        self.pos_ans_id = self.llama_tokenizer(self.pos_ans[0], add_special_tokens=False).input_ids[0]
        self.neg_ans_id = self.llama_tokenizer(self.neg_ans[0], add_special_tokens=False).input_ids[0]

    def _reset_diag(self):
        self._diag = {
            "n": 0, "cos": 0.0, "z_norm": 0.0, "z_std": 0.0,
            "loss_bce": 0.0, "loss_rank": 0.0, "loss_cf": 0.0,
            "loss_div": 0.0, "loss_attn": 0.0, "loss_var": 0.0, "loss_align": 0.0,
            "rank_pairs": 0.0, "hist_unk_rate": 0.0, "hist_slots_filled": 0.0,
            "attn_by_type": np.zeros((self.n_query, len(SLOT_TYPE_NAMES))),
            # mean number of *unmasked* slots of each type per row. Needed because
            # attention mass per type is not comparable across types: `user` is one
            # slot, `hist` is up to 50, so 0.5 vs 0.5 is a 50x difference per slot.
            "count_by_type": np.zeros(len(SLOT_TYPE_NAMES)),
            # within-history attention shape -- see _accumulate_diag
            "hist_attn_entropy": 0.0, "hist_attn_top1": 0.0,
            "hist_attn_n_valid": 0.0,
            "hist_attn_ent_ratio": 0.0, "hist_attn_ent_ratio_n": 0.0,
        }

    # ------------------------------------------------------------------ #
    # the Q-Former CIE forward
    # ------------------------------------------------------------------ #
    def encode_recdata_qformer(self, samples):
        """Returns a dict of soft embeddings plus the tensors the losses need.

        The whole CIE block runs in fp32 (``autocast`` disabled): the attention
        softmax, the anti-collapse losses and the RMS matching are all easier to
        trust in fp32, and the block is tiny compared with the frozen LLM.
        """
        uid = samples["UserID"].long()
        iid = samples["TargetItemID"].long()
        B = uid.shape[0]
        d_llm = self.llama_model.config.hidden_size

        with torch.cuda.amp.autocast(enabled=False):
            all_users, all_items = self.rec_encoder.computer()

            def item_emb_fn(ids):
                return self.rec_encoder.item_encoder(ids, all_items=all_items).float()

            def user_emb_fn(ids):
                return self.rec_encoder.user_encoder(ids, all_users=all_users).float()

            user_emb = user_emb_fn(uid)                       # (B, d1)
            item_emb = item_emb_fn(iid)                       # (B, d1)

            # ---- CoLLM's own CIE tokens for <UserID> / <TargetItemID>
            user_tok = self.llama_proj(user_emb.unsqueeze(-2)).reshape(B, self.proj_token_num, d_llm)
            item_tok = self.llama_proj(item_emb.unsqueeze(-2)).reshape(B, self.proj_token_num, d_llm)
            user_tok = ablate_content(user_tok, self.ablate_user_id)
            item_tok = ablate_content(item_tok, self.ablate_item_id)

            # ---- memory bank -> candidate-aware queries -> preference tokens
            pit_hist = samples.get("PitHistItems")
            if pit_hist is not None:
                pit_hist = pit_hist.to(uid.device)
            mem, mask, type_ids, prior, mem_stats = self.memory_encoder(
                uid, user_emb, item_emb_fn, user_emb_fn, pit_hist=pit_hist
            )
            queries, cand_code = self.query_gen(item_emb)
            attn_bias = self.query_gen.slot_bias(type_ids)                     # (B, L, N)
            lp = (centre_log_prior(prior, type_ids, mask) if self.prior_centred
                  else torch.log(prior.clamp_min(1e-6)))
            attn_bias = attn_bias + lp.unsqueeze(1)
            Z, A = self.qformer(queries, mem, mask, attn_bias)                 # (B,L,d_q), (B,L,N)
            pref_tok = self.pref_proj(Z)                                       # (B, L, d_llm)
            # CONTROL: what the LLM actually receives. Applied after the RMS
            # matching so the tokens keep the norm the projector was tuned to
            # produce -- the control is "these tokens carry no information", not
            # "these tokens are the wrong size".
            pref_tok = ablate_content(pref_tok, self.ablate_pref)
            s_cf = self.cf_head(Z, cand_code)

        return {
            "PrefTokens": pref_tok,
            "UserToken": user_tok,
            "TargetItemToken": item_tok,
            "Z": Z,
            "A": A,
            "mask": mask,
            "type_ids": type_ids,
            "mem_stats": mem_stats,
            "s_cf": s_cf,
        }

    # ------------------------------------------------------------------ #
    # prompt splicing
    # ------------------------------------------------------------------ #
    def soft_slot_plan(self, prompt):
        return soft_slot_plan(prompt, self.n_query, self.proj_token_num)

    def recprompt_wrap_qformer(self, encoded, samples, prompt):
        """CoLLM's unk-substitution splicing, extended with ``<PrefTokens>``.

        Each soft placeholder is expanded into ``n`` ``<unk>`` tokens (joined by
        ``.`` exactly as CoLLM does), the prompt is tokenised with left padding,
        and the soft embeddings are written into the ``<unk>`` positions. Since
        ``torch.nonzero`` walks the mask row-major, concatenating the soft blocks
        in prompt order lines them up with the ``<unk>`` positions.
        """
        assert prompt, "empty prompt"
        plan = self.soft_slot_plan(prompt)
        unk = self.llama_tokenizer.unk_token
        text = "<s>" + prompt
        for flag, n in plan:
            text = text.replace(flag, self.soft_token_glue.join([unk] * n))

        batch_size = samples["UserID"].shape[0]
        titles = samples.get("InteractedItemTitles", None)
        prompt_list = []
        for k in range(batch_size):
            p = text.replace("<TargetItemTitle>", samples["TargetItemTitle"][k])
            if TITLE_LIST_FLAG in p:
                kept = ""
                if titles is not None and self.n_titles_kept > 0:
                    kept = ", ".join(split_title_list(titles[k])[-self.n_titles_kept :])
                p = p.replace(TITLE_LIST_FLAG, kept if kept else "unkow")
            prompt_list.append(p)

        if not self.has_print_prompt:
            print("prompt example:", random.choice(prompt_list))
            self.has_print_prompt = True

        self.llama_tokenizer.padding_side = "left"
        tokens = self.llama_tokenizer(
            prompt_list,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(samples["UserID"].device)

        if not getattr(self, "has_pri_decode", False):
            print(
                "#######prompt decoded example: ",
                " ".join(self.llama_tokenizer.batch_decode(tokens.input_ids[0])),
            )
            self.has_pri_decode = True

        prompt_embeds = self.llama_model.model.embed_tokens(tokens.input_ids)
        if not plan:
            return prompt_embeds, tokens.attention_mask, tokens.attention_mask.sum(-1).float().mean()

        blocks = {
            PREF_TOKEN_FLAG: encoded["PrefTokens"],
            "<UserID>": encoded["UserToken"],
            "<TargetItemID>": encoded["TargetItemToken"],
        }
        merged = torch.cat([blocks[f] for f, _ in plan], dim=1)      # (B, n_soft, d_llm)
        n_soft = sum(n for _, n in plan)
        assert merged.shape[1] == n_soft, (merged.shape, n_soft)

        unk_id = self.llama_tokenizer.unk_token_id
        replaced_idx = torch.nonzero(tokens.input_ids == unk_id)
        assert replaced_idx.shape[0] == batch_size * n_soft, (
            f"expected {batch_size * n_soft} <unk> slots but found {replaced_idx.shape[0]}. "
            "Either the prompt was truncated by max_txt_len, or a title tokenised to <unk>."
        )
        prompt_embeds = prompt_embeds.clone()
        prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = merged.reshape(
            -1, merged.shape[-1]
        ).to(prompt_embeds.dtype)
        n_prompt_tokens = tokens.attention_mask.sum(-1).float().mean()
        return prompt_embeds, tokens.attention_mask, n_prompt_tokens

    # ------------------------------------------------------------------ #
    # shared forward core
    # ------------------------------------------------------------------ #
    def _llm_forward(self, samples, prompt):
        """Splice the prompt, run the LLM, return (s_yes, s_no, extras)."""
        use_qformer = PREF_TOKEN_FLAG in prompt
        encoded = self.encode_recdata_qformer(samples) if use_qformer else self._encode_ids_only(samples)
        sample_embeds, atts_samples, n_prompt_tokens = self.recprompt_wrap_qformer(
            encoded, samples, prompt
        )

        device = samples["UserID"].device
        self.llama_tokenizer.padding_side = "right"
        ans_ = {1: self.pos_ans[0], 0: self.neg_ans[0]}
        text = [ans_[int(t)] for t in samples["label"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(device)
        t_posi = to_regress_tokens.input_ids.shape[-1] + 1

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones(
            [atts_samples.shape[0], atts_samples.shape[1]], dtype=torch.long
        ).to(device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            llm = self.llama_model_lora if self.use_lora else self.llama_model
            outputs = llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        ans_logits = outputs.logits[:, -t_posi, :]
        s_yes = ans_logits[:, self.pos_ans_id]
        s_no = ans_logits[:, self.neg_ans_id]
        return s_yes, s_no, encoded, n_prompt_tokens

    def _encode_ids_only(self, samples):
        """CIE tokens only (no Q-Former) -- used when the prompt has no <PrefTokens>."""
        uid = samples["UserID"].long()
        iid = samples["TargetItemID"].long()
        B = uid.shape[0]
        d_llm = self.llama_model.config.hidden_size
        with torch.cuda.amp.autocast(enabled=False):
            all_users, all_items = self.rec_encoder.computer()
            u = self.rec_encoder.user_encoder(uid, all_users=all_users).float()
            i = self.rec_encoder.item_encoder(iid, all_items=all_items).float()
            ut = self.llama_proj(u.unsqueeze(-2)).reshape(B, self.proj_token_num, d_llm)
            it = self.llama_proj(i.unsqueeze(-2)).reshape(B, self.proj_token_num, d_llm)
            return {
                "PrefTokens": None,
                "UserToken": ablate_content(ut, self.ablate_user_id),
                "TargetItemToken": ablate_content(it, self.ablate_item_id),
                "Z": None, "A": None, "s_cf": None,
            }

    # ------------------------------------------------------------------ #
    # training / eval entry points (override the v2 paths)
    # ------------------------------------------------------------------ #
    def forward_v2(self, samples):
        prompt = self.prompt_list[0] if len(self.prompt_list) == 1 else random.choice(self.prompt_list)
        s_yes, s_no, enc, _ = self._llm_forward(samples, prompt)
        labels = samples["label"].float()

        # L_bce -- identical to CoLLM: BCE on the "Yes" logit at the answer slot
        loss_bce = F.binary_cross_entropy_with_logits(s_yes.float(), labels)
        total = loss_bce
        parts = {"loss_bce": loss_bce.detach()}
        lc = self.loss_cfg

        s_rank = (s_yes - s_no) if lc["rank_score"] == "yes_minus_no" else s_yes
        n_pairs = 0
        if lc["lambda_rank"] > 0:
            loss_rank, n_pairs = within_user_rank_loss(s_rank, samples["UserID"], labels)
            total = total + lc["lambda_rank"] * loss_rank
            parts["loss_rank"] = loss_rank.detach()
        if lc["lambda_rank_global"] > 0:
            loss_rg, _ = global_rank_loss(s_rank, labels)
            total = total + lc["lambda_rank_global"] * loss_rg
            parts["loss_rank_global"] = loss_rg.detach()

        if enc.get("Z") is not None:
            Z, A = enc["Z"], enc["A"]
            if lc["lambda_cf"] > 0:
                loss_cf = F.binary_cross_entropy_with_logits(enc["s_cf"].float(), labels)
                total = total + lc["lambda_cf"] * loss_cf
                parts["loss_cf"] = loss_cf.detach()
            if lc["lambda_div"] > 0:
                loss_div = token_decorrelation_loss(Z)
                total = total + lc["lambda_div"] * loss_div
                parts["loss_div"] = loss_div.detach()
            if lc["lambda_attn"] > 0:
                loss_attn = attention_disagreement_loss(A)
                total = total + lc["lambda_attn"] * loss_attn
                parts["loss_attn"] = loss_attn.detach()
            if lc["lambda_var"] > 0:
                loss_var = variance_hinge_loss(Z)
                total = total + lc["lambda_var"] * loss_var
                parts["loss_var"] = loss_var.detach()
            if lc["lambda_align"] > 0:
                loss_align = vocab_align_loss(
                    enc["PrefTokens"], self.llama_model.get_input_embeddings().weight
                )
                total = total + lc["lambda_align"] * loss_align
                parts["loss_align"] = loss_align.detach()
            self._accumulate_diag(enc, parts, n_pairs)

        self._step += 1
        return {"loss": total}

    @torch.no_grad()
    def generate_for_samples_v2(self, samples, return_all=False):
        prompt = self.prompt_list[0]
        s_yes, s_no, enc, n_prompt_tokens = self._llm_forward(samples, prompt)
        logits_ = s_yes
        loss = F.binary_cross_entropy_with_logits(logits_.float(), samples["label"].float())
        if return_all:
            return None, logits_
        out = {"loss": loss, "logits": logits_, "n_prompt_tokens": n_prompt_tokens}
        if enc.get("Z") is not None:
            out["token_cosine"] = mean_offdiag_cosine(enc["Z"]).detach()
        # history health on the split being evaluated -- the unk rate is ~0 on
        # train batches but nonzero on valid/test, and it is the number that
        # matters for the cold split
        for k, v in (enc.get("mem_stats") or {}).items():
            out[k] = v
        return out

    # ------------------------------------------------------------------ #
    # diagnostics -- the evidence that the L tokens are not collapsing
    # ------------------------------------------------------------------ #
    def _accumulate_diag(self, enc, parts, n_pairs):
        with torch.no_grad():
            Z, A = enc["Z"], enc["A"]
            d = self._diag
            d["n"] += 1
            d["cos"] += float(mean_offdiag_cosine(Z))
            d["z_norm"] += float(enc["PrefTokens"].float().norm(dim=-1).mean())
            d["z_std"] += float(Z.float().std(dim=0).mean())
            d["rank_pairs"] += n_pairs
            for k, v in parts.items():
                d[k] = d.get(k, 0.0) + float(v)
            for k, v in (enc.get("mem_stats") or {}).items():
                d[k] = d.get(k, 0.0) + float(v)
            # attention mass per (token, slot type). type_ids is per row now, so
            # the mass is accumulated with a scatter rather than a column slice.
            T = len(SLOT_TYPE_NAMES)
            tid = enc["type_ids"]                             # (B, N)
            Af = A.float()                                    # (B, L, N)
            onehot = F.one_hot(tid, T).to(Af.dtype)           # (B, N, T)
            per_type = torch.einsum("bln,bnt->lt", Af, onehot) / Af.shape[0]
            d["attn_by_type"] += per_type.cpu().numpy()
            valid = enc["mask"].to(Af.dtype)                   # (B, N)
            d["count_by_type"] += (
                torch.einsum("bn,bnt->t", valid, onehot) / Af.shape[0]
            ).cpu().numpy()

            # Shape of the attention INSIDE the history group. Both `attn_by_type`
            # and `per_slot` are blind to this by construction: a token spreading
            # 0.5 evenly over 30 history slots and a token putting 0.5 on one slot
            # have the same group total AND the same per-slot mean. Only the
            # distribution separates "the Q-Former selects" from "the Q-Former
            # mean-pools" -- and a mean over ~30 of a user's item vectors is close
            # to re-deriving that user's own vector, i.e. slot 0, which would
            # explain why destroying the whole bank costs 0.0005 AUC.
            #
            # Normalised against log(n_valid) PER ROW, not log(k_hist): ~half of
            # the 50 slots are padding on a typical row, so log(k_hist) is the
            # wrong ceiling and would make a uniform read look selective.
            k = self.memory_encoder.k_hist
            if k > 0:
                hm = enc["mask"][:, 1 : 1 + k].to(Af.dtype)          # (B, k)
                ah = Af[:, :, 1 : 1 + k] * hm.unsqueeze(1)
                ah = ah / ah.sum(-1, keepdim=True).clamp_min(1e-9)   # renormalise in-group
                ent = -(ah * ah.clamp_min(1e-9).log()).sum(-1)       # (B, L)
                n_valid = hm.sum(-1)                                 # (B,)
                d["hist_attn_entropy"] += float(ent.mean())
                d["hist_attn_top1"] += float(ah.max(-1).values.mean())
                d["hist_attn_n_valid"] += float(n_valid.mean())
                # rows with <2 valid slots have no entropy to speak of and a zero
                # denominator; excluding them is the difference between a ratio and
                # a division by log(1)
                rows = n_valid >= 2
                if bool(rows.any()):
                    d["hist_attn_ent_ratio"] += float(
                        (ent[rows] / n_valid[rows].log().unsqueeze(1)).mean()
                    )
                    d["hist_attn_ent_ratio_n"] += 1

        if self.diag_log_freq > 0 and self._step % self.diag_log_freq == 0:
            self.log_qformer_diagnostics()

    def log_qformer_diagnostics(self, output_dir=None, tag=""):
        d = self._diag
        if d["n"] == 0:
            return {}
        n = d["n"]
        summary = {
            "token_cosine_offdiag": d["cos"] / n,
            "pref_token_norm": d["z_norm"] / n,
            "llm_emb_norm": self.llm_emb_norm_mean,
            "z_std_mean": d["z_std"] / n,
            "rank_pairs_per_batch": d["rank_pairs"] / n,
            "proj_scale": float(getattr(self.pref_proj, "scale", torch.tensor(float("nan")))),
            # point-in-time history health: how many slots were actually filled and
            # what fraction of them fell back to the learned `unk_item` vector
            "hist_slots_filled": d["hist_slots_filled"] / n,
            "hist_unk_rate": d["hist_unk_rate"] / n,
            "history_source": self.memory_encoder.history_source,
            "k_hist": self.memory_encoder.k_hist,
        }
        if self.memory_encoder.k_hist > 0 and d.get("hist_attn_ent_ratio_n", 0):
            nr = d["hist_attn_ent_ratio_n"]
            summary["hist_attn_entropy"] = d["hist_attn_entropy"] / n
            summary["hist_attn_ent_ratio"] = d["hist_attn_ent_ratio"] / nr
            summary["hist_attn_top1"] = d["hist_attn_top1"] / n
            summary["hist_slots_valid"] = d["hist_attn_n_valid"] / n
        if not self.memory_encoder.use_user_slot:
            # under the `-user-slot` control, the fraction of rows that had nothing
            # else and kept slot 0 anyway. If this is not ~0 the control is leaky.
            summary["user_slot_kept"] = d.get("user_slot_kept", 0.0) / n
        for k in ["loss_bce", "loss_rank", "loss_rank_global", "loss_cf", "loss_div",
                  "loss_attn", "loss_var", "loss_align"]:
            if d.get(k, 0.0):
                summary[k] = d[k] / n
        attn = d["attn_by_type"] / n
        count = d["count_by_type"] / n
        # Per-slot mass. The per-type totals alone cannot answer "is the model
        # just re-reading slot 0?": `user` is always exactly one slot while `hist`
        # is up to 50, so equal type totals mean the user slot is ~50x heavier per
        # slot. This is the ratio that has to fall for the memory to be doing work.
        per_slot = attn / np.maximum(count, 1e-6)[None, :]
        # pooled over both history types, not the sum of their two per-slot rates:
        # `hist_unk` holds ~0.02 slots per row on ML-1M, so its own rate is a ratio
        # of two near-zero numbers and would dominate a sum.
        # Reported only when there ARE history slots: under `-memory`/`-hist` the
        # denominator is 0 and the ratio would come out as ~1e9 and fire the warning
        # below on a configuration that has nothing to warn about.
        hist_types = [SLOT_TYPE_NAMES.index("hist"), SLOT_TYPE_NAMES.index("hist_unk")]
        n_hist_slots = float(count[hist_types].sum())
        if n_hist_slots > 1e-3:
            hist_slot_mass = attn[:, hist_types].sum(axis=1) / n_hist_slots
            summary["user_vs_hist_per_slot"] = float(
                per_slot[:, SLOT_TYPE_NAMES.index("user")].mean()
                / max(float(hist_slot_mass.mean()), 1e-9)
            )
        with torch.no_grad():
            type_bias = self.query_gen.type_bias.detach().float().softmax(-1).cpu().numpy()

        msg = ["[qformer-diag]" + (f" {tag}" if tag else "")]
        msg.append("  " + "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in summary.items()
        ))
        msg.append("  slots_per_row=["
                   + " ".join(f"{SLOT_TYPE_NAMES[t]}:{count[t]:.2f}" for t in range(len(count)))
                   + "]")
        for l in range(attn.shape[0]):
            top = np.argsort(-attn[l])[:3]
            msg.append(
                f"  token{l}: attn_by_type=["
                + " ".join(f"{SLOT_TYPE_NAMES[t]}:{attn[l, t]:.3f}" for t in range(attn.shape[1]))
                + f"]  top3={[SLOT_TYPE_NAMES[t] for t in top]}"
                + "  per_slot=["
                + " ".join(f"{SLOT_TYPE_NAMES[t]}:{per_slot[l, t]:.4f}"
                           for t in range(attn.shape[1]))
                + "]  type_bias_softmax=["
                + " ".join(f"{v:.3f}" for v in type_bias[l])
                + "]"
            )
        if summary["token_cosine_offdiag"] > 0.6:
            msg.append("  WARNING: off-diagonal token cosine > 0.6 -- raise lambda_div to 0.3")
        ratio = summary["pref_token_norm"] / max(self.llm_emb_norm_mean, 1e-8)
        if ratio > 2.0 or ratio < 0.5:
            msg.append(f"  WARNING: pref-token norm is {ratio:.2f}x the LLM embedding norm")
        if summary.get("hist_attn_ent_ratio", 0.0) > 0.9:
            msg.append(
                f"  WARNING: within-history attention entropy is "
                f"{summary['hist_attn_ent_ratio']:.3f} of uniform (top1="
                f"{summary.get('hist_attn_top1', float('nan')):.4f} vs 1/n="
                f"{1.0 / max(summary.get('hist_slots_valid', 1.0), 1.0):.4f}) -- the "
                "Q-Former is mean-pooling the history, not selecting from it. A mean "
                "over a user's item vectors is close to that user's own vector, so the "
                "history slots would be re-deriving slot 0. Check the candidate path "
                "with qformer.use_candidate=False before changing the architecture."
            )
        if summary.get("user_vs_hist_per_slot", 0.0) > 10.0:
            msg.append(
                f"  WARNING: the user slot draws {summary['user_vs_hist_per_slot']:.0f}x the "
                "attention of an average history slot -- the bank is close to a "
                "re-read of slot 0. Control it with qformer.memory.user_slot=False"
            )
        logging.info("\n".join(msg))
        print("\n".join(msg))

        if output_dir is not None:
            payload = {
                "summary": summary,
                "attn_by_type": attn.tolist(),
                "count_by_type": count.tolist(),
                "attn_per_slot": per_slot.tolist(),
                "type_bias_softmax": type_bias.tolist(),
                "slot_type_names": SLOT_TYPE_NAMES,
                "tag": tag,
                "step": self._step,
            }
            path = os.path.join(output_dir, "qformer_diagnostics.jsonl")
            with open(path, "a") as f:
                f.write(json.dumps(payload) + "\n")
        self._reset_diag()
        return summary

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg):
        rec_config = cfg.get("rec_config")
        lora_config = cfg.get("lora_config")
        model = cls(
            rec_model=cfg.get("rec_model", "MF"),
            rec_config=rec_config,
            pretrained_rec=rec_config["pretrained_path"],
            freeze_rec=cfg.get("freeze_rec", True),
            rec_precision=cfg.get("rec_precision", "fp16"),
            llama_model=cfg.get("llama_model"),
            prompt_path=cfg.get("prompt_path", ""),
            prompt_template=cfg.get("prompt_template", ""),
            max_txt_len=cfg.get("max_txt_len", 32),
            end_sym=cfg.get("end_sym", "\n"),
            low_resource=cfg.get("low_resource", False),
            device_8bit=cfg.get("device_8bit", 0),
            proj_token_num=cfg.get("proj_token_num", 1),
            proj_drop=cfg.get("proj_drop", 0),
            lora_config=lora_config,
            proj_mid=cfg.get("proj_mid_times", 5),
            freeze_lora=cfg.get("freeze_lora", False),
            freeze_proj=cfg.get("freeze_proj", False),
            qformer_cfg=cfg.get("qformer"),
            loss_cfg=cfg.get("loss"),
            n_titles_kept=cfg.get("n_titles_kept", 0),
            diag_log_freq=cfg.get("diag_log_freq", 100),
            soft_token_glue=cfg.get("soft_token_glue", ""),
        )

        # `ckpt_lora` carries the cached stage-1 LoRA weights; `ckpt` carries the
        # stage-2 Q-Former/projector weights. Stage 3 needs both.
        for key in ["ckpt_lora", "ckpt"]:
            path = cfg.get(key, "")
            if path:
                assert os.path.exists(path), f"{key} not found: {path}"
                print(f"Load {key}: {path}")
                sd = torch.load(path, map_location="cpu")
                # Say WHICH run and WHICH epoch this file came from. A checkpoint
                # copied into a shared ckpt/ dir carries no output_dir and no log,
                # and `checkpoint_best.pth` from a run killed after its first
                # validation looks identical to a good one -- same name, same size,
                # same tensor list. That exact file was cached as the stage-1 LoRA
                # and silently initialised every stage-2/3 run; control A read it
                # as "text prompting is at chance" when it was "this LoRA saw 3 200
                # samples". The epoch was in the file the whole time; nothing
                # printed it.
                epoch = sd.get("epoch") if isinstance(sd, dict) else None
                prov = sd.get("provenance") if isinstance(sd, dict) else None
                if epoch is not None or prov:
                    print(f"  from epoch {epoch}"
                          + (f"  {prov.get('select_metric')}={prov.get('value'):.6f}"
                             if prov and prov.get("value") is not None else "")
                          + (f"  run={prov.get('output_dir')}" if prov else ""))
                if epoch == 0:
                    warn = (
                        f"WARNING: {key} is epoch 0 of its run. `checkpoint_best.pth` is "
                        "epoch 0 whenever a run was killed after its first validation, so "
                        "this may be a barely-trained checkpoint masquerading as a best "
                        f"one -- verify against the train.log of the run that produced it "
                        f"before trusting anything downstream of it. ({path})"
                    )
                    print(warn)
                    logging.warning(warn)
                sd = sd["model"] if "model" in sd else sd
                msg = model.load_state_dict(sd, strict=False)
                print(f"  unexpected keys: {list(msg.unexpected_keys)[:8]}")
                print(f"  n_missing={len(msg.missing_keys)} n_unexpected={len(msg.unexpected_keys)}")

        # never let a checkpoint silently overwrite a frozen pretrained MF
        pre = rec_config["pretrained_path"]
        if cfg.get("freeze_rec", True) and pre != "not_have" and os.path.exists(pre):
            model.rec_encoder.load_state_dict(torch.load(pre, map_location="cpu"))
            print("re-loaded the frozen pretrained MF weights")

        model.set_answer_type(mode=cfg.get("ans_type", "v2"))
        model.print_prompt()
        return model

    # keep the freeze switches explicit so ablations/stages are easy to audit
    def trainable_summary(self):
        groups = {}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            top = n.split(".")[0]
            groups[top] = groups.get(top, 0) + p.numel()
        return groups
