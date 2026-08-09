  ckpt_lora: /content/ckpt/stage1_lora_ml1m.pth
  ckpt: /content/logs/stage2_ml1m/20260809155/checkpoint_best.pth
[compat] stubbed unused modules: decord (imported by CoLLM, never called here)
/usr/local/lib/python3.12/dist-packages/timm/models/hub.py:4: FutureWarning: Importing from timm.models.hub is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
/usr/local/lib/python3.12/dist-packages/transformers/utils/generic.py:441: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  _torch_pytree._register_pytree_node(
/usr/local/lib/python3.12/dist-packages/transformers/utils/generic.py:309: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  _torch_pytree._register_pytree_node(
/usr/local/lib/python3.12/dist-packages/transformers/utils/generic.py:309: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  _torch_pytree._register_pytree_node(
Some weights of LlamaForCausalLM were not initialized from the model checkpoint at /tmp/tmpbufroi3e and are newly initialized: ['model.layers.0.self_attn.rotary_emb.inv_freq', 'model.layers.1.self_attn.rotary_emb.inv_freq']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
[compat] python=3.12.13 torch=2.11.0+cu128 transformers=4.36.2 peft=0.9.0 tokenizers=0.15.2 scikit-learn=1.6.1
Not using distributed mode
data path: /content/data/ml-1m/train data size: (33891, 7) rendered-title cap: 10 pos_rate: 0.5368 | point-in-time history: mean=37.7 median=22 max=338 empty=3.76% -> using width 50
data path: /content/data/ml-1m/valid data size: (10401, 7) rendered-title cap: 10 pos_rate: 0.5283 | point-in-time history: mean=57.8 median=38 max=345 empty=0.99% -> using width 50
data path: /content/data/ml-1m/test data size: (7331, 7) rendered-title cap: 10 pos_rate: 0.5454 | point-in-time history: mean=74.8 median=42 max=445 empty=0.85% -> using width 50
data dir: /content/data/ml-1m/
user_num=839 item_num=3256
runing MiniGPT4Rec_v2 ...... 
Loading Rec_model
### rec_encoder: MF
creat MF model, user num: 839 item num: 3256
successfully load the pretrained model......
Loading Rec_model Done
Loading LLAMA
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
Loading checkpoint shards: 100% 2/2 [00:10<00:00,  5.38s/it]
Loading LLAMA Done
Setting Lora
Setting Lora Done
type: <class 'int'> 2
Load 1 training prompts
Prompt List: 
['#Question: A user\'s viewing preferences are encoded in the features <PrefTokens>. Using all available information, make a prediction about whether the user would enjoy the movie titled <TargetItemTitle>? Answer with "Yes" or "No". \\n#Answer:']
running MiniGPT4RecQFormer ...... 
[qformer] d_q=128 L=4 slots=65 layers=2 llm_emb_norm=1.2861 target_rms=0.020095 match_llm_norm=True
Load ckpt_lora: /content/ckpt/stage1_lora_ml1m.pth
  unexpected keys: []
  n_missing=733 n_unexpected=0
Load ckpt: /content/logs/stage2_ml1m/20260809155/checkpoint_best.pth
  unexpected keys: []
  n_missing=652 n_unexpected=0
answer token ids: pos: 3869 neg ids: 1939
Prompt Pos Example 
#Question: A user's viewing preferences are encoded in the features <PrefTokens>. Using all available information, make a prediction about whether the user would enjoy the movie titled <TargetItemTitle>? Answer with "Yes" or "No". \n#Answer: Yes or No
trainable parameter groups: {'rec_encoder': 1048320, 'llama_model': 4194304, 'memory_encoder': 53248, 'query_gen': 166040, 'qformer': 529408, 'pref_proj': 2167297, 'cf_head': 66049}
INFO:root:logging to /content/logs/stage3_ml1m/20260809164/train.log
[compat] logging to /content/logs/stage3_ml1m/20260809164/train.log
INFO:root:cast 128 trainable tensors (4.2M params) to fp32 (were fp16/bf16); frozen backbone left untouched. e.g. llama_model.model.layers.0.self_attn.q_proj.lora_A.default.weight
[runner] cast 128 trainable tensors (4.2M params) to fp32 (were fp16/bf16); frozen backbone left untouched. e.g. llama_model.model.layers.0.self_attn.q_proj.lora_A.default.weight
INFO:root:early stopping: patience=20 validations (50 steps each), on uauc
INFO:root:Start training
INFO:root:dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline).
INFO:root:Loaded 33891 records for train split from the dataset.
INFO:root:Loaded 10401 records for valid split from the dataset.
INFO:root:Loaded 7331 records for test split from the dataset.
INFO:root:UserGroupedBatchSampler: 33891 rows, 740 users, 640 users with both labels, batch=8x6=48
INFO:root:trainable parameters: 8224666  by group: {'qformer': 2969242, 'proj': 0, 'proto': 12800, 'rec': 1048320, 'lora': 4194304}
trainable parameters: 8224666  by group: {'qformer': 2969242, 'proj': 0, 'proto': 12800, 'rec': 1048320, 'lora': 4194304}
  [rec] rec_encoder.user_embedding.weight (839, 256)
  [rec] rec_encoder.item_embedding.weight (3256, 256)
  [lora] llama_model.model.layers.0.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.0.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.0.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.0.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.1.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.1.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.1.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.1.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.2.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.2.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.2.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.2.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.3.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.3.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.3.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.3.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.4.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.4.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.4.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.4.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.5.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.5.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.5.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.5.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.6.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.6.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.6.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.6.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.7.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.7.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.7.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.7.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.8.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.8.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.8.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.8.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.9.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.9.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.9.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.9.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.10.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.10.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.10.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.10.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.11.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.11.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.11.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.11.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.12.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.12.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.12.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.12.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.13.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.13.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.13.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.13.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.14.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.14.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.14.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.14.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.15.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.15.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.15.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.15.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.16.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.16.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.16.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.16.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.17.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.17.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.17.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.17.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.18.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.18.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.18.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.18.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.19.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.19.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.19.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.19.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.20.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.20.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.20.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.20.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.21.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.21.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.21.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.21.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.22.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.22.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.22.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.22.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.23.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.23.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.23.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.23.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.24.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.24.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.24.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.24.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.25.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.25.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.25.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.25.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.26.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.26.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.26.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.26.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.27.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.27.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.27.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.27.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.28.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.28.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.28.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.28.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.29.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.29.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.29.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.29.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.30.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.30.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.30.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.30.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.31.self_attn.q_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.31.self_attn.q_proj.lora_B.default.weight (4096, 8)
  [lora] llama_model.model.layers.31.self_attn.v_proj.lora_A.default.weight (8, 4096)
  [lora] llama_model.model.layers.31.self_attn.v_proj.lora_B.default.weight (4096, 8)
  [qformer] memory_encoder.unk_item (256,)
  [proto] memory_encoder.genre_proto.weight (18, 256)
  [proto] memory_encoder.cluster_proto.weight (32, 256)
  [qformer] memory_encoder.proj.weight (128, 256)
  [qformer] memory_encoder.type_emb.weight (6, 128)
  [qformer] memory_encoder.rank_emb.weight (50, 128)
  [qformer] memory_encoder.ln.weight (128,)
  [qformer] memory_encoder.ln.bias (128,)
  [qformer] query_gen.Q0 (4, 128)
  [qformer] query_gen.type_bias (4, 6)
  [qformer] query_gen.cand_proj.weight (128, 256)
  [qformer] query_gen.cand_proj.bias (128,)
  [qformer] query_gen.cand_ln.weight (128,)
  [qformer] query_gen.cand_ln.bias (128,)
  [qformer] query_gen.film.weight (1024, 128)
  [qformer] query_gen.film.bias (1024,)
  [qformer] query_gen.ln.weight (128,)
  [qformer] query_gen.ln.bias (128,)
  [qformer] qformer.layers.0.ln_self.weight (128,)
  [qformer] qformer.layers.0.ln_self.bias (128,)
  [qformer] qformer.layers.0.self_attn.q_proj.weight (128, 128)
  [qformer] qformer.layers.0.self_attn.q_proj.bias (128,)
  [qformer] qformer.layers.0.self_attn.k_proj.weight (128, 128)
  [qformer] qformer.layers.0.self_attn.k_proj.bias (128,)
  [qformer] qformer.layers.0.self_attn.v_proj.weight (128, 128)
  [qformer] qformer.layers.0.self_attn.v_proj.bias (128,)
  [qformer] qformer.layers.0.self_attn.out_proj.weight (128, 128)
  [qformer] qformer.layers.0.self_attn.out_proj.bias (128,)
  [qformer] qformer.layers.0.ln_cross.weight (128,)
  [qformer] qformer.layers.0.ln_cross.bias (128,)
  [qformer] qformer.layers.0.cross_attn.q_proj.weight (128, 128)
  [qformer] qformer.layers.0.cross_attn.q_proj.bias (128,)
  [qformer] qformer.layers.0.cross_attn.k_proj.weight (128, 128)
  [qformer] qformer.layers.0.cross_attn.k_proj.bias (128,)
  [qformer] qformer.layers.0.cross_attn.v_proj.weight (128, 128)
  [qformer] qformer.layers.0.cross_attn.v_proj.bias (128,)
  [qformer] qformer.layers.0.cross_attn.out_proj.weight (128, 128)
  [qformer] qformer.layers.0.cross_attn.out_proj.bias (128,)
  [qformer] qformer.layers.0.ln_ffn.weight (128,)
  [qformer] qformer.layers.0.ln_ffn.bias (128,)
  [qformer] qformer.layers.0.ffn.0.weight (512, 128)
  [qformer] qformer.layers.0.ffn.0.bias (512,)
  [qformer] qformer.layers.0.ffn.3.weight (128, 512)
  [qformer] qformer.layers.0.ffn.3.bias (128,)
  [qformer] qformer.layers.1.ln_self.weight (128,)
  [qformer] qformer.layers.1.ln_self.bias (128,)
  [qformer] qformer.layers.1.self_attn.q_proj.weight (128, 128)
  [qformer] qformer.layers.1.self_attn.q_proj.bias (128,)
  [qformer] qformer.layers.1.self_attn.k_proj.weight (128, 128)
  [qformer] qformer.layers.1.self_attn.k_proj.bias (128,)
  [qformer] qformer.layers.1.self_attn.v_proj.weight (128, 128)
  [qformer] qformer.layers.1.self_attn.v_proj.bias (128,)
  [qformer] qformer.layers.1.self_attn.out_proj.weight (128, 128)
  [qformer] qformer.layers.1.self_attn.out_proj.bias (128,)
  [qformer] qformer.layers.1.ln_cross.weight (128,)
  [qformer] qformer.layers.1.ln_cross.bias (128,)
  [qformer] qformer.layers.1.cross_attn.q_proj.weight (128, 128)
  [qformer] qformer.layers.1.cross_attn.q_proj.bias (128,)
  [qformer] qformer.layers.1.cross_attn.k_proj.weight (128, 128)
  [qformer] qformer.layers.1.cross_attn.k_proj.bias (128,)
  [qformer] qformer.layers.1.cross_attn.v_proj.weight (128, 128)
  [qformer] qformer.layers.1.cross_attn.v_proj.bias (128,)
  [qformer] qformer.layers.1.cross_attn.out_proj.weight (128, 128)
  [qformer] qformer.layers.1.cross_attn.out_proj.bias (128,)
  [qformer] qformer.layers.1.ln_ffn.weight (128,)
  [qformer] qformer.layers.1.ln_ffn.bias (128,)
  [qformer] qformer.layers.1.ffn.0.weight (512, 128)
  [qformer] qformer.layers.1.ffn.0.bias (512,)
  [qformer] qformer.layers.1.ffn.3.weight (128, 512)
  [qformer] qformer.layers.1.ffn.3.bias (128,)
  [qformer] qformer.ln_out.weight (128,)
  [qformer] qformer.ln_out.bias (128,)
  [qformer] pref_proj.scale ()
  [qformer] pref_proj.net.0.weight (512, 128)
  [qformer] pref_proj.net.0.bias (512,)
  [qformer] pref_proj.net.3.weight (4096, 512)
  [qformer] pref_proj.net.3.bias (4096,)
  [qformer] cf_head.net.0.weight (256, 256)
  [qformer] cf_head.net.0.bias (256,)
  [qformer] cf_head.net.2.weight (1, 256)
  [qformer] cf_head.net.2.bias (1,)
lr_scale: {'qformer': 1.0, 'proj': 0.1, 'proto': 1.0, 'rec': 2.0, 'lora': 2.0}  init_lr: 5e-06
/content/CoLLM/minigpt4/runners/runner_base.py:140: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  self._scaler = torch.cuda.amp.GradScaler()
INFO:root:Start training epoch 0, 50 iters per inner epoch.
/content/QFormerRec/qformerrec/tasks/rec_qformer_task.py:91: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
/content/QFormerRec/qformerrec/models/minigpt4rec_qformer.py:206: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
prompt example: <s>#Question: A user's viewing preferences are encoded in the features <unk><unk><unk><unk>. Using all available information, make a prediction about whether the user would enjoy the movie titled "Freejack (1992)"? Answer with "Yes" or "No". \n#Answer:
#######prompt decoded example:  </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> </s> <s> # Question : A user ' s view ing prefer ences are encoded in the features  <unk> <unk> <unk> <unk> . Using all available information , make a prediction about whether the user would enjoy the movie titled " Blue Vel vet ( 1 9 8 6 )" ? Answer with " Yes " or " No ". \ n # Answer :
/content/CoLLM/minigpt4/models/rec_model.py:48: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  return torch.cuda.amp.autocast(dtype=dtype)
/content/QFormerRec/qformerrec/models/minigpt4rec_qformer.py:503: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:836.)
  "proj_scale": float(getattr(self.pref_proj, "scale", torch.tensor(float("nan")))),
INFO:root:[qformer-diag]
  token_cosine_offdiag=0.0432  pref_token_norm=1.2976  llm_emb_norm=1.2861  z_std_mean=0.5410  rank_pairs_per_batch=61.0000  proj_scale=0.0203  hist_slots_filled=15.6042  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.5484  loss_rank=0.4928  loss_cf=0.5800  loss_div=0.0198  loss_attn=0.9969  loss_var=0.4628
  token0: attn_by_type=[user:0.099 hist:0.225 neighbor:0.155 genre:0.264 cluster:0.257 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.086 hist:0.203 neighbor:0.147 genre:0.269 cluster:0.296 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.082 hist:0.185 neighbor:0.169 genre:0.296 cluster:0.268 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.086 hist:0.203 neighbor:0.167 genre:0.269 cluster:0.276 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
[qformer-diag]
  token_cosine_offdiag=0.0432  pref_token_norm=1.2976  llm_emb_norm=1.2861  z_std_mean=0.5410  rank_pairs_per_batch=61.0000  proj_scale=0.0203  hist_slots_filled=15.6042  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.5484  loss_rank=0.4928  loss_cf=0.5800  loss_div=0.0198  loss_attn=0.9969  loss_var=0.4628
  token0: attn_by_type=[user:0.099 hist:0.225 neighbor:0.155 genre:0.264 cluster:0.257 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.086 hist:0.203 neighbor:0.147 genre:0.269 cluster:0.296 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.082 hist:0.185 neighbor:0.169 genre:0.296 cluster:0.268 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.086 hist:0.203 neighbor:0.167 genre:0.269 cluster:0.276 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
INFO:root:train epoch 0 iter 0/50 loss=0.9858 lr=1.000e-06
Train: data epoch: [0]  [ 0/50]  eta: 0:01:29  lr: 0.000001  loss: 0.9858  time: 1.7964  data: 0.0000  max mem: 37315
INFO:root:train epoch 0 iter 49/50 loss=0.8171 lr=3.450e-06
Train: data epoch: [0]  [49/50]  eta: 0:00:00  lr: 0.000003  loss: 0.8171  time: 0.7904  data: 0.0000  max mem: 41426
Train: data epoch: [0] Total time: 0:00:40 (0.8020 s / it)
INFO:root:Averaged stats: lr: 0.000002  loss: 1.069105
INFO:root:[qformer-diag] epoch0
  token_cosine_offdiag=0.0382  pref_token_norm=1.2971  llm_emb_norm=1.2861  z_std_mean=0.5267  rank_pairs_per_batch=64.0816  proj_scale=0.0203  hist_slots_filled=13.3750  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.6059  loss_rank=0.5375  loss_cf=0.6038  loss_div=0.0176  loss_attn=0.9969  loss_var=0.4756
  token0: attn_by_type=[user:0.100 hist:0.231 neighbor:0.162 genre:0.252 cluster:0.255 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.087 hist:0.208 neighbor:0.157 genre:0.258 cluster:0.290 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.084 hist:0.192 neighbor:0.175 genre:0.286 cluster:0.263 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.088 hist:0.207 neighbor:0.173 genre:0.258 cluster:0.274 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
[qformer-diag] epoch0
  token_cosine_offdiag=0.0382  pref_token_norm=1.2971  llm_emb_norm=1.2861  z_std_mean=0.5267  rank_pairs_per_batch=64.0816  proj_scale=0.0203  hist_slots_filled=13.3750  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.6059  loss_rank=0.5375  loss_cf=0.6038  loss_div=0.0176  loss_attn=0.9969  loss_var=0.4756
  token0: attn_by_type=[user:0.100 hist:0.231 neighbor:0.162 genre:0.252 cluster:0.255 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.087 hist:0.208 neighbor:0.157 genre:0.258 cluster:0.290 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.084 hist:0.192 neighbor:0.175 genre:0.286 cluster:0.263 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.088 hist:0.207 neighbor:0.173 genre:0.258 cluster:0.274 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
INFO:root:Evaluating on valid.
Evaluation  [  0/163]  eta: 0:02:02  loss: 0.6396  acc: 0.6250  time: 0.7504  data: 0.2046  max mem: 41426
Evaluation  [ 32/163]  eta: 0:01:05  loss: 0.5830  acc: 0.6875  time: 0.4903  data: 0.0009  max mem: 41426
Evaluation  [ 64/163]  eta: 0:00:49  loss: 0.7358  acc: 0.5312  time: 0.4923  data: 0.0009  max mem: 41426
Evaluation  [ 96/163]  eta: 0:00:32  loss: 0.7874  acc: 0.5000  time: 0.4844  data: 0.0009  max mem: 41426
Evaluation  [128/163]  eta: 0:00:17  loss: 0.5338  acc: 0.7656  time: 0.4921  data: 0.0009  max mem: 41426
Evaluation  [160/163]  eta: 0:00:01  loss: 0.4521  acc: 0.8438  time: 0.4778  data: 0.0009  max mem: 41426
Evaluation  [162/163]  eta: 0:00:00  loss: 0.6627  acc: 0.6364  time: 0.4687  data: 0.0037  max mem: 41426
Evaluation Total time: 0:01:19 (0.4897 s / it)
INFO:root:UAUC users: {'n_users': 356, 'n_scored': 282, 'n_single_row': 30, 'n_single_class': 44}
INFO:root:Averaged stats: loss: 0.643656  acc: 0.644337 ***auc: 0.694528 ***uauc: 0.690052 ***prompt_tokens: 67.44 ***eval_s: 79.8 (7.67 ms/sample, n=10401) ***token_cos: 0.0297 ***hist_slots: 32.75 ***hist_unk: 0.48%
INFO:root:Saved checkpoint at epoch 0 to /content/logs/stage3_ml1m/20260809164/checkpoint_best.pth (339 tensors, 12.4M params, 115.8 MB)
INFO:root:[valid] auc=0.694528 uauc=0.690052 prompt_tokens=67.44
INFO:root:[valid] epoch 0 uauc=0.690052 | best=0.690052 @ epoch 0 | no improvement for 0/20 validations  <-- new best
INFO:root:Start training
INFO:root:Start training epoch 1, 50 iters per inner epoch.
/content/QFormerRec/qformerrec/tasks/rec_qformer_task.py:91: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
/content/QFormerRec/qformerrec/models/minigpt4rec_qformer.py:206: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
/content/CoLLM/minigpt4/models/rec_model.py:48: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  return torch.cuda.amp.autocast(dtype=dtype)
INFO:root:train epoch 1 iter 0/50 loss=1.1612 lr=3.500e-06
Train: data epoch: [1]  [ 0/50]  eta: 0:00:41  lr: 0.000004  loss: 1.1612  time: 0.8221  data: 0.0000  max mem: 41426
INFO:root:train epoch 1 iter 49/50 loss=1.0396 lr=4.958e-06
Train: data epoch: [1]  [49/50]  eta: 0:00:00  lr: 0.000005  loss: 1.0396  time: 0.7741  data: 0.0000  max mem: 41426
Train: data epoch: [1] Total time: 0:00:38 (0.7744 s / it)
INFO:root:Averaged stats: lr: 0.000005  loss: 1.053900
INFO:root:[qformer-diag] epoch1
  token_cosine_offdiag=0.0341  pref_token_norm=1.2969  llm_emb_norm=1.2861  z_std_mean=0.5241  rank_pairs_per_batch=65.2000  proj_scale=0.0203  hist_slots_filled=13.1788  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.6004  loss_rank=0.5120  loss_cf=0.6108  loss_div=0.0165  loss_attn=0.9969  loss_var=0.4778
  token0: attn_by_type=[user:0.101 hist:0.233 neighbor:0.162 genre:0.250 cluster:0.255 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.087 hist:0.210 neighbor:0.157 genre:0.258 cluster:0.289 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.084 hist:0.194 neighbor:0.175 genre:0.285 cluster:0.262 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.087 hist:0.209 neighbor:0.173 genre:0.256 cluster:0.276 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
[qformer-diag] epoch1
  token_cosine_offdiag=0.0341  pref_token_norm=1.2969  llm_emb_norm=1.2861  z_std_mean=0.5241  rank_pairs_per_batch=65.2000  proj_scale=0.0203  hist_slots_filled=13.1788  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.6004  loss_rank=0.5120  loss_cf=0.6108  loss_div=0.0165  loss_attn=0.9969  loss_var=0.4778
  token0: attn_by_type=[user:0.101 hist:0.233 neighbor:0.162 genre:0.250 cluster:0.255 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.087 hist:0.210 neighbor:0.157 genre:0.258 cluster:0.289 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.084 hist:0.194 neighbor:0.175 genre:0.285 cluster:0.262 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.087 hist:0.209 neighbor:0.173 genre:0.256 cluster:0.276 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
INFO:root:Evaluating on valid.
Evaluation  [  0/163]  eta: 0:02:04  loss: 0.6356  acc: 0.6406  time: 0.7621  data: 0.2185  max mem: 41426
Evaluation  [ 32/163]  eta: 0:01:05  loss: 0.5657  acc: 0.6562  time: 0.4929  data: 0.0010  max mem: 41426
Evaluation  [ 64/163]  eta: 0:00:49  loss: 0.8571  acc: 0.4219  time: 0.4943  data: 0.0009  max mem: 41426
Evaluation  [ 96/163]  eta: 0:00:33  loss: 0.7416  acc: 0.5781  time: 0.4850  data: 0.0009  max mem: 41426
Evaluation  [128/163]  eta: 0:00:17  loss: 0.5111  acc: 0.7812  time: 0.4923  data: 0.0009  max mem: 41426
Evaluation  [160/163]  eta: 0:00:01  loss: 0.4289  acc: 0.8906  time: 0.4774  data: 0.0009  max mem: 41426
Evaluation  [162/163]  eta: 0:00:00  loss: 0.6280  acc: 0.6364  time: 0.4685  data: 0.0040  max mem: 41426
Evaluation Total time: 0:01:20 (0.4913 s / it)
INFO:root:UAUC users: {'n_users': 356, 'n_scored': 282, 'n_single_row': 30, 'n_single_class': 44}
INFO:root:Averaged stats: loss: 0.639689  acc: 0.646638 ***auc: 0.694805 ***uauc: 0.687478 ***prompt_tokens: 67.44 ***eval_s: 80.1 (7.70 ms/sample, n=10401) ***token_cos: 0.0234 ***hist_slots: 32.75 ***hist_unk: 0.48%
INFO:root:[valid] auc=0.694805 uauc=0.687478 prompt_tokens=67.44
INFO:root:[valid] epoch 1 uauc=0.687478 | best=0.690052 @ epoch 0 | no improvement for 1/20 validations
INFO:root:Start training
INFO:root:Start training epoch 2, 50 iters per inner epoch.
/content/QFormerRec/qformerrec/tasks/rec_qformer_task.py:91: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
/content/QFormerRec/qformerrec/models/minigpt4rec_qformer.py:206: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
/content/CoLLM/minigpt4/models/rec_model.py:48: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  return torch.cuda.amp.autocast(dtype=dtype)
INFO:root:train epoch 2 iter 0/50 loss=0.9567 lr=4.957e-06
Train: data epoch: [2]  [ 0/50]  eta: 0:00:40  lr: 0.000005  loss: 0.9567  time: 0.8070  data: 0.0000  max mem: 41426
INFO:root:train epoch 2 iter 49/50 loss=0.9184 lr=4.904e-06
Train: data epoch: [2]  [49/50]  eta: 0:00:00  lr: 0.000005  loss: 0.9184  time: 0.7783  data: 0.0000  max mem: 41426
Train: data epoch: [2] Total time: 0:00:38 (0.7790 s / it)
INFO:root:Averaged stats: lr: 0.000005  loss: 1.083429
INFO:root:[qformer-diag] epoch2
  token_cosine_offdiag=0.0317  pref_token_norm=1.2998  llm_emb_norm=1.2861  z_std_mean=0.5243  rank_pairs_per_batch=64.1200  proj_scale=0.0203  hist_slots_filled=12.2096  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.6131  loss_rank=0.5403  loss_cf=0.6248  loss_div=0.0149  loss_attn=0.9968  loss_var=0.4775
  token0: attn_by_type=[user:0.102 hist:0.225 neighbor:0.163 genre:0.250 cluster:0.260 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.087 hist:0.202 neighbor:0.158 genre:0.258 cluster:0.295 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.084 hist:0.187 neighbor:0.176 genre:0.288 cluster:0.265 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.088 hist:0.202 neighbor:0.174 genre:0.257 cluster:0.279 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
[qformer-diag] epoch2
  token_cosine_offdiag=0.0317  pref_token_norm=1.2998  llm_emb_norm=1.2861  z_std_mean=0.5243  rank_pairs_per_batch=64.1200  proj_scale=0.0203  hist_slots_filled=12.2096  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.6131  loss_rank=0.5403  loss_cf=0.6248  loss_div=0.0149  loss_attn=0.9968  loss_var=0.4775
  token0: attn_by_type=[user:0.102 hist:0.225 neighbor:0.163 genre:0.250 cluster:0.260 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.087 hist:0.202 neighbor:0.158 genre:0.258 cluster:0.295 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.084 hist:0.187 neighbor:0.176 genre:0.288 cluster:0.265 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.088 hist:0.202 neighbor:0.174 genre:0.257 cluster:0.279 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
INFO:root:Evaluating on valid.
Evaluation  [  0/163]  eta: 0:02:00  loss: 0.6626  acc: 0.6250  time: 0.7363  data: 0.1956  max mem: 41426
Evaluation  [ 32/163]  eta: 0:01:05  loss: 0.6254  acc: 0.6875  time: 0.4922  data: 0.0009  max mem: 41426
Evaluation  [ 64/163]  eta: 0:00:49  loss: 0.6655  acc: 0.5312  time: 0.4986  data: 0.0009  max mem: 41426
Evaluation  [ 96/163]  eta: 0:00:33  loss: 0.8145  acc: 0.5312  time: 0.4862  data: 0.0009  max mem: 41426
Evaluation  [128/163]  eta: 0:00:17  loss: 0.5397  acc: 0.7500  time: 0.4949  data: 0.0010  max mem: 41426
Evaluation  [160/163]  eta: 0:00:01  loss: 0.4787  acc: 0.8281  time: 0.4793  data: 0.0009  max mem: 41426
Evaluation  [162/163]  eta: 0:00:00  loss: 0.6760  acc: 0.6364  time: 0.4701  data: 0.0040  max mem: 41426
Evaluation Total time: 0:01:20 (0.4925 s / it)
INFO:root:UAUC users: {'n_users': 356, 'n_scored': 282, 'n_single_row': 30, 'n_single_class': 44}
INFO:root:Averaged stats: loss: 0.653756  acc: 0.638586 ***auc: 0.694368 ***uauc: 0.682686 ***prompt_tokens: 67.44 ***eval_s: 80.3 (7.72 ms/sample, n=10401) ***token_cos: 0.0232 ***hist_slots: 32.75 ***hist_unk: 0.48%
INFO:root:[valid] auc=0.694368 uauc=0.682686 prompt_tokens=67.44
INFO:root:[valid] epoch 2 uauc=0.682686 | best=0.690052 @ epoch 0 | no improvement for 2/20 validations
INFO:root:Start training
INFO:root:Start training epoch 3, 50 iters per inner epoch.
/content/QFormerRec/qformerrec/tasks/rec_qformer_task.py:91: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
/content/QFormerRec/qformerrec/models/minigpt4rec_qformer.py:206: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
/content/CoLLM/minigpt4/models/rec_model.py:48: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  return torch.cuda.amp.autocast(dtype=dtype)
INFO:root:train epoch 3 iter 0/50 loss=1.1876 lr=4.903e-06
Train: data epoch: [3]  [ 0/50]  eta: 0:00:41  lr: 0.000005  loss: 1.1876  time: 0.8231  data: 0.0000  max mem: 41426
INFO:root:train epoch 3 iter 49/50 loss=1.1914 lr=4.830e-06
Train: data epoch: [3]  [49/50]  eta: 0:00:00  lr: 0.000005  loss: 1.1914  time: 0.7757  data: 0.0000  max mem: 41426
Train: data epoch: [3] Total time: 0:00:39 (0.7846 s / it)
INFO:root:Averaged stats: lr: 0.000005  loss: 1.036642
INFO:root:[qformer-diag] epoch3
  token_cosine_offdiag=0.0430  pref_token_norm=1.3015  llm_emb_norm=1.2861  z_std_mean=0.5407  rank_pairs_per_batch=62.9400  proj_scale=0.0204  hist_slots_filled=12.3571  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.5912  loss_rank=0.4952  loss_cf=0.6169  loss_div=0.0158  loss_attn=0.9969  loss_var=0.4618
  token0: attn_by_type=[user:0.103 hist:0.218 neighbor:0.169 genre:0.248 cluster:0.263 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.087 hist:0.197 neighbor:0.163 genre:0.257 cluster:0.296 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.085 hist:0.183 neighbor:0.183 genre:0.286 cluster:0.264 hist_unk:0.000]  top3=['genre', 'cluster', 'neighbor']  type_bias_softmax=[0.167 0.167 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.089 hist:0.197 neighbor:0.180 genre:0.256 cluster:0.278 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
[qformer-diag] epoch3
  token_cosine_offdiag=0.0430  pref_token_norm=1.3015  llm_emb_norm=1.2861  z_std_mean=0.5407  rank_pairs_per_batch=62.9400  proj_scale=0.0204  hist_slots_filled=12.3571  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.5912  loss_rank=0.4952  loss_cf=0.6169  loss_div=0.0158  loss_attn=0.9969  loss_var=0.4618
  token0: attn_by_type=[user:0.103 hist:0.218 neighbor:0.169 genre:0.248 cluster:0.263 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.087 hist:0.197 neighbor:0.163 genre:0.257 cluster:0.296 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.085 hist:0.183 neighbor:0.183 genre:0.286 cluster:0.264 hist_unk:0.000]  top3=['genre', 'cluster', 'neighbor']  type_bias_softmax=[0.167 0.167 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.089 hist:0.197 neighbor:0.180 genre:0.256 cluster:0.278 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
INFO:root:Evaluating on valid.
Evaluation  [  0/163]  eta: 0:02:03  loss: 0.6579  acc: 0.6406  time: 0.7573  data: 0.2182  max mem: 41426
Evaluation  [ 32/163]  eta: 0:01:05  loss: 0.5746  acc: 0.7031  time: 0.4916  data: 0.0009  max mem: 41426
Evaluation  [ 64/163]  eta: 0:00:49  loss: 0.8274  acc: 0.4844  time: 0.4953  data: 0.0009  max mem: 41426
Evaluation  [ 96/163]  eta: 0:00:33  loss: 0.7877  acc: 0.5625  time: 0.4857  data: 0.0009  max mem: 41426
Evaluation  [128/163]  eta: 0:00:17  loss: 0.5064  acc: 0.7656  time: 0.4928  data: 0.0009  max mem: 41426
Evaluation  [160/163]  eta: 0:00:01  loss: 0.4072  acc: 0.8906  time: 0.4779  data: 0.0008  max mem: 41426
Evaluation  [162/163]  eta: 0:00:00  loss: 0.6420  acc: 0.6364  time: 0.4698  data: 0.0043  max mem: 41426
Evaluation Total time: 0:01:20 (0.4913 s / it)
INFO:root:UAUC users: {'n_users': 356, 'n_scored': 282, 'n_single_row': 30, 'n_single_class': 44}
INFO:root:Averaged stats: loss: 0.649724  acc: 0.649705 ***auc: 0.697132 ***uauc: 0.686836 ***prompt_tokens: 67.44 ***eval_s: 80.1 (7.70 ms/sample, n=10401) ***token_cos: 0.0350 ***hist_slots: 32.75 ***hist_unk: 0.48%
INFO:root:[valid] auc=0.697132 uauc=0.686836 prompt_tokens=67.44
INFO:root:[valid] epoch 3 uauc=0.686836 | best=0.690052 @ epoch 0 | no improvement for 3/20 validations
INFO:root:Start training
INFO:root:Start training epoch 4, 50 iters per inner epoch.
/content/QFormerRec/qformerrec/tasks/rec_qformer_task.py:91: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
/content/QFormerRec/qformerrec/models/minigpt4rec_qformer.py:206: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
/content/CoLLM/minigpt4/models/rec_model.py:48: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  return torch.cuda.amp.autocast(dtype=dtype)
INFO:root:[qformer-diag]
  token_cosine_offdiag=0.0660  pref_token_norm=1.3039  llm_emb_norm=1.2861  z_std_mean=0.5519  rank_pairs_per_batch=67.0000  proj_scale=0.0204  hist_slots_filled=10.2292  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.6904  loss_rank=0.6472  loss_cf=0.6974  loss_div=0.0170  loss_attn=0.9970  loss_var=0.4528
  token0: attn_by_type=[user:0.101 hist:0.197 neighbor:0.168 genre:0.264 cluster:0.270 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.085 hist:0.177 neighbor:0.165 genre:0.271 cluster:0.302 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.083 hist:0.164 neighbor:0.187 genre:0.303 cluster:0.263 hist_unk:0.000]  top3=['genre', 'cluster', 'neighbor']  type_bias_softmax=[0.167 0.167 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.089 hist:0.179 neighbor:0.184 genre:0.264 cluster:0.284 hist_unk:0.000]  top3=['cluster', 'genre', 'neighbor']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
[qformer-diag]
  token_cosine_offdiag=0.0660  pref_token_norm=1.3039  llm_emb_norm=1.2861  z_std_mean=0.5519  rank_pairs_per_batch=67.0000  proj_scale=0.0204  hist_slots_filled=10.2292  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.6904  loss_rank=0.6472  loss_cf=0.6974  loss_div=0.0170  loss_attn=0.9970  loss_var=0.4528
  token0: attn_by_type=[user:0.101 hist:0.197 neighbor:0.168 genre:0.264 cluster:0.270 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.085 hist:0.177 neighbor:0.165 genre:0.271 cluster:0.302 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.083 hist:0.164 neighbor:0.187 genre:0.303 cluster:0.263 hist_unk:0.000]  top3=['genre', 'cluster', 'neighbor']  type_bias_softmax=[0.167 0.167 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.089 hist:0.179 neighbor:0.184 genre:0.264 cluster:0.284 hist_unk:0.000]  top3=['cluster', 'genre', 'neighbor']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
INFO:root:train epoch 4 iter 0/50 loss=1.2276 lr=4.829e-06
Train: data epoch: [4]  [ 0/50]  eta: 0:00:38  lr: 0.000005  loss: 1.2276  time: 0.7783  data: 0.0000  max mem: 41426
INFO:root:train epoch 4 iter 49/50 loss=1.0062 lr=4.736e-06
Train: data epoch: [4]  [49/50]  eta: 0:00:00  lr: 0.000005  loss: 1.0062  time: 0.7687  data: 0.0000  max mem: 41426
Train: data epoch: [4] Total time: 0:00:38 (0.7717 s / it)
INFO:root:Averaged stats: lr: 0.000005  loss: 0.980795
INFO:root:[qformer-diag] epoch4
  token_cosine_offdiag=0.0425  pref_token_norm=1.3041  llm_emb_norm=1.2861  z_std_mean=0.5456  rank_pairs_per_batch=64.2857  proj_scale=0.0204  hist_slots_filled=12.6888  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.5621  loss_rank=0.4387  loss_cf=0.6002  loss_div=0.0156  loss_attn=0.9969  loss_var=0.4567
  token0: attn_by_type=[user:0.102 hist:0.229 neighbor:0.161 genre:0.245 cluster:0.262 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.088 hist:0.207 neighbor:0.155 genre:0.254 cluster:0.296 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.086 hist:0.191 neighbor:0.173 genre:0.285 cluster:0.265 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.089 hist:0.206 neighbor:0.172 genre:0.256 cluster:0.278 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
[qformer-diag] epoch4
  token_cosine_offdiag=0.0425  pref_token_norm=1.3041  llm_emb_norm=1.2861  z_std_mean=0.5456  rank_pairs_per_batch=64.2857  proj_scale=0.0204  hist_slots_filled=12.6888  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.5621  loss_rank=0.4387  loss_cf=0.6002  loss_div=0.0156  loss_attn=0.9969  loss_var=0.4567
  token0: attn_by_type=[user:0.102 hist:0.229 neighbor:0.161 genre:0.245 cluster:0.262 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.088 hist:0.207 neighbor:0.155 genre:0.254 cluster:0.296 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.086 hist:0.191 neighbor:0.173 genre:0.285 cluster:0.265 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.089 hist:0.206 neighbor:0.172 genre:0.256 cluster:0.278 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
INFO:root:Evaluating on valid.
Evaluation  [  0/163]  eta: 0:02:03  loss: 0.6568  acc: 0.6875  time: 0.7554  data: 0.2149  max mem: 41426
Evaluation  [ 32/163]  eta: 0:01:05  loss: 0.5772  acc: 0.6875  time: 0.4907  data: 0.0009  max mem: 41426
Evaluation  [ 64/163]  eta: 0:00:49  loss: 0.8387  acc: 0.4844  time: 0.4933  data: 0.0009  max mem: 41426
Evaluation  [ 96/163]  eta: 0:00:32  loss: 0.7729  acc: 0.5469  time: 0.4852  data: 0.0009  max mem: 41426
Evaluation  [128/163]  eta: 0:00:17  loss: 0.5079  acc: 0.7500  time: 0.4946  data: 0.0009  max mem: 41426
Evaluation  [160/163]  eta: 0:00:01  loss: 0.4047  acc: 0.8906  time: 0.4786  data: 0.0009  max mem: 41426
Evaluation  [162/163]  eta: 0:00:00  loss: 0.6121  acc: 0.6667  time: 0.4697  data: 0.0040  max mem: 41426
Evaluation Total time: 0:01:20 (0.4910 s / it)
INFO:root:UAUC users: {'n_users': 356, 'n_scored': 282, 'n_single_row': 30, 'n_single_class': 44}
INFO:root:Averaged stats: loss: 0.650075  acc: 0.649604 ***auc: 0.696124 ***uauc: 0.687035 ***prompt_tokens: 67.44 ***eval_s: 80.0 (7.70 ms/sample, n=10401) ***token_cos: 0.0428 ***hist_slots: 32.75 ***hist_unk: 0.48%
INFO:root:[valid] auc=0.696124 uauc=0.687035 prompt_tokens=67.44
INFO:root:[valid] epoch 4 uauc=0.687035 | best=0.690052 @ epoch 0 | no improvement for 4/20 validations
INFO:root:Start training
INFO:root:Start training epoch 5, 50 iters per inner epoch.
/content/QFormerRec/qformerrec/tasks/rec_qformer_task.py:91: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
/content/QFormerRec/qformerrec/models/minigpt4rec_qformer.py:206: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
/content/CoLLM/minigpt4/models/rec_model.py:48: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  return torch.cuda.amp.autocast(dtype=dtype)
INFO:root:train epoch 5 iter 0/50 loss=1.0649 lr=4.734e-06
Train: data epoch: [5]  [ 0/50]  eta: 0:00:38  lr: 0.000005  loss: 1.0649  time: 0.7668  data: 0.0000  max mem: 41426
INFO:root:train epoch 5 iter 49/50 loss=0.8207 lr=4.623e-06
Train: data epoch: [5]  [49/50]  eta: 0:00:00  lr: 0.000005  loss: 0.8207  time: 0.7734  data: 0.0000  max mem: 41426
Train: data epoch: [5] Total time: 0:00:39 (0.7822 s / it)
INFO:root:Averaged stats: lr: 0.000005  loss: 1.060880
INFO:root:[qformer-diag] epoch5
  token_cosine_offdiag=0.0428  pref_token_norm=1.3035  llm_emb_norm=1.2861  z_std_mean=0.5473  rank_pairs_per_batch=64.9600  proj_scale=0.0204  hist_slots_filled=13.0442  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.6037  loss_rank=0.5229  loss_cf=0.6086  loss_div=0.0148  loss_attn=0.9968  loss_var=0.4552
  token0: attn_by_type=[user:0.103 hist:0.231 neighbor:0.161 genre:0.242 cluster:0.262 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.089 hist:0.208 neighbor:0.154 genre:0.252 cluster:0.297 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.085 hist:0.191 neighbor:0.172 genre:0.284 cluster:0.267 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.166 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.090 hist:0.206 neighbor:0.170 genre:0.254 cluster:0.280 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
[qformer-diag] epoch5
  token_cosine_offdiag=0.0428  pref_token_norm=1.3035  llm_emb_norm=1.2861  z_std_mean=0.5473  rank_pairs_per_batch=64.9600  proj_scale=0.0204  hist_slots_filled=13.0442  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.6037  loss_rank=0.5229  loss_cf=0.6086  loss_div=0.0148  loss_attn=0.9968  loss_var=0.4552
  token0: attn_by_type=[user:0.103 hist:0.231 neighbor:0.161 genre:0.242 cluster:0.262 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.089 hist:0.208 neighbor:0.154 genre:0.252 cluster:0.297 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.085 hist:0.191 neighbor:0.172 genre:0.284 cluster:0.267 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.166 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.090 hist:0.206 neighbor:0.170 genre:0.254 cluster:0.280 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
INFO:root:Evaluating on valid.
Evaluation  [  0/163]  eta: 0:02:00  loss: 0.6388  acc: 0.6250  time: 0.7395  data: 0.1998  max mem: 41426
Evaluation  [ 32/163]  eta: 0:01:05  loss: 0.5718  acc: 0.7031  time: 0.4918  data: 0.0009  max mem: 41426
Evaluation  [ 64/163]  eta: 0:00:49  loss: 0.8720  acc: 0.4219  time: 0.4946  data: 0.0009  max mem: 41426
Evaluation  [ 96/163]  eta: 0:00:33  loss: 0.7224  acc: 0.5781  time: 0.4856  data: 0.0009  max mem: 41426
Evaluation  [128/163]  eta: 0:00:17  loss: 0.4978  acc: 0.7656  time: 0.4924  data: 0.0009  max mem: 41426
Evaluation  [160/163]  eta: 0:00:01  loss: 0.4142  acc: 0.8906  time: 0.4779  data: 0.0008  max mem: 41426
Evaluation  [162/163]  eta: 0:00:00  loss: 0.5861  acc: 0.6667  time: 0.4689  data: 0.0038  max mem: 41426
Evaluation Total time: 0:01:20 (0.4909 s / it)
INFO:root:UAUC users: {'n_users': 356, 'n_scored': 282, 'n_single_row': 30, 'n_single_class': 44}
INFO:root:Averaged stats: loss: 0.638152  acc: 0.647495 ***auc: 0.695943 ***uauc: 0.683458 ***prompt_tokens: 67.44 ***eval_s: 80.0 (7.69 ms/sample, n=10401) ***token_cos: 0.0414 ***hist_slots: 32.75 ***hist_unk: 0.48%
INFO:root:[valid] auc=0.695943 uauc=0.683458 prompt_tokens=67.44
INFO:root:[valid] epoch 5 uauc=0.683458 | best=0.690052 @ epoch 0 | no improvement for 5/20 validations
INFO:root:Start training
INFO:root:Start training epoch 6, 50 iters per inner epoch.
/content/QFormerRec/qformerrec/tasks/rec_qformer_task.py:91: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
/content/QFormerRec/qformerrec/models/minigpt4rec_qformer.py:206: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
/content/CoLLM/minigpt4/models/rec_model.py:48: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  return torch.cuda.amp.autocast(dtype=dtype)
INFO:root:train epoch 6 iter 0/50 loss=1.0635 lr=4.621e-06
Train: data epoch: [6]  [ 0/50]  eta: 0:00:38  lr: 0.000005  loss: 1.0635  time: 0.7725  data: 0.0000  max mem: 41426
INFO:root:train epoch 6 iter 49/50 loss=1.1235 lr=4.492e-06
Train: data epoch: [6]  [49/50]  eta: 0:00:00  lr: 0.000004  loss: 1.1235  time: 0.7902  data: 0.0000  max mem: 41426
Train: data epoch: [6] Total time: 0:00:39 (0.7815 s / it)
INFO:root:Averaged stats: lr: 0.000005  loss: 1.056425
INFO:root:[qformer-diag] epoch6
  token_cosine_offdiag=0.0448  pref_token_norm=1.3045  llm_emb_norm=1.2861  z_std_mean=0.5560  rank_pairs_per_batch=64.0400  proj_scale=0.0204  hist_slots_filled=14.1538  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.6004  loss_rank=0.5224  loss_cf=0.6056  loss_div=0.0151  loss_attn=0.9968  loss_var=0.4468
  token0: attn_by_type=[user:0.101 hist:0.235 neighbor:0.164 genre:0.244 cluster:0.257 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.087 hist:0.212 neighbor:0.156 genre:0.254 cluster:0.291 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.085 hist:0.193 neighbor:0.174 genre:0.286 cluster:0.262 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.166 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.088 hist:0.210 neighbor:0.173 genre:0.254 cluster:0.275 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
[qformer-diag] epoch6
  token_cosine_offdiag=0.0448  pref_token_norm=1.3045  llm_emb_norm=1.2861  z_std_mean=0.5560  rank_pairs_per_batch=64.0400  proj_scale=0.0204  hist_slots_filled=14.1538  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.6004  loss_rank=0.5224  loss_cf=0.6056  loss_div=0.0151  loss_attn=0.9968  loss_var=0.4468
  token0: attn_by_type=[user:0.101 hist:0.235 neighbor:0.164 genre:0.244 cluster:0.257 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.087 hist:0.212 neighbor:0.156 genre:0.254 cluster:0.291 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.085 hist:0.193 neighbor:0.174 genre:0.286 cluster:0.262 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.166 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.088 hist:0.210 neighbor:0.173 genre:0.254 cluster:0.275 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
INFO:root:Evaluating on valid.
Evaluation  [  0/163]  eta: 0:02:03  loss: 0.6334  acc: 0.6406  time: 0.7572  data: 0.2113  max mem: 41426
Evaluation  [ 32/163]  eta: 0:01:05  loss: 0.5696  acc: 0.7031  time: 0.4911  data: 0.0009  max mem: 41426
Evaluation  [ 64/163]  eta: 0:00:49  loss: 0.8285  acc: 0.4531  time: 0.4939  data: 0.0009  max mem: 41426
Evaluation  [ 96/163]  eta: 0:00:33  loss: 0.7091  acc: 0.5469  time: 0.4858  data: 0.0009  max mem: 41426
Evaluation  [128/163]  eta: 0:00:17  loss: 0.4937  acc: 0.7656  time: 0.4930  data: 0.0009  max mem: 41426
Evaluation  [160/163]  eta: 0:00:01  loss: 0.4350  acc: 0.8750  time: 0.4786  data: 0.0009  max mem: 41426
Evaluation  [162/163]  eta: 0:00:00  loss: 0.5921  acc: 0.6970  time: 0.4703  data: 0.0045  max mem: 41426
Evaluation Total time: 0:01:20 (0.4914 s / it)
INFO:root:UAUC users: {'n_users': 356, 'n_scored': 282, 'n_single_row': 30, 'n_single_class': 44}
INFO:root:Averaged stats: loss: 0.634179  acc: 0.646914 ***auc: 0.696297 ***uauc: 0.678472 ***prompt_tokens: 67.44 ***eval_s: 80.1 (7.70 ms/sample, n=10401) ***token_cos: 0.0427 ***hist_slots: 32.75 ***hist_unk: 0.48%
INFO:root:[valid] auc=0.696297 uauc=0.678472 prompt_tokens=67.44
INFO:root:[valid] epoch 6 uauc=0.678472 | best=0.690052 @ epoch 0 | no improvement for 6/20 validations
INFO:root:Start training
INFO:root:Start training epoch 7, 50 iters per inner epoch.
/content/QFormerRec/qformerrec/tasks/rec_qformer_task.py:91: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
/content/QFormerRec/qformerrec/models/minigpt4rec_qformer.py:206: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
/content/CoLLM/minigpt4/models/rec_model.py:48: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  return torch.cuda.amp.autocast(dtype=dtype)
INFO:root:train epoch 7 iter 0/50 loss=1.1103 lr=4.489e-06
Train: data epoch: [7]  [ 0/50]  eta: 0:00:38  lr: 0.000004  loss: 1.1103  time: 0.7699  data: 0.0000  max mem: 41426
INFO:root:train epoch 7 iter 49/50 loss=0.9208 lr=4.344e-06
Train: data epoch: [7]  [49/50]  eta: 0:00:00  lr: 0.000004  loss: 0.9208  time: 0.7960  data: 0.0000  max mem: 41426
Train: data epoch: [7] Total time: 0:00:39 (0.7831 s / it)
INFO:root:Averaged stats: lr: 0.000004  loss: 1.039456
INFO:root:[qformer-diag] epoch7
  token_cosine_offdiag=0.0446  pref_token_norm=1.3036  llm_emb_norm=1.2861  z_std_mean=0.5573  rank_pairs_per_batch=64.2000  proj_scale=0.0204  hist_slots_filled=12.6042  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.5952  loss_rank=0.4965  loss_cf=0.6123  loss_div=0.0151  loss_attn=0.9967  loss_var=0.4454
  token0: attn_by_type=[user:0.104 hist:0.225 neighbor:0.165 genre:0.242 cluster:0.263 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.089 hist:0.203 neighbor:0.156 genre:0.255 cluster:0.298 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.087 hist:0.185 neighbor:0.176 genre:0.284 cluster:0.268 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.166 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.090 hist:0.201 neighbor:0.174 genre:0.254 cluster:0.282 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
[qformer-diag] epoch7
  token_cosine_offdiag=0.0446  pref_token_norm=1.3036  llm_emb_norm=1.2861  z_std_mean=0.5573  rank_pairs_per_batch=64.2000  proj_scale=0.0204  hist_slots_filled=12.6042  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.5952  loss_rank=0.4965  loss_cf=0.6123  loss_div=0.0151  loss_attn=0.9967  loss_var=0.4454
  token0: attn_by_type=[user:0.104 hist:0.225 neighbor:0.165 genre:0.242 cluster:0.263 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.089 hist:0.203 neighbor:0.156 genre:0.255 cluster:0.298 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.087 hist:0.185 neighbor:0.176 genre:0.284 cluster:0.268 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.166 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.090 hist:0.201 neighbor:0.174 genre:0.254 cluster:0.282 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
INFO:root:Evaluating on valid.
Evaluation  [  0/163]  eta: 0:02:02  loss: 0.6817  acc: 0.6250  time: 0.7508  data: 0.2121  max mem: 41426
Evaluation  [ 32/163]  eta: 0:01:05  loss: 0.6439  acc: 0.7031  time: 0.4912  data: 0.0010  max mem: 41426
Evaluation  [ 64/163]  eta: 0:00:49  loss: 0.6879  acc: 0.5469  time: 0.4936  data: 0.0009  max mem: 41426
Evaluation  [ 96/163]  eta: 0:00:33  loss: 0.8366  acc: 0.5156  time: 0.4855  data: 0.0009  max mem: 41426
Evaluation  [128/163]  eta: 0:00:17  loss: 0.5179  acc: 0.7344  time: 0.4924  data: 0.0009  max mem: 41426
Evaluation  [160/163]  eta: 0:00:01  loss: 0.4444  acc: 0.8281  time: 0.4779  data: 0.0009  max mem: 41426
Evaluation  [162/163]  eta: 0:00:00  loss: 0.6705  acc: 0.6364  time: 0.4687  data: 0.0036  max mem: 41426
Evaluation Total time: 0:01:19 (0.4906 s / it)
INFO:root:UAUC users: {'n_users': 356, 'n_scored': 282, 'n_single_row': 30, 'n_single_class': 44}
INFO:root:Averaged stats: loss: 0.666035  acc: 0.638969 ***auc: 0.697231 ***uauc: 0.683579 ***prompt_tokens: 67.44 ***eval_s: 80.0 (7.69 ms/sample, n=10401) ***token_cos: 0.0358 ***hist_slots: 32.75 ***hist_unk: 0.48%
INFO:root:[valid] auc=0.697231 uauc=0.683579 prompt_tokens=67.44
INFO:root:[valid] epoch 7 uauc=0.683579 | best=0.690052 @ epoch 0 | no improvement for 7/20 validations
INFO:root:Start training
INFO:root:Start training epoch 8, 50 iters per inner epoch.
/content/QFormerRec/qformerrec/tasks/rec_qformer_task.py:91: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=use_amp):
/content/QFormerRec/qformerrec/models/minigpt4rec_qformer.py:206: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
/content/CoLLM/minigpt4/models/rec_model.py:48: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  return torch.cuda.amp.autocast(dtype=dtype)
INFO:root:[qformer-diag]
  token_cosine_offdiag=0.0356  pref_token_norm=1.3048  llm_emb_norm=1.2861  z_std_mean=0.5641  rank_pairs_per_batch=59.0000  proj_scale=0.0204  hist_slots_filled=14.6667  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.6513  loss_rank=0.4853  loss_cf=0.6511  loss_div=0.0130  loss_attn=0.9966  loss_var=0.4388
  token0: attn_by_type=[user:0.108 hist:0.226 neighbor:0.153 genre:0.251 cluster:0.261 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.090 hist:0.206 neighbor:0.144 genre:0.265 cluster:0.295 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.085 hist:0.188 neighbor:0.160 genre:0.294 cluster:0.273 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.166 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.089 hist:0.203 neighbor:0.159 genre:0.265 cluster:0.284 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
[qformer-diag]
  token_cosine_offdiag=0.0356  pref_token_norm=1.3048  llm_emb_norm=1.2861  z_std_mean=0.5641  rank_pairs_per_batch=59.0000  proj_scale=0.0204  hist_slots_filled=14.6667  hist_unk_rate=0.0000  history_source=pit  k_hist=50  loss_bce=0.6513  loss_rank=0.4853  loss_cf=0.6511  loss_div=0.0130  loss_attn=0.9966  loss_var=0.4388
  token0: attn_by_type=[user:0.108 hist:0.226 neighbor:0.153 genre:0.251 cluster:0.261 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token1: attn_by_type=[user:0.090 hist:0.206 neighbor:0.144 genre:0.265 cluster:0.295 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.166 0.166 0.167 0.167]
  token2: attn_by_type=[user:0.085 hist:0.188 neighbor:0.160 genre:0.294 cluster:0.273 hist_unk:0.000]  top3=['genre', 'cluster', 'hist']  type_bias_softmax=[0.167 0.166 0.167 0.167 0.166 0.167]
  token3: attn_by_type=[user:0.089 hist:0.203 neighbor:0.159 genre:0.265 cluster:0.284 hist_unk:0.000]  top3=['cluster', 'genre', 'hist']  type_bias_softmax=[0.167 0.167 0.167 0.166 0.167 0.167]
INFO:root:train epoch 8 iter 0/50 loss=1.0972 lr=4.341e-06
Train: data epoch: [8]  [ 0/50]  eta: 0:00:38  lr: 0.000004  loss: 1.0972  time: 0.7705  data: 0.0000  max mem: 41426
