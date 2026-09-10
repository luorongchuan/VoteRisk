# HARDWARE_ADAPTATION.md

EDAS 官方训练脚本是为 8B / 32B 模型 + 8× / 16× GPU 集群设计的。本实验在 **2 × A100 80GB** 上做 Qwen3-4B 训练 / 评测,需要做的硬件适配如下。**所有适配 EDAS / VoteRisk 完全一致,不留方法差异。**

## 1. GPU 资源

| 项 | 官方 (8B / 8 卡) | 本实验 (4B / 2 卡) |
|---|---|---|
| `n_gpus_per_node` | 8 | **2** |
| `nnodes` | 1 | 1 |
| 每张卡显存 | H800 80GB | A100-SXM4 80GB |

→ **保持 effective batch size / rollout G / LR / 训练步数一致**,显存层面的差异通过下面的 micro-batch / FSDP offload / vLLM memory 调整来消化。

## 2. Batch / Micro-batch

| 项 | EDAS 8B (8 卡) | 本实验 (2 卡) | 一致性结论 |
|---|---|---|---|
| `data.train_batch_size` | 256 prompts | **256 prompts** | ✅ 完全一致 (effective batch) |
| `data.gen_batch_size` | 512 | **512** | ✅ 完全一致 |
| `actor.ppo_mini_batch_size` | 64 | **64** | ✅ 完全一致 |
| `actor.ppo_micro_batch_size_per_gpu` | 1 | **1** | ✅ 完全一致 (4B 模型 2 卡用 micro=1 已经够用) |
| `actor.use_dynamic_bsz` | True | **True** | ✅ 一致 |
| `actor.ulysses_sequence_parallel_size` | 1 | **1** | ✅ 一致 |

我们没有降低 train_batch_size,所以 **gradient accumulation step = ppo_mini_batch / (micro * n_gpu) = 64 / 2 = 32**,与 EDAS 8B 一致。

> 备注:如果 4B 模型在 `train_batch_size=256 / ppo_mini_batch=64` 上出现 OOM(尚未出现),优先减 micro_batch,然后减 ppo_mini_batch;**两个方法必须同步**。

## 3. Rollout / vLLM

| 项 | EDAS 8B (8 卡) | 本实验 (2 卡) | 一致性结论 |
|---|---|---|---|
| `rollout.n` | 10 | **10** | ✅ 一致 |
| `rollout.gpu_memory_utilization` | 0.8 (32B 卡) | **0.4** | override (colocated 模式下 FSDP actor+ref 已占 ~51 GiB/卡,vLLM 必须 ≤ 28 GiB → 0.4×79.25=31.7 GiB) |
| `rollout.tensor_model_parallel_size` | 2 | **1** | override (4B 单卡就能放;且 tp=1 与 8B tp=2 对 actor GPU 占位不同 — 但 rollout 行为本身不变) |
| `rollout.max_num_batched_tokens` | 32768 | **32768** | ✅ 一致 |
| `rollout.temperature` / `top_p` | 1.0 / 1.0 | **1.0 / 1.0** | ✅ 一致 |
| `rollout.max_response_length` | 8192 | **8192** | ✅ 一致 |
| `rollout.enable_dynamic_max_tokens` | True | **True** | ✅ 一致 |
| `rollout.enable_chunked_prefill` | True | **True** | ✅ 一致 |

> `tensor_parallel_size` 在 vLLM 内部只影响 KV/weight sharding,不改变 sampling 行为,所以不影响 Maj@K / Pass@K 的可比性。**两个方法都用 tp=1。**

## 4. FSDP / 显存策略

| 项 | EDAS 8B | 本实验 |
|---|---|---|
| `actor.fsdp_config.param_offload` | True | **True** |
| `actor.fsdp_config.optimizer_offload` | True | **True** |
| `ref.fsdp_config.param_offload` | True | **True** |
| `model.enable_gradient_checkpointing` | True | **True** |

✅ 一致。

## 5. 训练步数 / Epochs

| 项 | EDAS 8B | 本实验 |
|---|---|---|
| `trainer.total_epochs` | 2 (DAPO 8B 复刻默认值) | **2** |
| `trainer.save_freq` | 10 | **10** |
| `trainer.test_freq` | 5 | **5** |
| `trainer.max_actor_ckpt_to_keep` | 20 | **20** |

✅ 一致。**Final checkpoint = global_step_2×train_steps_per_epoch** 下的 `actor/`。两方法同时取 final step 评测。

## 6. Reward / Verifier

| 项 | EDAS 8B | 本实验 |
|---|---|---|
| `reward.custom_reward_function.path` | `examples/branch_adv/reward_function/math_no_format.py` | **同上** |
| `reward.custom_reward_function.name` | `compute_score_batch` | **同上** |
| `reward.reward_manager.name` | `batch` | **同上** |
| `enable_llm_judge` | True (连 10.119.97.103:9000-9007) | **False** (本地无 judge cluster,只保留 rule + sympy) |
| `penalty_max_length` | 50000 | **50000** |
| `overlong_buffer_length` | 0 | **0** |

⚠️ **改动说明**: LLM judge 关闭对 EDAS 和 VoteRisk 是同样的对称改动,不影响相对比较。训练 reward 仍由 rule + sympy 等价判断,功能上等价于"judge 在第一步就给出相同 answer"。

## 7. Smoke-test 设置

当 `bash run_train.sh <mode> --smoke` 时:

| 项 | 值 |
|---|---|
| `total_epochs` | 2(快速验证) |
| `save_freq` / `test_freq` / `save_limit` | 1 / 1 / 2 |

两个方法的 smoke test 同样设置,只用来验证 reward / KL / gradient / VoteRisk 数值是否稳定。

## 8. 评测硬件策略

| 项 | EDAS 训练后评测 | VoteRisk 训练后评测 |
|---|---|---|
| GPU | 与训练同卡(避免 warm-up / cache miss) | 同 |
| vLLM tp_size | 1 | 1 |
| vLLM gpu_memory_utilization | 0.85 | 0.85 |
| max_model_len | 12288 (8192 resp + 4096 prompt headroom) | 同 |
| dtype | bfloat16 | 同 |

✅ 两个方法完全相同的 vLLM 配置。

## 9. 不动的"硬件决定"

为了不破坏算法公平性,**以下项目 EDAS 和 VoteRisk 一律相同**(即使在 2 卡上更激进也保留 EDAS 8B 设置):

- `data.train_batch_size` = 256 prompts
- `actor.ppo_mini_batch_size` = 64
- `rollout.n` = 10
- `trainer.total_epochs` = 2
- `optim.lr` = 1e-6
- `clip_ratio_low/high` = 0.2 / 0.28
- 训练数据 / 顺序 / 数据 shuffle seed
- 评测 N=64 / temperature / top_p / seed=42

## 10. 顺序运行策略(避免一张卡跑一个模型并行浪费)

`run_sequential.sh` 在同一组 GPU 上依次跑:

```
[base eval (optional)]
→ train EDAS (2 GPU) → eval EDAS (1-2 GPU)
→ train VoteRisk (2 GPU) → eval VoteRisk (1-2 GPU)
→ build_final_report.py
```

每个阶段都有 `_DONE` 标记,可断点续跑;`SEQ_METHODS=edas` 可跳过 VoteRisk。