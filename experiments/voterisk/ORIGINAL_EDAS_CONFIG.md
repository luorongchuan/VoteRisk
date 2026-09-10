# ORIGINAL_EDAS_CONFIG.md

EDAS 官方 Qwen3-4B / Qwen3-8B / DAPO 训练配置整理。所有 EDAS 与 VoteRisk 实验 **必须** 在这些设置之上完全相同地运行。

## 1. 引用来源 (Source of truth)

| 项 | 路径 |
|---|---|
| EDAS 官方 README | `EDAS/README.md` |
| DAPO + EDAS launcher | `EDAS/examples/branch_adv/run_dapo_branch_adv.sh` |
| EDAS 8B DAPO+branch_adv 训练脚本 | `EDAS/examples/branch_adv/run_8b_non_thinking_04_30_20_wo_format_17k.sh` |
| 数据预处理 | `EDAS/examples/branch_adv/preprocess_parquet.py` |
| Reward function | `EDAS/examples/branch_adv/reward_function/math_no_format.py` |
| DAPO trainer | `EDAS/recipe/dapo/dapo_ray_trainer.py` |
| Advantage 算法 | `EDAS/verl/trainer/ppo/core_algos.py` |
| 官方 base 配置 | `EDAS/recipe/dapo/config/dapo_trainer.yaml` |

EDAS 仓库 **没有** 单独的 Qwen3-4B DAPO 训练脚本 (`run_dapo_qwen3_*` 系列只有 8B / 14B / 30B 等)。我们以 **8B DAPO+EDAS 脚本作为模板**(同样基于 DAPO-Math-17K,enable_thinking=False),把模型从 8B 切到 4B;算法本身 (rollout G、clip、KL、optimizer、prompt template、reward、verifier) **完全沿用 8B 设置**。

## 2. 模型 / 数据 / 路径

| 项 | 值 |
|---|---|
| Base 模型 | `/home/luorongchuan/workspace_135/models/Qwen3-4B-Base` |
| 训练 parquet | `/home/luorongchuan/workspace_135/datasets/dapo-math-17k.formatted.parquet` (由 `prepare_data.py` 从 `dapo-math-17k.parquet` 生成,只注入 `extra_info.split='train'`) |
| Test 数据 | `MATH-500 / AMC23 / AIME24 / AIME25`,来自 `datasets/test_data/` |
| Reward function | `examples/branch_adv/reward_function/math_no_format.py::compute_score_batch` |
| LLM judge | **关闭**(`enable_llm_judge=False`),本地无 judge cluster;只保留 rule + sympy 等价 |
| Prompt template | 已嵌入 chat-format parquet(用户问题直接送 tokenizer) |
| `apply_chat_template_kwargs.enable_thinking` | `False`(Qwen3 non-thinking) |

## 3. 算法超参数 (Algorithm) — 与 EDAS 8B 完全一致

| 名称 | 值 | 来源 |
|---|---|---|
| Advantage estimator | `grpo` (recipe/dapo + RayDAPOTrainer) | run_dapo_branch_adv.sh:321 |
| `algorithm.use_kl_in_reward` | `False` | DISABLE_KL=True |
| `actor.use_kl_loss` | `False` | DISABLE_KL=True |
| `kl_loss_coef` / `kl_ctrl.kl_coef` | `0.0` | DISABLE_KL=True |
| `kl_penalty` / `kl_loss_type` | `low_var_kl` | run_dapo_branch_adv.sh:324,362 |
| `clip_ratio_low` | `0.2` | EDAS 8B 脚本 |
| `clip_ratio_high` | `0.28` | EDAS 8B 脚本 |
| `grad_clip` | `1.0` | EDAS 8B 脚本 |
| `ppo_mini_batch_size` | `64` | EDAS 8B 脚本 |
| `ppo_micro_batch_size_per_gpu` | `1` | EDAS 8B 脚本 |
| `use_dynamic_bsz` | `True` | EDAS 8B 脚本 |
| `ulysses_sequence_parallel_size` | `1` | EDAS 8B 脚本 |
| `fsdp_config.param_offload` | `True` | EDAS 8B 脚本 |
| `fsdp_config.optimizer_offload` | `True` | EDAS 8B 脚本 |
| `filter_groups.enable` | `True` | EDAS 8B 脚本 |
| `filter_groups.metric` | `accuracy` | EDAS 8B 脚本 |
| `filter_groups.max_num_gen_batches` | `20` | EDAS 8B 脚本 |
| `optim.lr` | `1e-6` | EDAS 8B 脚本 |
| `optim.weight_decay` | `1e-2` | EDAS 8B 脚本 |
| `norm_adv_by_std_in_grpo` | `True` | EDAS 8B 脚本 |

## 4. 数据 / Batch / Rollout

| 名称 | 值 | 来源 |
|---|---|---|
| `data.train_files` | `datasets/dapo-math-17k.formatted.parquet` | 见 §2 |
| `data.val_files` | 同上(便于 sanity check) |  |
| `data.filter_overlong_prompts` | `True` | EDAS 8B 脚本 |
| `data.truncation` | `error` | EDAS 8B 脚本 |
| `data.shuffle` | `True` | EDAS 8B 脚本 |
| `data.train_batch_size` | `256` | EDAS 8B 脚本 |
| `data.gen_batch_size` | `512` (= train × 2) | EDAS 8B 脚本 |
| `data.val_batch_size` | `1024` | EDAS 8B 脚本 |
| `data.max_prompt_length` | `2048` | EDAS 8B 脚本 |
| `data.max_response_length` | `8192` | EDAS 8B 脚本 |
| `rollout.name` | `vllm` | EDAS 8B 脚本 |
| `rollout.n` | `10` | EDAS 8B 脚本 |
| `rollout.temperature` | `1.0` | EDAS 8B 脚本 |
| `rollout.top_p` | `1.0` | EDAS 8B 脚本 |
| `rollout.gpu_memory_utilization` | `0.7` (4B 模型,见 HARDWARE_ADAPTATION.md) | override |
| `rollout.tensor_model_parallel_size` | `1` (4B 单卡足够,见 HARDWARE_ADAPTATION.md) | override |
| `rollout.max_num_batched_tokens` | `32768` | EDAS 8B 脚本 |
| `rollout.enable_chunked_prefill` | `True` | EDAS 8B 脚本 |
| `rollout.enable_dynamic_max_tokens` | `True` | EDAS 8B 脚本 |
| `rollout.length_log_interval` | `1` | EDAS 8B 脚本 |

## 5. Checkpoint / Trainer

| 名称 | 值 | 来源 |
|---|---|---|
| `trainer.total_epochs` | `2` | EDAS 8B 脚本 |
| `trainer.save_freq` | `10` | EDAS 8B 脚本 |
| `trainer.test_freq` | `5` | EDAS 8B 脚本 |
| `trainer.max_actor_ckpt_to_keep` | `20` | EDAS 8B 脚本 |
| `trainer.val_before_train` | `False` (跳过 smoke-style val) | override (避免每 epoch 浪费 10-30 min) |
| `trainer.default_local_dir` | `experiments/voterisk/checkpoints/{task_name}` | override |
| `trainer.logger` | `[console]` (无 SwanLab 上传) | override |
| `trainer.project_name` | `voterisk_qwen3_4b` | override |

## 6. EDAS 自身超参数 (paper defaults)

| 名称 | 值 | 来源 |
|---|---|---|
| `branch_adv_alpha` | `0.4` | run_dapo_branch_adv.sh (8B 脚本里给的 0.4/0.3/2.0) |
| `branch_adv_beta` | `0.3` | 同上 |
| `branch_adv_kappa` | `2.0` | 同上 |
| `branch_adv_warmup_start` / `_steps` | `0` / `0` (立即生效) | 默认 |
| `branch_adv_acc_key` / `answer_key` / `format_key` | `accuracy` / `answer_pred` / `format` | 默认 |

⚠️ 上面的 α=0.4 / β=0.3 是 EDAS 8B 脚本里的"recipe hyperparameters"(`run_dapo_branch_adv.sh` 第 96–100 行)。verl `core_algos.py` 里的默认 α=β=0.2 是算法模块默认值。我们的 launcher 把这些值作为"recipe 默认"读入(因为它们来自 EDAS 官方 launcher 而非算法模块),并固定沿用。**VoteRisk 不读 α/β**,改用下面的 `vote_risk_*`。

## 7. VoteRisk 自身超参数 (paper design + task instructions)

| 名称 | 值 | 说明 |
|---|---|---|
| `vote_risk_G` | `10` (= rollout.n) | 直接沿用 rollout 配置 |
| `vote_risk_K_target` | `16` | 任务要求:vote budget 16 |
| `vote_risk_alpha_smooth` | `0.5` | Dirichlet 平滑强度,任务锁定 |
| `vote_risk_lambda` | `0.4` | shaping 强度,任务锁定 |
| `vote_risk_warmup_start` / `_steps` | `0` / `0` | 立即生效 |

## 8. 评测统一设置 (EDAS 与 VoteRisk 完全一致)

| 名称 | 值 |
|---|---|
| 模型 | `trainer.default_local_dir/global_step_N/actor` (final step) |
| 数据集 | `MATH-500`, `AMC23`, `AIME24`, `AIME25` |
| Sampling temperature | `1.0` |
| Sampling top_p | `1.0` |
| Sampling seed | `42` |
| 每题 rollout N | `64` |
| max_response_length | `8192` (与训练一致) |
| Pass@K 报告 | K ∈ {1, 4, 8, 16, 32} |
| Maj@K 报告 | K ∈ {4, 8, 16, 32} |
| Tie 规则 | 与正确并列第一 → 计为 failure |
| Grader | `mathruler.grader.grade_answer` (与 EDAS 训练 reward 函数同源) |
| Answer extraction | `mathruler.grader.extract_boxed_content` + `\\boxed{...}` regex fallback |

## 9. 不修改 EDAS 的代码 (read-only invariant)

- `verl/trainer/ppo/core_algos.py::_apply_branch_advantage_adjustment` (EDAS 算法主体)
- `verl/trainer/config/algorithm.py` 中 `branch_adv_*` 字段
- `examples/branch_adv/reward_function/math_no_format.py`(reward 函数)
- `examples/branch_adv/preprocess_parquet.py`(数据预处理脚本,本实验用简化版 `prepare_data.py`)

VoteRisk 通过 **新增** `_apply_vote_risk_advantage_adjustment` + 在 `compute_grpo_outcome_advantage` 里添加 **互斥 dispatch**(VoteRisk 和 EDAS 不同时启用)接入,完全不动 EDAS 自身的函数体。