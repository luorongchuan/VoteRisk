# VoteRisk vs EDAS — Qwen3-4B / DAPO 实验

公平比较:

```
方法 A：DAPO + EDAS          (paper baseline, branch_advantage_adjustment)
方法 B：DAPO + VoteRisk-16    (本实验,_apply_vote_risk_advantage_adjustment)
```

两者只在 advantage shaping 阶段不同,其它一切(模型、数据、超参、reward、verifier、optimizer、KL、clip、prompt、采样)完全相同。

---

## 0. 当前修改清单

| 文件 | 改动 |
|---|---|
| `verl/trainer/ppo/core_algos.py` | 新增 `_apply_vote_risk_advantage_adjustment`、`_vote_risk_single_mode`、`_get_multinomial_table`;在 `compute_grpo_outcome_advantage` 中添加 vote_risk_* 参数与 dispatch(EDAS 与 VoteRisk 互斥)。EDAS 算法自身 **未改动**。 |
| `verl/trainer/config/algorithm.py` | 新增 `vote_risk_enabled` / `vote_risk_G` / `vote_risk_K_target` / `vote_risk_alpha_smooth` / `vote_risk_lambda` / `vote_risk_warmup_start` / `vote_risk_warmup_steps`。 |
| `verl/trainer/ppo/ray_trainer.py` | `compute_advantage(...)` 新增 `vote_risk_kwargs` 参数;`fit()` 增加 VoteRisk 分支组装。 |
| `recipe/dapo/dapo_ray_trainer.py` | 同上 (镜像改动)。 |

新增文件全部在 `experiments/voterisk/`:

```
experiments/voterisk/
├── ORIGINAL_EDAS_CONFIG.md          # 官方训练配置清单
├── HARDWARE_ADAPTATION.md           # 2×A100 适配说明
├── README.md                        # 本文件
├── configs/                         # (空,以后放 yaml 用)
├── data/                            # (空)
├── logs/                            # 训练 / 评测日志
├── checkpoints/                     # 训练出来的 final ckpt
├── eval_rollouts/                   # 64 rollouts × 数据集 raw jsonl
├── results/                         # CSV / markdown / png
└── scripts/
    ├── prepare_data.py              # dapo-math-17k → formatted.parquet
    ├── run_train.sh                 # 单方法训练入口 (edas|voterisk)
    ├── eval_model.py                # 单方法评测入口 (Maj@K, Pass@K, ...)
    ├── run_sequential.sh            # 顺序跑 EDAS → VoteRisk(训练+评测)
    ├── build_final_report.py        # 汇总 CSV / png / FINAL_COMPARISON.md
    ├── test_vote_risk_core.py       # VoteRisk 算法单元测试
    └── test_dispatch.py             # EDAS↔VoteRisk dispatch 测试
```

---

## 1. 一次性准备

```bash
cd /home/luorongchuan/workspace_135/VoteRisk
git checkout voterisk-experiment          # 当前已经在该分支

# 跑一次单元测试,确认 VoteRisk 算法 OK
source /home/luorongchuan/miniconda3/etc/profile.d/conda.sh
conda activate verl
python experiments/voterisk/scripts/test_vote_risk_core.py
python experiments/voterisk/scripts/test_dispatch.py

# 生成 formatted 训练 parquet (一次性)
python experiments/voterisk/scripts/prepare_data.py
```

预期输出:`ALL VoteRisk unit tests PASSED.` 与 `ALL dispatch tests PASSED.`,以及 `[ok] wrote /home/luorongchuan/workspace_135/datasets/dapo-math-17k.formatted.parquet (17917 rows)`。

---

## 2. 一次跑完两个方法的训练 + 评测 (推荐)

```bash
# 默认行为:在 2 张 A100 上顺序跑 EDAS → VoteRisk,每个方法训练完立刻评测。
# GPU 通过 CUDA_VISIBLE_DEVICES 控制 (默认 0,1)。
SEQ_SMOKE=0 SEQ_EVAL_N=64 \
CUDA_VISIBLE_DEVICES=0,1 \
bash experiments/voterisk/scripts/run_sequential.sh
```

可选环境变量:

| 变量 | 默认 | 含义 |
|---|---|---|
| `SEQ_METHODS` | `edas voterisk` | 要跑哪些方法 |
| `SEQ_DATASETS` | `MATH-500,AMC23,AIME24,AIME25` | 评测数据集 |
| `SEQ_EVAL_N` | `64` | 每题 rollout 数 |
| `SEQ_SMOKE` | `0` | 是否给 train 加 --smoke |
| `SEQ_BASE_EVAL` | `0` | 是否在两个方法前先评测 base 模型 (1 = 是) |
| `SKIP_TRAIN` / `SKIP_EVAL` | `0` | 跳过训练 / 评测阶段 |
| `NUM_GPUS` | `2` | 训练用 GPU 数 |
| `EVAL_GPUS` | `1` | 评测 vLLM tensor parallel size |
| `CUDA_VISIBLE_DEVICES` | `0,1` | 可见 GPU |

`run_sequential.sh` 会在每一步写 `_DONE` 标记,断点续跑友好;每个步骤的 stdout/stderr 在 `experiments/voterisk/logs/seq_*.log`。

---

## 3. 分阶段手动跑 (调试 / 重跑单个方法)

```bash
# (1) Smoke test 5~10 个 training steps 确认无 OOM / NaN
SEQ_SMOKE=1 SKIP_EVAL=1 bash experiments/voterisk/scripts/run_sequential.sh

# (2) 单方法正式训练
CUDA_VISIBLE_DEVICES=0,1 bash experiments/voterisk/scripts/run_train.sh edas
CUDA_VISIBLE_DEVICES=0,1 bash experiments/voterisk/scripts/run_train.sh voterisk

# (3) 评测其中一种方法的 final ckpt
CUDA_VISIBLE_DEVICES=0 python experiments/voterisk/scripts/eval_model.py \
    --model_path experiments/voterisk/checkpoints/edas_qwen3_4b_<TS>/global_step_<N>/actor \
    --method edas \
    --datasets MATH-500,AMC23,AIME24,AIME25 \
    --n 64 \
    --tp_size 1 \
    --gpu_mem 0.85

# (4) 生成最终报告
python experiments/voterisk/scripts/build_final_report.py \
    --exp_root experiments/voterisk \
    --methods edas voterisk
```

---

## 4. 期望输出

`experiments/voterisk/results/`:

```
FINAL_COMPARISON.md          # 11/12 节回答任务文档里的 12 个问题
main_metrics.csv             # 一行一方法,平均 Pass@K / Maj@K / 机制指标
per_dataset_metrics.csv      # 一行一(方法,数据集)
per_problem_metrics.csv      # 一行一(方法,数据集,问题)
mechanism_metrics.csv        # p_correct / p_wrong_max / vote_margin / vote_risk_16 ...
training_metrics.csv         # 从训练日志解析的 reward / KL / grad_norm 等

maj_at_k.png                 # 各数据集 Maj@4/8/16/32 折线
pass_at_k.png                # 各数据集 Pass@1/4/8/16/32 折线
vote_margin.png              # vote_margin 柱状图
p_wrong_max.png              # p_wrong_max 柱状图
vote_risk_16.png             # vote_risk_16 柱状图
training_reward.png          # 训练 reward 曲线
```

---

## 5. 关键不变量 (fairness invariants)

训练和评测严格不动的部分 (EDAS / VoteRisk 完全相同):

- base 模型:`Qwen3-4B-Base`(`/home/luorongchuan/workspace_135/models/Qwen3-4B-Base`)
- 训练数据:`dapo-math-17k.formatted.parquet`,顺序由 verl `data.shuffle=True` 控制(同 seed)
- DAPO 配置:clip 0.2/0.28, KL off, filter_groups on, gen_batch_size=512
- rollout G=10, max_prompt_length=2048, max_response_length=8192
- optimizer:AdamW lr=1e-6, weight_decay=1e-2, grad_clip=1.0
- FSDP:param/optimizer offload, gradient_checkpointing
- reward:`compute_score_batch`(rule + sympy,LLM judge off)
- 评测:N=64 rollouts/problem, temperature=1.0, top_p=1.0, seed=42
- grader:`mathruler.grader.grade_answer`(与训练 reward 同源)
- tie rule:与正确并列第一 → failure

VoteRisk 与 EDAS 唯一差异:`_apply_vote_risk_advantage_adjustment` vs `_apply_branch_advantage_adjustment`,两者使用相同的 kappa-clip + 中心化重分配框架,仅 per-mode signal 不同。

---

## 6. 预期实验结果形态

按任务文档第 31 节:

**最理想** (VoteRisk 优于 EDAS):

```
                       Pass@1  Pass@16  Maj@16  Maj@32  p_wrong_max  vote_margin  vote_risk_16
EDAS                   ≈ base   ≈ base    较低    较低    较高         较低          较高
VoteRisk-16            ≈ EDAS   ≈ EDAS    ↑       ↑       ↓           ↑             ↓
```

Pass@K 不明显退化 + Maj@16/32 提升 + p_wrong_max/vote_risk_16 下降 + vote_margin 上升 → 说明 VoteRisk 真正抑制了威胁投票的错误模式。

---

## 7. 常见问题

**Q: VoteRisk 跑通了,但 Maj@K 没变?**
A: 看 `vote_risk/mean_risk` 是否始终很低(< 0.05),说明多数 group 风险小、shaping 没生效 → 训练步数不够,或 EDAS 的 shaping 太强。可以检查 `training_metrics.csv` 的 `vote_risk_mean_risk` 演化曲线。

**Q: OOM?**
A: 减小 `ACTOR_PPO_MICRO_BSZ_PER_GPU` 或 `ROLLOUT_GPU_MEM_UTIL`,**两个方法同步减**。记录到 `HARDWARE_ADAPTATION.md`。

**Q: 评测跑得太慢?**
A: 用更小 `--n` (例如 16) 做快速 sanity check;正式跑 `--n 64`。也可以把 `SEQ_DATASETS` 临时改成 `MATH-500` 单数据集。

**Q: 想从断点继续?**
A: `run_sequential.sh` 在每个阶段写 `_DONE` 标记在 `experiments/voterisk/checkpoints/<method>/_DONE`;删掉该标记即可重跑。`SKIP_TRAIN=1` / `SKIP_EVAL=1` 可独立跳过阶段。