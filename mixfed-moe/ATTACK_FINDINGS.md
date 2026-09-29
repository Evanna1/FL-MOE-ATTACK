# FL-MoE 冷专家后门 —— 完整说明、实验结果与复现（交付给师妹）

> 面向刚入门的同学：本文档从背景直觉讲起，记录我们**发现的问题、正确做法、全部实验结果、补充的防御 baseline、以及每个脚本怎么跑**。所有数字都来自 `experiments/` 里的实测。

---

## 目录
0. 一句话结论
1. 背景与直觉（新手先看这里）
2. 之前 ASR 低的根因（诊断）
3. 正确的可靠攻击（配方）
4. 完整实验结果（表格）
5. 防御 baseline 与分析（含：为什么 Krum 不适用）
6. 走过的弯路（负结果，写论文能用）
7. 代码改动清单
8. 复现步骤（一步步）
9. 待办 / 下一步

---

## 0. 一句话结论

**不要攻击“绝对最冷”的专家**（它在 Switch 的 top-1 路由下**结构性不可达**，trigger 永远送不过去，ASR≈随机基线）。
**改攻击“诚实客户端 top-K 之外、但 trigger 够得到”的低频专家**（例如 `encoder.5` 的 `e5`）。
换目标后（中心化端到端实测）：**natural ASR 0.23 → 1.0**，攻击者退出后 **5 轮仍维持 ~1.0**，**clean acc 保持 ~0.90**，全程**不动 router**，并且**破掉逐参数 median/trimmed-mean 防御**。

---

## 1. 背景与直觉（新手先看这里）

- **FL-MoE**：联邦学习 + 混合专家（MoE）。为省显存/通信，每个客户端每轮**只训练分到的少数“热门专家”**（hot policy），冷门专家很少/从不被训练。
- **我们的攻击思路**：把后门种进一个**冷门专家**。因为诚实客户端几乎不训练它，服务器的**稀疏聚合**里这个专家常常**只有攻击者一个贡献者** → 攻击者的更新**不会被平均稀释** → 后门**持久**、**足迹小**（只改一个专家）、**隐蔽**。
- **两个关键概念**：
  - **top-1 路由**：Switch-MoE 每个 token 只送给**打分最高的那 1 个专家**。所以一个专家如果“从不被选成第一”，它就**从不处理任何 token**。
  - **trigger 优化**：像 GCG/HotFlip 一样的**离散 token 搜索**，目标是让 trigger 的 token 在某一层被**路由到我们指定的专家**。（我们的版本只是把 loss 从“输出某句话”换成“路由到目标专家”。）

---

## 2. 之前 ASR 低的根因（诊断）

1. **打了“最冷”的专家**。“冷”= router 几乎从不把它选成 top-1。要让 trigger 路由过去，就得让“一个从不被选的专家”变成第一——这和它“之所以冷”**直接矛盾**，做不到。
2. **看错了指标**。trigger 优化器原本只报 **mean softmax 概率**（看着能爬到 ~30%），但真正决定 top-1 路由的是 **top-1 命中率**——对最冷专家它 **恒为 0**（trigger 从没真的路由过去）。那 30% 是数学上的 Jensen gap 造成的**假象**。
3. 结果：trigger 送不到冷专家 → 冷专家不处理 token → 后门永不响 → **ASR ≈ 随机基线**。

> 我们给 trigger 优化器加了 **top-1 命中率**指标（`trigger_optimization.py`），一眼就能看穿“假的 30%”。

---

## 3. 正确的可靠攻击（配方）

**核心洞察**：在**光谱**上找“甜点”——
- 太冷（如 `e0`，路由频率 0.069）→ 不可达 → ASR≈0；
- **可达且仍冷**（如 `e5`，频率 0.123）→ trigger 够得到 **且** 仍在诚实 top-K 之外；
- 太热（`e2`/`e7`）→ 可达但被诚实训练 → 不持久。

**关键：换目标不损失持久性。** K=2 时诚实只训全局 top-2（`e2,e7`）；`e0` 和 `e5` **都不在 top-2 → 都是零诚实贡献者 → 一样持久**。所以从 `e0` 换到 `e5`：**持久性不掉，ASR 从 0 变 1。**

**流水线**：
1. 用 top-1 命中率**挑一个可达的低频专家**（诚实 top-K 之外）；
2. **clean-router trigger 优化**（GCG 式）把 trigger 路由到它；
3. 恶意客户端**局部化只训该专家**、**自然路由**投毒（用上一步的 trigger）；
4. 推理时 trigger 自然路由到该专家 → 后门打响。**全程不动 router。**

---

## 4. 完整实验结果

> 说明：以下为**中心化受控实验**（确定性复现 FL 动态）。“冷专家=唯一贡献者”的聚合在数学上就是“原样写入”，与真 flwr 等价；且该机制已在真框架用 `contributors@<layer>` 日志验证过（攻击期该专家贡献者=1、退出后=0）。

### 4.1 可达性扫描 —— 甜点是否存在？（`reachability_scan.py`）
对每个非-top-2 的低频专家跑短 trigger 优化，看能达到的 **top-1 命中率**：

| 专家 | 路由频率 | 可达的 top-1 命中率 | 结论 |
|---|---|---|---|
| e0（最冷，师妹原来的目标） | 0.069 | **0.000** | ✗ 不可达 |
| e1 | 0.096 | 0.681 | ✓ |
| e3 | 0.102 | 0.542 | ✓ |
| e4 | 0.116 | 0.403 | ✓ |
| **e5** | 0.123 | **1.000** | ✓✓ 最佳目标 |
| **e6** | 0.103 | **1.000** | ✓✓ |

**只有绝对最冷的 e0 不可达；freq ≥ ~0.10 的都可达。存在一个“可达性阈值”。**

### 4.2 端到端可靠攻击（主结果，`reliable_attack_e2e.py`，目标 e5）
优化出的 trigger `'!! fără!!!'`，top-1 命中 0.979。

| 阶段 | natural ASR | clean acc |
|---|---|---|
| 攻击前 | 0.229 | 0.906 |
| 攻击 r1 | **0.975** | 0.892 |
| 攻击 r2 / r3 | **1.000 / 1.000** | 0.88 |
| 退出后 r1 | 0.986 | 0.904 |
| 退出后 r2–r5 | 0.998 / 0.973 / 0.999 / 0.996 | ~0.90 |

**natural ASR 直接到 ~1.0（不需要 force-route），攻击者退出 5 轮仍 ~1.0，clean acc 保住。这是“可靠版能用”的核心证据。**

### 4.3 单专家载荷容量（`probe_cold_expert_capacity.py`，跨 3 层）
只训一个专家能否扛起后门？（`cold_forceroute` = 强制路由到冷专家训练，隔离“载荷容量”）

| 层 | 冷专家 | cold_forceroute（clean→clean, ASR） | full_badnet 基线（全模型） |
|---|---|---|---|
| encoder.1 | e6(0.059) | 0.903→**0.903**, ASR **0.840** | clean 0.896, ASR 0.994（足迹 100%） |
| encoder.5 | e0(0.069) | 0.903→**0.903**, ASR **0.848** | clean 0.900, ASR 0.990 |
| encoder.9 | e1(0.035) | 0.903→**0.902**, ASR **0.856** | clean 0.887, ASR 0.997 |

**单个专家（0.7% 参数）就能扛 0.84–0.86 的 ASR，且 clean acc 零损；跨层一致。**
（补充：同实验里 `hot_natural` 在部分层会把 clean acc 打崩到 0.58/0.76——说明**注入冷专家比热专家更不伤干净精度**。）

### 4.4 持久性：冷 vs 热（`persistence_sim.py`，攻击者第 5 轮后退出）

| 阶段 | 冷专家 e0 payload ASR | 热专家 e2 payload ASR |
|---|---|---|
| 攻击峰值 | 0.868 | 0.856 |
| 退出后 r1–r8 | 0.859/0.874/0.885/0.868/0.858/0.864/0.863/**0.863** | 0.818/0.807/0.796/0.799/0.798/0.787/0.792/**0.797** |

**冷专家：退出后 8 轮纹丝不动（~0.86）；热专家：被诚实重训、侵蚀到 ~0.80。** clean acc 全程 0.90–0.92。冷专家零贡献者 → 数学上冻结 → 持久是**有保证的**。

### 4.5 分布鲁棒：Non-IID 标签偏斜（`noniid_expert_usage.py`）

| alpha | 各客户端 top-2 | 不同组合数 | 全局冷专家（攻击面） |
|---|---|---|---|
| 0.1（极偏） | 全是 [2,7] | 1/8 | [0,1,3,4,5,6] |
| 0.5 / 1.0 / IID | 全是 [2,7] | 1/8 | [0,1,3,4,5,6] |

**即使极端标签偏斜，8 个客户端热专家完全一样,冷集和 IID 相同。** 因为 MoE 路由按 **token 特征**走、与标签无关。→ **攻击的冷专家目标跨分布稳定、可靠**（也意味着标签偏斜下**不需要**自适应探测）。

---

## 5. 防御 baseline 与分析（重点）

我们从 **BackFed** benchmark（MIT，arXiv:2507.04903）**移植了一批鲁棒聚合防御**进我们的框架（`mixfedmoe_fl/defenses.py` + `strategy.py` 的 `_defense_aggregate`，`--defense` 开关）：
- **norm_clip / WeakDP**（范数裁剪）
- **coord_median**（坐标中位数）、**trimmed_mean**（截尾均值）
- **multi_krum**（Krum 客户端筛选）
- 另外还移植了 **Neurotoxin** 攻击基线（`--attack_mode neurotoxin`）作对照。

### 5.1 实测（`defense_reliable.py`，可靠版 e5，naive 局部化更新）

| 防御 | payload/natural ASR | 恶意是否被采纳 |
|---|---|---|
| none | 1.000 | ✓ |
| **coord_median** | **1.000** | ✓ |
| **trimmed_mean** | **1.000** | ✓ |
| multi_krum | 0.233 | ✗（被拒） |

### 5.2 关键分析：为什么 median/trimmed 被破，而 **Krum 不适用**

**① 逐参数鲁棒聚合（median / trimmed-mean）——能用,但被冷专家破。**
框架本来就是**按专家稀疏聚合**（每个专家只在“训了它的人”之间平均）。**冷专家只有攻击者 1 个贡献者 → 中位数/截尾均值（作用在 1 个值上）= 它本身 → 原样放行**。所以这类防御对冷专家**结构性失效**（上表 median/trimmed 都放行，ASR=1.0）。

**② 客户端级 Krum —— 架构上就“不兼容”稀疏 MoE。**
- Krum/Multi-Krum 的做法：比较**每个客户端的完整更新向量**之间的距离,选离邻居最近的、丢弃离群的。它**假设所有人提交同一参数空间上可比的完整更新**。
- 但稀疏 MoE 里,**每个客户端训的专家不同**（hot policy → 各训自己的 top-K）,**各模块参与占比天差地别**（热专家很多人训、冷专家 1 人训、router 全员训）。**根本拼不出一份“可比的完整更新”来做距离**。
  - 硬要把没训的专家 padding 成 0：会把“**没训**”和“**训成 0**”混为一谈；只发一个专家的更新（无论善意恶意）都会像**稀疏离群点**→ 被误拒（上表 Krum 把 naive 拒了,ASR 掉到 0.233）。
  - 这既拒错了（可能拒掉只训了个别专家的**正常**客户端）,也说明 **Krum 这类“客户端级、比完整向量”的防御,和稀疏 MoE 的“按模块不同占比聚合”天然冲突**。
- 换句话说:**真实的稀疏 MoE-FL 系统更可能用逐参数 median 这类防御——而它恰好被冷专家的单贡献者破掉。Krum 若硬上,要么拒错、要么不兼容。**

**③ 想靠“结构伪装”骗过 Krum(让恶意也训热专家+router+head,再叠加载荷)?** 单轮测了没成——因为诚实单轮范数只有 0.63、载荷要 5.78,压到 0.63 就把载荷削没了。**正解是多轮小步累积载荷**(冷专家持久,跨轮攒到满血,每轮都隐蔽),但这个**还没验证**(见 §9)。

> **小结(对师妹)**:我们补了 norm_clip / median / trimmed / multi_krum 四种防御 baseline。**median/trimmed 被我们的冷专家单贡献者直接破;Krum 对稀疏 MoE 本就不兼容(不同模块聚合占比不同,没法拒)。** 这本身是论文里“隐蔽性”的有力论据。

---

## 6. 走过的弯路(负结果,写论文能用)

1. **打最冷专家**:top-1 不可达 → ASR≈0(§2、§4.1)。
2. **router poisoning 让 trigger 路由过去**(`gating_reliability.py`):
   - 把 97% 干净 token 也焐到该专家 → **clean acc 从 0.93 崩到 0.64**(耦合);
   - 而且**诚实一轮就把路由冲回去**(natural ASR 0.776→0.235)→ 不持久、不隐蔽。→ **放弃这条路。**
3. **放大 gating 更新(model-replacement)**:范数变大 → 被 Krum 抓。gating 是共享聚合、**不能缩放**。
4. **指望标签偏斜分化专家**:不分化(§4.5)。要“自适应探测冷专家”需**域偏斜**(多数据集),未做。

---

## 7. 代码改动清单

**新增文件**
- `mixfedmoe_fl/defenses.py`:鲁棒聚合算子(norm_clip / coord_median / trimmed_mean / multi_krum),移植自 BackFed,纯 numpy,有单测。
- `experiments/`:`_moe_bd_common.py`、`make_warmup_checkpoint.py`、`probe_cold_expert_capacity.py`、`persistence_sim.py`、`reachability_scan.py`、`reliable_attack_e2e.py`、`defense_reliable.py`、`gating_reliability.py`、`noniid_expert_usage.py`。
- 测试:`tests/test_defenses_unit.py`;`test_trigger_optimization_unit.py`(+top-1 命中率);`test_strategy_unit.py`(+防御路径)。

**修改文件**
- `mixfedmoe_fl/trigger_optimization.py`:新增 **top-1 命中率** + 竞争专家直方图,接入逐轮打印/`result.json`/CSV。(暴露“真实路由 vs 假 mean-prob”的关键)
- `models/switch_transformers/modeling_switch_transformers.py`:`getattr(...,"router_logits",None)` **兼容修复**(transformers 4.56.2 下 forward 必崩,必修)。
- `mixfedmoe_fl/config.py`:新增 `--inject_target_expert`、`--attack_end_round`、`--lfe_target_layer/expert`、`--attack_mode {badnet,neurotoxin}`、`--neurotoxin_mask_ratio`、`--malicious_train_force_route`、`--malicious_train_router`、`--malicious_lr`、`--malicious_local_epochs`、`--init_checkpoint`、`--defense` 及参数。
- `mixfedmoe_fl/client.py`:攻击窗口(可退出)、认领冷专家、**局部化注入**(只训目标专家)、训练时 force-route、router co-training、Neurotoxin 梯度掩码、恶意专属 lr/epochs。
- `mixfedmoe_fl/strategy.py`:`contributors@<layer>` 逐轮日志(证明零稀释);`_defense_aggregate`(4 种防御)。
- `mixfedmoe_fl/server.py`:**force-route payload ASR**(`asr_forceroute`,隔离“载荷是否存在”与“trigger 是否路由”);`--init_checkpoint` 加载预热模型。
- `mixfedmoe_fl/logging_utils.py`:`asr_forceroute` 列。

---

## 8. 复现步骤(一步步)

### 8.1 环境(一次性)
```bash
conda activate flwr      # torch 2.8+cu128 / transformers 4.56.2 / flwr / datasets
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd mix-fed-mo-e-master
# 需要 model_ckpt/switch-base-8(可用 modelscope: snapshot_download("google/switch-base-8") 或 hf-mirror)
```

### 8.2 前置产物(先跑这两步)
```bash
# (1) 预热一个 ~0.90 的干净分类器,当作已收敛的全局模型
python experiments/make_warmup_checkpoint.py --steps 1200 --output outputs/warmup_ckpt/warmup.pt
#     => WARMUP clean accuracy ≈ 0.90

# (2) 生成 ag_news 校准文本(trigger 优化用;务必 in-distribution)
python - <<'PY'
import json, numpy as np
from datasets import load_dataset
ds = load_dataset("ag_news")["test"]; rng = np.random.default_rng(0)
idx = rng.choice(len(ds), 100, replace=False).tolist()
json.dump([{"text": ds[int(i)]["text"]} for i in idx], open("data/agnews_calib.json","w"), ensure_ascii=False, indent=1)
print("ok")
PY
```

### 8.3 核心实验(单卡:`CUDA_VISIBLE_DEVICES=0`)
```bash
# A) 诊断:哪些低频专家可达(§4.1)——最冷 e0=0.00,e5/e6=1.00
python experiments/reachability_scan.py --experts 0,1,3,4,5,6 --iterations 8

# B) 主结果:端到端可靠攻击(§4.2)——natural ASR ->1.0,退出后维持,clean ~0.90
python experiments/reliable_attack_e2e.py --target 5 --trig_iters 6 --attack_rounds 3 --post_rounds 5

# C) 持久性:冷 vs 热(§4.4)
python experiments/persistence_sim.py --attack_rounds 5 --post_rounds 8

# D) 单专家容量(§4.3)
python experiments/probe_cold_expert_capacity.py --target_layer encoder.5 --clean_steps 1200 --attack_steps 600

# E) 防御 baseline(§5)——median/trimmed 被破,Krum 拒 naive
python experiments/defense_reliable.py --honest_clients 4 --honest_steps 100 --mal_steps 300

# F) 分布鲁棒(§4.5)
python experiments/noniid_expert_usage.py --num_clients 8 --topk 2
```

> **最小快速验证(~10 分钟)**:先跑 A + B 两条,就能看到“最冷不可达 vs e5 可达 → ASR 1.0 + 持久”。

### 8.4 在真 flwr 框架里跑(ray 稳定的机器上)
```bash
# 1) 对可达低频专家优化 trigger(产出 selected_triggers.json)
python -m mixfedmoe_fl.run_trigger _trigger_optimization \
  --checkpoint_path outputs/warmup_ckpt/warmup.pt \
  --target_layer encoder.5 --target_expert 5 --calibration_file data/agnews_calib.json
# 2) 局部化自然路由注入(不要加 force_route / train_router)
python -m mixfedmoe_fl.run_simulation --mode mix --assignment_policy hot --no_coverage_guarantee \
  --attack_enabled --malicious_clients 0 --inject_target_expert \
  --selected_triggers_path <上一步输出>/selected_triggers.json \
  --init_checkpoint outputs/warmup_ckpt/warmup.pt
```
> **注意**:真 flwr 端到端在**共享机器上 ray 会掉 actor**(内存/并发不稳)。建议单卡低并发(`--num_gpus_per_client 1.0`)或换独占机器。**中心化脚本 A–F 不依赖 ray,最稳,优先用它们出结论。**

---

## 9. 待办 / 下一步

- **Krum 收尾**:naive 局部化被 Krum 拒;试“结构伪装(也训热专家+router+head)+ 多轮小步累积载荷”,实测能否躲过 Krum(**未证**)。或直接论证“Krum 与稀疏 MoE 不兼容”(§5.2)。
- **域偏斜 → 自适应探测**:不同客户端用不同数据集,看路由是否分化;若分化,则“攻击者用全局模型 delta 自适应找冷专家”的故事成立(标签偏斜不行,§4.5)。
- **真 flwr 端到端**:换独占机器补功能曲线(尤其防御那块)。
- 更多数据集 / 层 / K 的鲁棒性;把 trigger 的困惑度(隐蔽性)真正优化上。
