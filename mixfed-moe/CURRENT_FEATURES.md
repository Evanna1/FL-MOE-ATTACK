# MixFedMoE 当前功能说明

本文档依据当前仓库代码整理，描述现阶段已经可用的功能、运行方式、输出文件和已知限制。项目用于本地、隔离环境中的联邦学习与混合专家模型（MoE）安全研究。

## 1. 项目概览

MixFedMoE 基于 Flower、PyTorch 和 Hugging Face，实现了 Switch Transformer 文本分类的联邦学习仿真。当前核心能力包括：

- 四种客户端训练模式：`full`、`mix`、`drop`、`flex`；
- 基于专家激活热度或随机策略的逐层专家分配；
- 面向稀疏 MoE 参数的按贡献者聚合；
- 中央服务器干净准确率与攻击成功率（ASR）评估；
- BadNet、Neurotoxin 和低频专家定向注入实验；
- 低频专家分析、路由触发器优化和触发 token 路由追踪；
- Norm Clip、坐标中位数、截尾均值和 Multi-Krum 防御；
- 检查点、逐轮指标、攻击指标及专家分析结果持久化；
- 多组批量实验、消融实验和独立验证实验。

主入口为：

```bash
python -m mixfedmoe_fl.run_simulation --mode full
```

## 2. 技术栈与模型

- Python 3.10（环境脚本默认值）
- PyTorch 2.8.0
- Transformers 5.0.0
- Datasets 2.21.0
- Flower Simulation 1.30.0
- Flower Datasets 0.6.0
- Ray（由 Flower Simulation 后端使用）

默认模型路径为 `model_ckpt/switch-base-8`。仓库中的 `SwitchTransformersForSequenceClassification` 在 Switch Transformer 编解码器上增加了句子分类头，并使用最后一个 EOS 位置对应的 decoder 表示完成分类。模型支持单标签分类、回归和多标签损失，但当前内置数据集均用于单标签文本分类。

## 3. 支持的数据集

| 命令行名称 | Hugging Face 数据集 |
|---|---|
| `ag_news` | `ag_news` |
| `imdb` | `imdb` |
| `20news` | `SetFit/20_newsgroups` |
| `sst2` | `SetFit/sst2` |
| `yelp_polarity` | `yelp_polarity` |
| `emotion` | `dair-ai/emotion` |

数据处理具备以下行为：

- 使用 Flower Datasets 的 `IidPartitioner` 将训练集 IID 划分给客户端；
- 数据集必须能规范化为 `text`、`label` 两列；
- tokenizer 会为每条输入保留空间并追加 EOS；
- 客户端使用完整本地分区训练，不创建本地验证集；
- 服务器使用官方测试集做集中评估；
- `--test_samples <= 0` 使用完整测试集，否则使用确定性打乱后的前 N 条样本；
- 数据集、分区和随机行为由 `--seed` 控制。

> 当前主训练管线只实现 IID 划分。`--alpha` 是兼容参数，不参与数据划分。因此，脚本中带有 `niid` 名称或传入不同 `alpha` 的运行，在主仿真里仍是 IID。

## 4. 联邦训练流程

每轮联邦训练按以下流程执行：

1. 服务器从可用客户端中按 `fraction_fit` 采样；
2. 服务器为每个客户端、每个稀疏 MoE 层分配专家；
3. 客户端重新创建模型并加载完整全局状态，保持客户端无状态；
4. 客户端根据模式冻结、降精度或裁剪专家并执行本地训练；
5. 客户端仅上传实际可训练参数及参数名、专家分配和激活统计；
6. 服务器按样本数和参数实际贡献者进行稀疏聚合；
7. 服务器保存指定轮次检查点并进行集中测试；
8. 指标写入 CSV 和 JSONL。

Flower 仿真使用 Ray 后端。每个客户端的 CPU/GPU 资源分别由 `--num_cpus_per_client` 和 `--num_gpus_per_client` 控制。

### 4.1 客户端模式

| 模式 | 专家结构与精度 | 可训练内容 | 主要用途 |
|---|---|---|---|
| `full` | 完整模型，FP32 | 全部参数 | 完整训练基线 |
| `mix` | 分配到的热专家为 FP32；未分配专家为冻结 BF16 | 热专家、router、embedding、分类头 | 降低可训练参数与冷专家显存占用 |
| `drop` | 本地物理删除未分配专家，并裁剪 router 输出行 | 保留专家、局部 router、embedding、分类头 | 结构化裁剪与通信/计算研究 |
| `flex` | 完整模型保持 FP32，每层只训练 1 个专家 | 单专家、router、embedding、分类头 | 固定 K=1 的灵活稀疏训练 |

`flex` 强制要求 `--K 1 --assignment_policy hot`。`mix`、`drop` 和 `flex` 都执行精确 K 分配；K 不能超过对应层的专家数。

### 4.2 专家分配策略

支持两种策略：

- `hot`：第一轮循环分配；后续轮次根据各客户端上一轮的专家激活次数排序，优先分配热专家。
- `random`：每轮使用由种子、轮次和层编号派生的随机数生成器，无放回随机选择 K 个专家。

全局覆盖行为：

- `full` 覆盖所有专家；
- `flex` 始终要求全局专家覆盖；
- `mix/drop + hot` 默认修复孤立专家，保证每层所有专家至少有一个贡献者；
- `--no_coverage_guarantee` 可关闭 `mix/drop + hot` 的覆盖修复；
- `mix/drop + random` 不保证全局覆盖。

开启覆盖时，采样客户端数必须至少达到 `ceil(专家数 / K)`。

### 4.3 稀疏聚合

服务器不是把未训练专家视为零更新，而是只在该参数的真实贡献客户端之间进行按样本数加权平均：

- 普通共享参数在所有上传该参数的客户端之间聚合；
- 专家参数只在被分配该专家的客户端之间聚合；
- `drop` 模式上传的局部专家编号会映射回全局专家编号；
- `drop` 模式的 router 行按全局专家编号逐行聚合；
- 没有贡献者且未要求覆盖的参数保留上一轮全局值；
- 开启覆盖时，缺失专家参数或 router 行会直接报错。

每轮墙钟训练时间使用所有客户端本地训练时间的最大值，累计为 `cumulative_time`。

## 5. 评估功能

项目只启用服务器集中评估，因此 `--fraction_evaluate` 必须为 `0.0`。

每轮可记录：

- 测试损失 `eval_loss`；
- 测试准确率 `eval_accuracy` / `clean_accuracy`；
- 按客户端样本数加权的训练损失；
- 本轮耗时与累计耗时；
- 采样客户端数；
- 攻击成功率 `asr`；
- 目标专家在触发 token 上的平均路由概率；
- 目标专家 Top-1 命中率；
- 经过容量限制后真正执行目标专家的比例；
- 良性客户端对指定目标专家的激活次数和激活率。

ASR 测试会排除真实标签已等于目标标签的样本，给其余测试样本插入触发器，然后统计预测为目标标签的比例。

`--monitor_asr` 可在完全良性的训练中观测自然 ASR；它与 `--attack_enabled` 互斥，并且从完成第 1 轮后开始统计。

## 6. 攻击研究功能

### 6.1 BadNet 数据投毒

开启 `--attack_enabled` 后，指定恶意客户端会在攻击轮次内：

- 从非目标类别样本中确定性选取 `floor(本地样本数 × poison_rate)` 条；
- 在文本末尾追加 `--trigger`，或插入优化器给出的精确 `trigger_token_ids`；
- 将这些样本标签改为 `--target_label`；
- 使用投毒后的本地数据训练。

`--malicious_clients` 支持一个或多个客户端 ID。攻击窗口由 `--attack_start_round` 和 `--attack_end_round` 控制；结束轮为 `0` 表示攻击者始终参与攻击。

### 6.2 Neurotoxin

`--attack_mode neurotoxin` 使用相邻两轮全局模型之差作为服务器聚合更新的观测，然后只保留更新幅度最小的一定比例坐标用于投毒训练。保留比例由 `--neurotoxin_mask_ratio` 控制，用于在正常联邦训练较少修改的参数子空间中保存后门。

该实现遵循原论文的下载更新思想，但服务器不感知也不协助攻击：服务器向所有客户端正常下发当前全局模型；恶意客户端通过 Flower `Context.state` 独立保存自己上次收到的全局模型，并以两次观测之差构造精确 bottom-k 掩码。`--neurotoxin_mask_per_layer` 可改为逐参数张量独立构造掩码。掩码外坐标会在每次优化后恢复，避免 AdamW weight decay 造成漂移。还可通过 `--neurotoxin_max_update_norm` 对本地更新执行 L2 投影，并通过 `--neurotoxin_update_scale` 在上传前缩放最终的掩码更新。

恶意客户端第一次参与时没有历史全局模型，该次退化为不加 Neurotoxin 掩码的投毒训练；之后使用当前模型与自己上次观测模型之差。客户端即使尚未进入攻击窗口，也会在被采样时更新本地快照。若 `fraction_fit < 1` 导致该客户端中间缺席若干轮，差值表示两次实际参与之间的累计全局变化，而不是严格相邻轮变化。Non-IID 情况下，该信号来自全局聚合模型，不依赖恶意客户端自己的局部梯度。

### 6.3 低频专家定向注入

`--inject_target_expert` 允许恶意客户端偏离服务器分配，将目标专家强制放入目标层的 K 个上传槽位。攻击有效轮次会自动对精确 trigger token 位置的目标 Router 行施加交叉熵损失，无需额外的 Router 训练开关；`drop` 模式会自动把全局专家编号映射为裁剪后的本地 Router 行。

目标可通过以下方式设置：

- 直接传入 `--lfe_target_layer encoder.5 --lfe_target_expert 5`；
- 使用 `--selected_triggers_path .../selected_triggers.json` 同时加载目标层、目标专家、触发文本和精确 token ID；
- 显式的 `--lfe_target_layer/--lfe_target_expert` 会覆盖文件中的目标元数据。

可用 `--malicious_lr` 和 `--malicious_local_epochs` 单独增强攻击轮中的恶意本地训练；值为 `0` 时沿用普通客户端配置。

### 6.4 攻击持久性

通过设置有限攻击窗口，可以在攻击者停止投毒后继续训练并逐轮观察 ASR。`shells/run_persistence_attack.sh` 已提供冷专家、热专家、标准 BadNet 和攻击者持续在线四组对照。

## 7. 低频专家分析

### 7.1 训练过程路由画像

开启 `--lfe_profile_enabled` 后，每个参与客户端会对实际训练 forward 注册 router hook，按层统计：

- 有效 token 数；
- 每个专家的 Top-1 选择次数和频率；
- 被选择时的平均概率；
- Top-1 与 Top-2 的平均概率间隔；
- 每层频率最低的前 K 个专家。

统计会忽略 padding，并在 `drop` 模式下把局部专家 ID 映射回全局 ID。

若配置了目标层和目标专家，即使未开启完整画像，系统也会跟踪该目标在真实本地训练中的激活，并汇总良性客户端结果。

输出位置为：

```text
<run_dir>/lfe_profiles/client_<id>/
├── round_0001.json
└── lfe_rounds.xlsx
```

Excel 包含 `rounds`、`low_frequency`、`all_experts` 三个工作表。

### 7.2 I-GCG 路由触发器优化

入口：

```bash
python -m mixfedmoe_fl.run_igcg_trigger \
  --checkpoint_path outputs/.../checkpoints/round_0020.pt \
  --target_layer encoder.5 \
  --target_expert 5
```

也可以用 `--lfe_profile` 代替 `--target_expert`，自动选择指定层排名第一的低频专家。

优化器具备以下功能：

- 对目标 router 的目标专家概率计算可微路由损失；
- 用 token embedding 梯度生成 GCG 单坐标替换候选；
- 过滤无法稳定完成 decode/encode 往返的候选；
- 合并排名靠前的候选形成 I-GCG 累积更新；
- 支持 append、prepend 和固定位置三种触发器位置；
- 使用独立 held-out 文本按 Top-1 命中率、margin、概率和损失选优；
- 达到目标概率与命中率阈值后可提前停止。

输出包括：

```text
<output_dir>/<timestamp>/
├── selected_triggers.json
├── optimization_history.json
└── heldout_evaluations.json
```

其中 `selected_triggers.json` 可直接传给主仿真的 `--selected_triggers_path`。

## 8. 防御功能

`--defense` 支持：

| 值 | 行为 |
|---|---|
| `none` | 默认按样本数加权聚合 |
| `norm_clip` | 共享参数与每个专家分别计算动态阈值；共享组使用 `Median + 3 × 1.4826 × MAD`，专家组使用 `Median + 2 × 1.4826 × MAD` |
| `coord_median` | 每个参数逐坐标取贡献者中位数 |
| `trimmed_mean` | 每个参数逐坐标去除两端一定比例后求均值 |
| `multi_krum` | 共享参数与每个 `(layer, expert)` 分组执行 Multi-Krum，只比较同组贡献者并分别筛选后聚合 |

当前 robust aggregation 只支持 `mix` 模式。相关参数：

- `--defense_trim_ratio`：截尾比例，范围 `[0, 0.5)`；
- `--defense_num_malicious`：Krum 假定的恶意客户端数，`0` 时取恶意客户端列表长度；
- `--defense_krum_m`：每个共享/专家组内 Multi-Krum 保留的贡献者数；`0` 时分别取“该组贡献者数减恶意数”。

对于只有一个贡献者的冷专家，坐标中位数和截尾均值会退化为直接接受该贡献；这也是仓库独立防御实验重点验证的现象。

## 9. 检查点与初始化

默认在第 `5、10、15、20` 轮保存检查点，可用 `--checkpoint_rounds` 自定义；不跟任何值可禁用保存。

格式为：

```python
{
    "round": 20,
    "parameter_names": [...],
    "state_dict": {...},
}
```

`--checkpoint_path` 可从以下格式初始化全局模型：

- 本项目的 `{"state_dict": ...}` 检查点；
- `{"model_state_dict": ...}` 包装；
- 原始 PyTorch state dict。

`--init_checkpoint` 是 `--checkpoint_path` 的兼容别名。检查点必须包含当前模型全部参数。

## 10. 输出目录与文件

主仿真会在指定 `output_dir` 下再创建唯一时间戳目录：

```text
<output_dir>/<YYYYmmdd_HHMMSS_microseconds>/
├── run_config.json
├── round_metrics.jsonl
├── round_metrics.csv
├── attack_results.csv
├── benign_lfe_activation.csv        # 配置目标专家且有良性统计时
├── checkpoints/
│   └── round_XXXX.pt
└── lfe_profiles/                    # 开启画像时
    └── client_<id>/
        ├── round_XXXX.json
        └── lfe_rounds.xlsx
```

`attack_results.csv` 是逐轮攻击相关指标的精简视图；即使未启用攻击也会创建，相关列为空。

## 11. 命令行参数总览

### 11.1 训练与资源

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--mode` | 必填 | `full/mix/drop/flex` |
| `--model_name_or_path` | `model_ckpt/switch-base-8` | 模型目录或名称 |
| `--dataset_name` | `ag_news` | 数据集 |
| `--num_clients` | `8` | 客户端总数 |
| `--num_rounds` | `100` | 联邦轮数 |
| `--K`, `--k` | `4` | 每层分配专家数 |
| `--fraction_fit` | `1.0` | 每轮训练客户端比例 |
| `--fraction_evaluate` | `0.0` | 必须保持 0，仅中央评估 |
| `--local_epochs` | `1.0` | 本地 epoch，可为小数 |
| `--learning_rate` | `2e-5` | 学习率 |
| `--weight_decay` | `0.0` | 权重衰减 |
| `--train_batch_size` | `16` | 本地训练批大小 |
| `--eval_batch_size` | `16` | 服务器评估/画像批大小 |
| `--max_length` | `256` | 最大 token 长度 |
| `--seed` | `42` | 随机种子 |
| `--num_cpus_per_client` | `4` | Ray 每客户端 CPU |
| `--num_gpus_per_client` | `1` | Ray 每客户端 GPU，可为小数或 0 |
| `--output_dir` | `outputs/mixfedmoe` | 运行输出根目录 |
| `--test_samples` | `1000` | 测试样本上限，非正数表示全部 |

### 11.2 专家调度与画像

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--assignment_policy` | `hot` | `hot/random` |
| `--coverage_guarantee` | 开启 | 保证 `hot` 策略全局专家覆盖 |
| `--no_coverage_guarantee` | - | 关闭上述覆盖保证 |
| `--lfe_profile_enabled` | 关闭 | 保存逐客户端、逐轮专家画像 |
| `--lfe_top_k` | `3` | 每层报告的低频专家数，范围 1～8 |

### 11.3 攻击与 ASR

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--attack_enabled` | 关闭 | 启用恶意客户端数据投毒 |
| `--monitor_asr` | 关闭 | 良性训练中监控自然 ASR |
| `--malicious_clients` | `0` | 恶意客户端 ID 列表 |
| `--poison_rate` | `0.1` | 投毒比例 |
| `--target_label` | `0` | 攻击目标标签 |
| `--trigger` | `cf` | 文本触发器 |
| `--attack_mode` | `badnet` | `badnet/neurotoxin` |
| `--neurotoxin_mask_ratio` | `0.95` | 保留的低全局更新幅度坐标比例 |
| `--neurotoxin_aggregate_all_layers` | 开启 | 在全部可训练参数的全局模型差上构造一个精确 bottom-k 掩码 |
| `--neurotoxin_mask_per_layer` | 关闭 | 改为对每个参数张量的全局模型差独立构造 bottom-k 掩码 |
| `--neurotoxin_max_update_norm` | `0` | 本地恶意更新的 L2 投影半径；0 表示关闭 |
| `--neurotoxin_update_scale` | `1` | 上传前对最终恶意模型增量的缩放倍数 |
| `--attack_start_round` | `1` | 攻击起始轮 |
| `--attack_end_round` | `0` | 攻击结束轮；0 表示不结束 |
| `--malicious_lr` | `0` | 恶意学习率；0 表示沿用普通值 |
| `--malicious_local_epochs` | `0` | 恶意本地 epoch；0 表示沿用普通值 |
| `--inject_target_expert` | 关闭 | 强制目标专家进入恶意客户端分配，只训练目标专家 |
| `--lfe_target_layer` | 空 | 目标 MoE 层，如 `encoder.5` |
| `--lfe_target_expert` | 空 | 目标专家编号 |
| `--selected_triggers_path` | 空 | I-GCG 的 `selected_triggers.json` |

### 11.4 兼容或废弃参数

| 参数 | 状态 |
|---|---|
| `--alpha` | 已废弃；主训练始终 IID |
| `--client_eval_ratio` | 已废弃；客户端不划本地验证集 |
| `--init_checkpoint` | `--checkpoint_path` 的别名 |
| `--lfe_profile_trainings` | 兼容参数；训练 hook 画像按每个参与轮次执行 |
| `--lfe_calibration_samples` | 兼容参数；训练 hook 使用实际训练批次 |
完整、实时的参数帮助可通过下列命令查看：

```bash
python -m mixfedmoe_fl.run_simulation --help
python -m mixfedmoe_fl.run_igcg_trigger --help
```

## 12. 快速开始

### 12.1 安装环境

```bash
conda create -n flwr python=3.10 -y
conda activate flwr
pip install -r requirements.txt
```

模型应位于默认目录 `model_ckpt/switch-base-8`，也可通过 `--model_name_or_path` 指定其他兼容路径。

### 12.2 基础训练

```bash
python -m mixfedmoe_fl.run_simulation \
  --mode full \
  --dataset_name ag_news \
  --num_clients 8 \
  --num_rounds 20 \
  --num_gpus_per_client 0
```

### 12.3 trigger优化

```bash
python -m mixfedmoe_fl.run_igcg_trigger \
  --checkpoint_path model_ckpt/round_0010.pt \
  --target_layer encoder.xx \
  --target_expert x \
  --calibration_file data/calibration.json \
  --calibration_samples 100 \
  --heldout_samples 20 \
  --trigger_length 3 \
  --iterations 50 \
  --candidate_batch_size 256 \
  --top_k 256 \
  --combine_top_p 7 \
  --margin 1.0 \
  --margin_weight 1.0 
```

### 12.4 良性 ASR 基线

```bash
python -m mixfedmoe_fl.run_simulation \
  --mode full \
  --monitor_asr \
  --trigger cf \
  --target_label 0
```

### 12.5 标准 BadNet

```bash
python -m mixfedmoe_fl.run_simulation \
  --mode full \
  --attack_enabled \
  --malicious_clients 0 \
  --poison_rate 0.1 \
  --trigger cf \
  --target_label 0
```

### 12.6 低频专家定向攻击

```bash
python -m mixfedmoe_fl.run_simulation \
  --mode full \
  --num_rounds 10 \
  --attack_enabled \
  --checkpoint_path model_ckpt/round_0010.pt \
  --selected_triggers_path outputs/igcg_trigger_optimization/.../selected_triggers.json \
  --malicious_clients 0 \
  --poison_rate 0.1 \
```

### 12.7 Robust aggregation

```bash
python -m mixfedmoe_fl.run_simulation \
  --mode mix \
  --defense trimmed_mean \
  --defense_trim_ratio 0.2
```

## 13. 批处理脚本与 Notebook

| 文件 | 当前用途 |
|---|---|
| `shells/run_9_experiments.sh` | 当前实际只对 `20news/ag_news/emotion` 运行 `flex`，尽管文件名和注释仍称多模式实验 |
| `shells/run_k_ablation.sh` | 对 K 值运行 `drop/mix/flex` 消融；注意主管线仍为 IID |
| `shells/run_random_policy_ablation.sh` | 随机专家策略消融；不同 alpha 当前不会改变划分 |
| `shells/run_persistence_attack.sh` | 四组冷/热专家后门持久性对照 |
| `shells/cache_datasets.sh` | 预下载部分常用数据集供离线运行 |
| `shells/create_env.sh` | 创建开发环境并下载 Switch Base 8 |
| `proof_of_concept/train_poc.sh` | 单机 PoC 训练入口 |
| `proof_of_concept/plot_poc.ipynb` | PoC 结果绘图 |
| `shells/notebooks/profile_memory_communication.ipynb` | 内存与通信分析 |
| `shells/notebooks/figure_end_to_end_performance.ipynb` | 端到端性能图 |
| `flower_hf_example/simulation/tutorial.ipynb` | Flower/Hugging Face 示例 |

离线运行可设置：

```bash
export HF_HUB_OFFLINE=true
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 14. 独立研究实验

`experiments/` 下的脚本大多是无需 Ray 的集中式或受控仿真实验，用于快速验证攻击假设：

| 脚本 | 功能 |
|---|---|
| `make_warmup_checkpoint.py` | 中央干净预训练并生成 Flower 兼容检查点 |
| `probe_cold_expert_capacity.py` | 验证单个冷专家能否承载后门，并比较自然/强制路由 |
| `reachability_scan.py` | 扫描低频专家能否被干净 router 上的触发器稳定到达 |
| `reliable_attack_e2e.py` | 不投毒 router 的可达低频专家端到端攻击与持久性验证 |
| `persistence_sim.py` | 确定性比较冷专家与热专家后门的遗忘曲线 |
| `gating_reliability.py` | 检验 router 扰动与良性训练冲刷效应 |
| `noniid_expert_usage.py` | 独立构造 Dirichlet 标签偏斜并分析各客户端专家使用差异 |
| `defense_evasion.py` | 比较朴素/隐蔽联合更新在多种防御下的效果 |
| `defense_reliable.py` | 对自然路由、局部专家攻击测试稀疏聚合防御与 Krum |

这些脚本有自己的参数和输出路径，可运行 `python experiments/<脚本>.py --help` 查看。

## 15. 测试

运行非集成测试：

```bash
pytest tests -v -m "not integration"
```

运行全部测试（集成测试可能加载模型和数据，耗时较长）：

```bash
pytest tests -v
```

测试覆盖配置校验、数据处理、攻击数据构造、客户端训练行为、参数加载、稀疏聚合、专家分配、防御算法、日志、检查点、专家画像、I-GCG 优化辅助函数及集成冒烟流程。

## 16. 当前限制与注意事项

- 主联邦数据划分只有 IID；真正的 Dirichlet Non-IID 仅存在于独立实验 `noniid_expert_usage.py` 中。
- 客户端没有本地验证流程，所有正式评估均在服务器完成。
- Robust aggregation 当前仅支持 `mix`，不支持 `full/drop/flex`。
- `flex` 只支持热度策略且 K 必须为 1。
- `random` 策略不做孤立专家修复。
- 定向专家注入目前冻结 router，只依赖自然路由；两个 router 攻击开关尚未生效。
- 精确触发 token 路由指标要求 `selected_triggers.json` 中包含 `trigger_token_ids`；只有文本触发器时仍可评估 ASR，但不能精确报告触发 token 的路由指标。
- `run_9_experiments.sh` 当前的 `MODES` 实际只包含 `flex`，名称与顶部注释未同步。
- 批处理脚本需要 Bash；Windows 原生 PowerShell 用户可在 WSL/Git Bash 中运行，或直接使用 Python 模块入口。
- 项目面向授权的本地学术安全研究，不应对第三方系统或未授权模型开展攻击。

## 17. 主要代码结构

```text
mixfedmoe_fl/
├── config.py                     # CLI、配置与校验
├── data.py                       # 数据加载、IID 分区、tokenization
├── client.py                     # 四种客户端模式、本地训练、攻击逻辑
├── strategy.py                   # 专家分配、稀疏聚合、防御接入、检查点
├── server.py                     # 服务端模型和集中评估
├── attack.py                     # 检查点/触发器加载、BadNet 数据构造
├── defenses.py                   # Robust aggregation 算法
├── expert_profiler.py            # 训练路由画像与 JSON/XLSX 输出
├── igcg_trigger_optimization.py  # I-GCG 路由触发器优化
├── logging_utils.py              # CSV/JSONL 日志
├── run_simulation.py             # 主仿真入口
└── run_igcg_trigger.py           # 触发器优化入口

models/switch_transformers/
└── modeling_switch_transformers.py  # 自定义 Switch Transformer 分类模型

experiments/                      # 独立攻击、防御和持久性实验
proof_of_concept/                 # 单机 MoE 模式 PoC
shells/                           # 批处理、环境和缓存脚本
tests/                            # 单元测试与集成测试
```
