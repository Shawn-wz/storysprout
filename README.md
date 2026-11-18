# StorySprout

StorySprout 是一个小型英文故事语言模型项目。主线是：在 TinyStories 上做下一词预训练 → 用指令和故事做监督微调（SFT）→ 用偏好对训练奖励模型 → 用 GRPO 更新故事生成策略。GRPO 使用冻结的 SFT 参考模型和奖励模型，不需要 critic。

## 从哪里读起

| 阶段 | 入口与实现 | 输入 → 输出 |
| --- | --- | --- |
| 数据与分词 | `codes/prepare_data.py`、`codes/prepare_tokenizer.py` | 原始故事 → 本地 JSONL 和 BPE tokenizer |
| 预训练 | `codes/pretrain.py`、`codes/data.py` | 故事 → 续写模型 |
| SFT | `codes/sft.py`、`codes/sft_data.py` | 预训练权重 + prompt/response → 按指令写故事的模型 |
| 奖励模型 | `codes/train_reward.py`、`codes/reward_model.py` | SFT 骨干 + chosen/rejected 偏好对 → 对完整回答打分的模型 |
| GRPO | `codes/train_grpo.py`、`codes/grpo.py` | SFT 策略 + 奖励模型 + prompts → 更新后的策略 |

共享实现也在 `codes/`：`config.py` 定义配置，`model.py` 实现 Transformer，`training.py` 提供预训练/SFT 的训练和验证循环，`checkpoint.py` 管理权重，`generate.py` 提供续写接口。

```text
codes/         主流程和共享模块
tools/         手动检查、模型对比、单样本诊断、数据/上下文消融实验脚本
legacy/        原始预训练实现，保留作回归对照
tests/         自动化测试
artifacts/     tokenizer.json
data/          预训练故事、SFT 数据和偏好对
models/        每次训练的 checkpoint、指标和配置（视阶段而定）
experiments/   消融实验数据清单、独立运行和比较报告
reports/       诊断工具生成的报告
```

## 运行主流程

以下 PowerShell 命令均从 **storysprout 项目根目录** 执行。使用 Python 3.11 和已有的 `storysprout_env`；依赖版本见 [requirements.txt](requirements.txt)。在这个目录里，权重路径以 `.\models\` 开头。先用 `python -m codes.train_grpo --help` 查看参数不会启动训练；四个训练入口均支持 `--help` 和直接脚本调用。

已有 `data/train.jsonl`、`data/validation.jsonl` 和 `artifacts/tokenizer.json` 时跳过准备步骤。从零准备时依次运行 `python -m codes.prepare_data`（需要网络）和 `python -m codes.prepare_tokenizer`。两个命令默认拒绝覆盖已有文件；已有权重应配套使用其原 tokenizer。

下面是四个**单独执行**的训练命令。变量指向本工作区已有权重；如果重新训练某一阶段，把下一阶段的输入改为新运行的产物。

```powershell
$pretrained = ".\models\run_20261117_164236_439489\best.pth"
$sft = ".\models\sft_run_20261117_193832_830825\best.pth"
$reward = ".\models\reward_run_20261118_002839_831482\best.pth"

python -m codes.pretrain
python -m codes.sft --checkpoint $pretrained --data-dir .\data\sft_v1
python -m codes.train_reward --checkpoint $sft --epochs 3 --batch-size 8 --learning-rate 1e-5
python -m codes.train_grpo --checkpoint $sft --reward-checkpoint $reward --epochs 1 --batch-size 2 --group-size 4 --max-new-tokens 64 --updates-per-rollout 1 --learning-rate 1e-5
```

- **预训练**默认 3 epochs、batch size 8、学习率 0.001、CUDA。`--checkpoint PATH --epochs 1` 表示恢复 Adam、步数和 epoch 并追加一轮；未显式传学习率时沿用 checkpoint 的值。续训写入新目录，不覆盖来源；不保证逐样本复现随机顺序。
- **SFT** 从输入权重开始，使用新优化器。默认数据是 `data/sft_pilot`；上例显式使用 [sft_v1](data/sft_v1/README.md)。
- **奖励模型**默认读取 [reward_preferences_v1](data/reward_preferences_v1/README.md)，校验 tokenizer 和偏好张量指纹。标签是规则合成的偏好，不是人工评分。验证集选出最佳模型后才在测试集评估。
- **GRPO** 默认只读 `data/sft_v1/train.jsonl` 的 prompt。`--batch-size` 是每批 prompt 数，`--group-size` 是每个 prompt 的回答数。每组独立归一化奖励；固定参考模型约束策略更新。单次更新时训练 loss 接近零是预期现象，应在相同提示词上比较故事质量。

每次运行都新建 `models/` 子目录，记录 `metrics.jsonl` 和每轮的 `epoch_N.pth`。预训练、SFT 和奖励模型另存验证 loss 最低的 `best.pth`；GRPO 不做这项选择。语言模型使用 `load_checkpoint` 读取，奖励权重需要载入 `RewardModel`。新版 checkpoint 记录 tokenizer 的 SHA-256；旧格式可用于推理，但可能缺少指纹或续训状态。

## 可选工具与验证

从项目根目录用 `python -m tools.<模块名>` 调用工具。以下命令不训练模型：

```powershell
python -m tools.inspect_data
python -m tools.compare_models $sft .\models\grpo_run_20261118_151607_799147\epoch_1.pth --prompt "Write a story about a lost kite."
python -m unittest discover -s tests -p "test_generate.py" -v
```

`tools.overfit_sft` 是会训练的单样本诊断：例如 `python -m tools.overfit_sft --checkpoint $pretrained --steps 100`，结果写到 `reports/`。五组消融的准备、训练和汇总见[实验说明](experiments/data_context_20261117/README.md)及[结果](experiments/data_context_20261117/results.md)。旧版实现位于 `legacy/pretrain_old.py`。

完整测试命令是 `python -m unittest discover -s tests -p "test_*.py" -v`，其中包含小规模训练、续训和优化器更新。维护主流程和工具时，使用简短英文文档字符串、显式参数和返回类型；注释用于解释张量对齐、冻结边界等不直观的约定。

