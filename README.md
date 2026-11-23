# StorySprout

StorySprout 是一个约 **735 万参数**的小型英文故事语言模型项目，使用 TinyStories 预训练，并通过数据、上下文、结构、优化配置、SFT 与 GRPO 对照研究人物关系和故事一致性。

**当前状态：阶段性完成，暂时停止新增实验。** 已完成完整训练与评估流程，模型能够生成英文故事片段，但人物、物品、目标与因果关系仍不可靠。专项测试的改善尚未稳定迁移到开放故事生成。

完整背景、配置、逐轮结果、限制、权重清单及复核方法见 [项目报告](reports/PROJECT_REPORT.md)，整理日期为 2026-11-19。

## 主要结果

全量预训练使用 1,689,700 篇清洗故事，一轮处理约 3.25 亿有效目标 token。基准结构为 4 层、256 维、4 个 64 维注意力头、FFN 1024、RoPE、8192 词表，默认上下文 256。

|阶段|普通故事验证 loss|原人物归属开发题|
|---|---:|---:|
|50 万篇单轮，C 组|1.9994|59.6%|
|10 万篇五轮|2.0325|47.7%|
|全量单轮底座|1.8309|53.4%|
|专项状态 SFT|2.1328|77.1%|
|90% 故事 + 10% QA|1.7948|66.1%|
|60% 故事 + 10% QA + 30% 一致性续写|1.8078|77.9%|

这里的 loss 使用相同 tokenizer、验证文本与目标覆盖。人物归属为反复使用的 384 道模板开发题，不能解释为故事正确率。混合续写组在新同类 QA 上达到 87.9%，在新名字、新问句和长间隔组合上只有 13.8%；小样本开放故事审阅也未显示可靠优势。

困难 GRPO 相对继续 SFT 在分布外题上为 30.0% 对 24.1%，在同分布题上为 85.6% 对 90.6%，合计仅多答对 3/640。结果支持局部收益，不支持“RL 已解决故事逻辑”。各组主要只有一个训练种子，部分比较同时增加数据和计算，详见报告。

## 实验导航

|问题|记录|
|---|---|
|项目全貌、结论与能力边界|[完整项目报告](reports/PROJECT_REPORT.md)|
|清洗、数据量与 256/512 上下文|[五组实验](experiments/data_context_20261117/results.md)|
|注意力、FFN 与深度分配|[结构消融](experiments/architecture_20261118/results.md)|
|学习率、Adam 与梯度裁剪|[优化配置](experiments/optimizer_20261118/results.md)|
|新故事与重复训练|[等更新预算比较](experiments/repeated_data_20261118/results.md)|
|全量训练准备与评估|[准备说明](data/tinystories_full/README.md)、[评估分析](reports/full_pretrain_20261118/analysis.md)|
|状态 SFT 与规则奖励 GRPO|[分析](experiments/state_rl_20261118/analysis.md)、[困难任务对照](experiments/state_rl_20261118/hard_results.md)|
|实际语料的关系与因果问题|[数据审查](reports/narrative_data_audit_20261118/analysis.md)|
|回放普通故事与一致性续写|[混合实验](experiments/story_mix_20261118/analysis.md)|

## 实现入口

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

## 读取现有权重

GitHub 仓库包含代码、配置、实验记录和 tokenizer。下表列出的权重路径用于放置本地训练产物。运行推理示例前，请先在对应路径准备模型权重。全量训练所需的数据、缓存和本地配置见[准备说明](data/tinystories_full/README.md)。

全量预训练底座是后续状态和混合实验的共同起点；早期故事 SFT 使用的是另一个多阶段底座。两个阶段应分别识别。

|用途|权重|说明|
|---|---|---|
|全量预训练底座|`models/run_20261118_183053_375283/best.pth`|用于普通故事续写与后续实验基准|
|专项状态 SFT|`experiments/state_rl_20261118/sft/best.pth`|短答任务改善，普通故事预测有退化|
|困难 GRPO|`experiments/state_rl_20261118/grpo_hard/step_400.pth`|局部泛化研究分支|
|故事回放对照|`experiments/story_mix_20261118/control/step_2000.pth`|当前汇总中普通故事 loss 最低|
|一致性续写混合|`experiments/story_mix_20261118/mixed/step_2000.pth`|熟悉状态任务更好，开放故事未显示优势|

所有语言模型需配套使用原 [tokenizer.json](artifacts/tokenizer.json)。没有一个权重在所有任务上最好。

在项目根目录、已有 Python 环境中运行下面的 Python 示例，只进行加载与采样，不训练。

```python
from pathlib import Path

from tokenizers import Tokenizer

from codes.checkpoint import load_checkpoint
from codes.generate import generate

root = Path.cwd()
tokenizer_path = root / "artifacts" / "tokenizer.json"
tokenizer = Tokenizer.from_file(str(tokenizer_path))
model, metadata = load_checkpoint(
    root / "models" / "run_20261118_183053_375283" / "best.pth",
    device="cpu",
    tokenizer_path=tokenizer_path,
)
print(generate(
    model,
    tokenizer,
    "Once upon a time, a little girl found a red ball.",
    max_new_tokens=96,
    temperature=0.8,
    seed=2026,
))
```

`generate` 只接受正数 temperature，并在 EOS、新 token 预算或总上下文上限处停止。总上下文包含 BOS、prompt 和续写。聊天式 SFT 需要与训练一致的角色前缀，普通续写接口不会自动组装聊天格式。

## 全量训练检查与恢复实验

以下命令在项目根目录执行。阶段性收尾无需再次训练。

```powershell
.\start_full_pretrain.ps1 -CheckOnly
```

上面的命令核对清单并对训练、验证各执行一次前向，不更新权重。去掉 `-CheckOnly` 则会**从头启动新的全量训练**，设置为一轮、batch size 8、学习率 3e-4，详见 [全量准备说明](data/tinystories_full/README.md)。直接运行 `python -m codes.pretrain` 的默认设置不是这次全量实验。

|恢复实验入口|说明|
|---|---|
|`python -m tools.state_rl_experiment run`|专项状态 SFT、继续 SFT 与规则奖励 GRPO|
|`python -m tools.state_rl_hard`|追加困难 SFT 与 GRPO|
|`python -m tools.story_mix_experiment train`|普通故事回放与一致性续写混合|

这些入口会训练；已完成阶段会按结果标志跳过。详细数据、预算与选模策略见各实验协议，不能把命令再次执行成功当作又训练了一次。

### 共享词权重与配对状态问答

新模型可用 `--tie-word-embeddings --num-blocks 6` 启用共享词权重与六层结构。按原词表和 FFN 宽度计算，可训练参数为 6,826,496。使用 `--dry-run` 可以检查初始化、数据加载和前向计算。加载已有 checkpoint 时，结构以保存的配置为准。

```powershell
python -m tools.paired_state_experiment prepare
python -m tools.paired_state_experiment run --asset-root ../storysprout
```

`prepare` 只生成数据。`run` 使用 `--asset-root` 指定目录中的全量预训练权重、原 tokenizer 和 TinyStories 缓存，进行两组等预算 SFT。默认每组 500 次更新，每次监督 900 个故事 token 和 100 个问答 token。实验组对同一事件链改写叙述和问句，并交换最后两件物品的去向来构造相似表达、答案不同的样本。测试按事件链划分，分别报告熟悉句式、保留句式和新人名加长事件链的组合条件，并记录配对两题都答对的次数。数据、权重和结果默认写入 `experiments/paired_state/`。

## 早期主流程演示

以下 PowerShell 命令均从 **storysprout 项目根目录** 执行。使用 Python 3.11 和已有的 `storysprout_env`；依赖版本见 [requirements.txt](requirements.txt)。先用 `python -m codes.train_grpo --help` 查看参数不会启动训练；四个训练入口均支持 `--help` 和直接脚本调用。

已有 `data/train.jsonl`、`data/validation.jsonl` 和 `artifacts/tokenizer.json` 时跳过准备步骤。从零准备时依次运行 `python -m codes.prepare_data`（需要网络）和 `python -m codes.prepare_tokenizer`。两个命令默认拒绝覆盖已有文件；已有权重应配套使用其原 tokenizer。

下面保留早期故事 SFT、学习奖励模型与 GRPO 的四个**单独执行**的训练命令。它们用于流程演示，不等于后来的规则奖励状态实验。变量指向当时已有权重，不会自动读取上一行的新产物；重新训练后需手动更新来源路径。

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

上述四个主入口每次训练都新建 `models/` 子目录，记录 `metrics.jsonl` 和每轮的 `epoch_N.pth`。预训练、SFT 和奖励模型另存验证 loss 最低的 `best.pth`；GRPO 不做这项选择。语言模型使用 `load_checkpoint` 读取，奖励权重需要载入 `RewardModel`。新版 checkpoint 记录 tokenizer 的 SHA-256；旧格式可用于推理，但可能缺少指纹或续训状态。

## 可选工具与验证

从项目根目录用 `python -m tools.<模块名>` 调用工具。以下命令不训练模型：

```powershell
python -m tools.inspect_data
python -m tools.compare_models .\models\sft_run_20261117_193832_830825\best.pth .\models\grpo_run_20261118_151607_799147\epoch_1.pth --prompt "Write a story about a lost kite."
python -m unittest discover -s tests -p "test_generate.py" -v
```

`tools.overfit_sft` 是会训练的单样本诊断：例如 `python -m tools.overfit_sft --checkpoint $pretrained --steps 100`，结果写到 `reports/`。五组消融的准备、训练和汇总见 [实验说明](experiments/data_context_20261117/README.md) 及 [结果](experiments/data_context_20261117/results.md)。旧版实现位于 `legacy/pretrain_old.py`。

完整测试命令是 `python -m unittest discover -s tests -p "test_*.py" -v`，其中包含小规模训练、续训和优化器更新。维护主流程和工具时，使用简短英文文档字符串、显式参数和返回类型；注释用于解释张量对齐、冻结边界等不直观的约定。

重新执行评估工具可能重写相应结果与审阅材料；已有审阅需要和相同输出逐条对应。此次文档整理没有重新训练或重跑完整测试，历史检查结果及限制保存在项目报告和各实验审计中。

