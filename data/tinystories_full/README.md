# 全量 TinyStories 训练准备

官方默认训练集共 2,119,719 篇，4 个完整原始 Parquet 分片已下载并通过 SHA-256 校验。
固定版本为 f54c09fd23315a6f9c86f9dc80f725de7d8f9c64，原始文件位置及指纹见 tokenized_manifest.json 的 sources 字段。
沿用此前清洗规则，得到 1,689,700 篇训练故事。完整排除统计见 manifest 的 cleaning 字段。
规则包括 NFC 标准化、过滤空白与疑似乱码、按规范化空白去重，以及排除与现有验证集重复的文本。
验证集使用已有 2,000 篇故事。tokenizer 使用项目现有的 8,192 词表版本。

训练从随机初始化开始，4 层、4 个 64 维注意力头、FFN 1024、735 万参数。
启动设置为 1 epoch、batch size 8、Adam 默认参数、固定学习率 3e-4、seed 2026、CUDA。
每轮 211,213 次更新，325,448,604 个有效目标 token。
上下文为 256，缓存保存每篇前 257 个 token 以构造下一词目标，311,141 篇在此长度被截断。
JSONL 中保留清洗后完整故事文本。分词缓存以 uint16 token 和 uint64 偏移存储，并在加载时逐样本转成 long tensor。

## 手动启动

在 PowerShell 中执行以下命令。

```powershell
& 'D:\个人资料\青云计划\storysprout\start_full_pretrain.ps1'
```

仅检查数据与前向计算时使用下面的命令。

```powershell
& 'D:\个人资料\青云计划\storysprout\start_full_pretrain.ps1' -CheckOnly
```

启动配置保存在 training_config.json，完整数据和缓存指纹保存在 tokenized_manifest.json。
正式训练会在项目 models 下创建新的运行目录，记录逐轮指标并保存每轮权重及 best.pth。
修改轮数时同时修改 training_config.json 的 train_config.epochs 和 arguments 中 --epochs 后面的值。
缓存对应 256 上下文及当前 tokenizer。更改这两项需要重新生成缓存。

官方数据源为 https://huggingface.co/datasets/roneneldan/TinyStories。
