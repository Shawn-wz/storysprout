# 数据与上下文对照实验

五组均从相同随机初始化开始，仅训练 1 epoch，学习率固定为 `3e-4`。

|组别|训练语料|上下文|
|---|---|---|
|control|原始 `data/train.jsonl`，10 万条|256|
|A|清洗并补齐的 10 万条|256|
|B|与 A 完全相同|512|
|C|清洗后的 50 万条，包含 A 的全部故事|256|
|D|与 C 完全相同|512|

保留原模型结构、8192 词表 tokenizer、Adam 默认参数、batch size 8、FP32、右侧 padding、next-token loss 和原有训练函数。无学习率调度、梯度裁剪或梯度累积。随机种子 2026。上下文变化只改变 RoPE 缓存长度，不改变初始可训练参数。

数据清洗采用 NFC、首尾去空白，过滤疑似乱码和非法控制字符，按规范化空白后的文本精确去重，排除与原验证集相同的文本。保留原 10 万条中的有效故事，不足部分从官方全部四个训练分片的 reservoir 随机样本补齐。未做语义质量判定或近似去重。实际过滤计数、源版本、文件指纹见 `data_manifest.json`。

所有组使用原始的同一份 2000 条验证故事。`native_val_loss` 为各自上下文下的 loss；`common_256_val_loss` 统一只覆盖每篇故事前 256 个目标 token，用于组间比较。一个 epoch 的训练 token 和更新次数随数据量、上下文变化，并非等计算量对照。

每组目录 `runs/<group>/` 保存独立配置、`epoch_1.pth`（含 Adam 状态）、训练记录、四个固定故事开头在温度 0.5/0.8、固定种子下的续写，以及完成标志 `result.json`。这里的生成样本仅为定性检查。

复现入口（在 StorySprout 目录下）：

```powershell
python -m codes.data_ablation prepare
python -u -m codes.data_ablation train
```

准备阶段需要网络。训练阶段仅使用本地文件。已有 `result.json` 的组会跳过；未完成的组从头重跑，不会把中断训练当成完成的一轮。
