# 奖励模型偏好数据 v1

用途：先跑通奖励模型训练。标签由现有合成 SFT 参考答案与规则劣化答案构造，非人工偏好；没有逐项语义审核。完整来源与限制见 manifest.json。

每条 JSONL 包含 prompt、chosen、rejected、negative_type 与来源。每个提示词生成四类比较：替换要求的实体、长度接近的离题回答、结尾重复、去掉结尾。简单规则仍可能产生长度等捷径，不能把高准确率当作真实故事评判能力。

先按 prompt 划分：160 个训练提示词、40 个验证提示词、8 个测试提示词；每个提示词 4 对。RM 三个集合之间没有相同的规范化 prompt。原 SFT 已接触过源示例；测试来自 SFT 开发集，不能用于证明完全独立的语言模型泛化。

每个 split 同时提供 .jsonl 与 .pt。后者已经使用当前 tokenizer 编码，包含 chosen_ids、chosen_lengths、rejected_ids、rejected_lengths 四个 long tensor。长度包含 BOS、角色标记、完整 prompt、回答及 EOS，不包含 PAD。回答不做 next-token shift；最大完整长度不超过 256，超长不会静默截断。为方便直接构造 TensorDataset，每个 split 的 chosen/rejected 分别右侧补齐至其最大长度，batch 中允许进一步裁剪，但不需要自己实现 collate。

所有 .pt 均可用 torch.load(path, weights_only=True) 读取。tokenizer、JSONL 与 tensor 文件的 SHA-256 见 manifest.json。不要将同一提示词的不同比较对再随机分散到不同集合。
