from tokenizers import Tokenizer
from checkpoint import load_checkpoint
from generate import generate

model1, _ = load_checkpoint(
    r"D:\个人资料\青云计划\storysprout\models\run_20261117_164236_439489\best.pth",
    device="cuda",
)
model2, _ = load_checkpoint(
    r"D:\个人资料\青云计划\storysprout\models\sft_run_20261117_193442_194663\best.pth",
    device="cuda",
)
tokenizer = Tokenizer.from_file(
    r"D:\个人资料\青云计划\storysprout\artifacts\tokenizer.json"
)
request = "Write a short story about Alice and Bob finding a lost dog."
chat_prompt = f"<user>{request}<assistant>"
ans1 = generate(model1, tokenizer, chat_prompt)
ans2 = generate(model2, tokenizer, chat_prompt)
print(f"Before:\n{ans1}\n\nAfter:\n{ans2}")
