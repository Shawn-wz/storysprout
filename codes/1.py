from tokenizers import Tokenizer
from checkpoint import load_checkpoint
from generate import generate


model, checkpoint = load_checkpoint(r"D:\个人资料\青云计划\storysprout\models\run_20261117_132554_034152\best.pth", device="cuda")
tokenizer = Tokenizer.from_file(r"D:\个人资料\青云计划\storysprout\artifacts\tokenizer.json")
prompt = "Alice and Bob were playing in a forest."
ans = generate(model, tokenizer, prompt)
print(ans)
