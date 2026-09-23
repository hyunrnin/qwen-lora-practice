import torch
from transformers import AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
)

total_params = sum(p.numel() for p in model.parameters())

print("전체 Parameter 개수:", f"{total_params:,}")
print("전체 Parameter 개수(M):", round(total_params / 1_000_000, 2), "M")