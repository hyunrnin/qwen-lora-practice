import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Tokenizer 불러오는 중...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Model 불러오는 중...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
)

model = model.to("cuda")

print("모델 로딩 완료!")
print("모델이 올라간 장치:", next(model.parameters()).device)