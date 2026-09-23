import torch

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# 1. 기존 Qwen 불러오기
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
)


# 2. LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


# 3. Qwen에 LoRA 붙이기
model = get_peft_model(
    model,
    lora_config
)


# 4. Parameter 확인
model.print_trainable_parameters()

print("\n=== 학습되는 Parameter ===")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)