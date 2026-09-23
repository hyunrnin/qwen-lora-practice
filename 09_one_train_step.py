import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# =========================
# 1. 모델 준비
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
).to("cuda")


# =========================
# 2. LoRA 붙이기
# =========================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(
    model,
    lora_config
)


# =========================
# 3. 학습 데이터 1개
# =========================

messages = [
    {
        "role": "user",
        "content": "LoRA가 무엇인가요?"
    },
    {
        "role": "assistant",
        "content": "LoRA는 기존 모델의 가중치를 동결하고 작은 저랭크 행렬을 추가하여 학습하는 파인튜닝 기법입니다."
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False
)

inputs = tokenizer(
    text,
    return_tensors="pt"
)

inputs = {
    key: value.to("cuda")
    for key, value in inputs.items()
}

labels = inputs["input_ids"].clone()


# =========================
# 4. Optimizer 준비
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)


# =========================
# 5. 학습 전 LoRA 값 저장
# =========================

target_parameter = None
target_name = None

for name, param in model.named_parameters():

    if param.requires_grad and "lora_B" in name:
        target_parameter = param
        target_name = name
        break


before = target_parameter.detach().clone()

print("확인할 Parameter:")
print(target_name)


# =========================
# 6. Forward
# =========================

model.train()

outputs = model(
    **inputs,
    labels=labels
)

loss = outputs.loss

print("\n학습 전 Loss:", loss.item())


# =========================
# 7. Backward
# =========================

optimizer.zero_grad()

loss.backward()


# =========================
# 8. Parameter Update
# =========================

optimizer.step()


# =========================
# 9. 변경 확인
# =========================

after = target_parameter.detach().clone()

difference = (after - before).abs().sum()

print("Parameter 변화량:", difference.item())