import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# 1. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# 2. Qwen
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
).to("cuda")


# 3. LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)


# 4. 학습 데이터 하나
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


# 5. Chat Template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False
)


# 6. Tokenize
inputs = tokenizer(
    text,
    return_tensors="pt"
)

inputs = {
    key: value.to("cuda")
    for key, value in inputs.items()
}


# 7. 정답 labels 만들기
labels = inputs["input_ids"].clone()


print("Input shape:", inputs["input_ids"].shape)
print("Label shape:", labels.shape)


# 8. Forward + Loss 계산
outputs = model(
    **inputs,
    labels=labels
)


loss = outputs.loss

print("Loss:", loss.item())