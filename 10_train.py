import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# =========================
# 1. 학습 데이터
# =========================

data = [
    {
        "question": "LoRA가 무엇인가요?",
        "answer": "LoRA는 기존 모델의 가중치를 동결하고 작은 저랭크 행렬을 추가하여 학습하는 파인튜닝 기법입니다."
    },
    {
        "question": "LoRA의 장점은 무엇인가요?",
        "answer": "LoRA는 전체 모델을 학습하는 것보다 학습해야 하는 파라미터 수와 메모리 사용량을 크게 줄일 수 있습니다."
    },
    {
        "question": "LoRA에서 기존 모델의 가중치는 어떻게 되나요?",
        "answer": "기존 모델의 가중치는 동결하고 LoRA로 추가된 파라미터를 중심으로 학습합니다."
    },
    {
        "question": "LoRA와 Full Fine-Tuning의 차이는 무엇인가요?",
        "answer": "Full Fine-Tuning은 모델의 많은 기존 파라미터를 업데이트하지만 LoRA는 기존 가중치를 동결하고 작은 추가 파라미터만 학습합니다."
    },
    {
        "question": "LoRA의 r은 무엇인가요?",
        "answer": "r은 LoRA에서 사용하는 저랭크 행렬의 랭크를 의미하며 추가되는 파라미터 수와 표현 능력에 영향을 줍니다."
    },
]


# =========================
# 2. Tokenizer + Qwen
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
).to("cuda")


# =========================
# 3. LoRA
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

model.print_trainable_parameters()


# =========================
# 4. Optimizer
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)


# =========================
# 5. Training
# =========================

EPOCHS = 10

model.train()

for epoch in range(EPOCHS):

    total_loss = 0

    for example in data:

        messages = [
            {
                "role": "user",
                "content": example["question"]
            },
            {
                "role": "assistant",
                "content": example["answer"]
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

        # 이전 Gradient 제거
        optimizer.zero_grad()

        # Forward
        outputs = model(
            **inputs,
            labels=labels
        )

        loss = outputs.loss

        # Backward
        loss.backward()

        # LoRA Parameter Update
        optimizer.step()

        total_loss += loss.item()


    average_loss = total_loss / len(data)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Average Loss: {average_loss:.4f}"
    )


# =========================
# 6. LoRA Adapter 저장
# =========================

model.save_pretrained("./my_lora")

print("\n학습 완료!")
print("LoRA 저장 위치: ./my_lora")