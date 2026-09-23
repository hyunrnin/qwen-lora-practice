import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "./my_lora"


# =========================
# 1. Tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# =========================
# 2. 원본 Qwen
# =========================

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
).to("cuda")


# =========================
# 3. 학습한 LoRA Adapter 장착
# =========================

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

model.eval()


# =========================
# 4. 질문
# =========================

messages = [
    {
        "role": "user",
        "content": "LoRA가 무엇인지 한 문장으로 설명해줘."
    }
]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    text,
    return_tensors="pt"
)

inputs = {
    key: value.to("cuda")
    for key, value in inputs.items()
}


# =========================
# 5. 답변 생성
# =========================

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )


generated_tokens = outputs[0][
    inputs["input_ids"].shape[1]:
]


answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)


print("\n=== LoRA 학습 후 답변 ===")
print(answer)