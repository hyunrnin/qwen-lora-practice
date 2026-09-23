import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
).to("cuda")

# 우리가 모델에게 보낼 대화
messages = [
    {
        "role": "user",
        "content": "LoRA가 무엇인지 한 문장으로 설명해줘."
    }
]

# 대화 형식을 Qwen이 이해하는 형태로 변환
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("\n=== 모델에게 실제로 전달되는 문장 ===")
print(text)

# 문장 → 숫자(Token ID)
inputs = tokenizer(
    text,
    return_tensors="pt"
)

print("\n=== Token ID ===")
print(inputs["input_ids"])

print("\n토큰 개수:", inputs["input_ids"].shape[1])

# Token을 GPU로 이동
inputs = {
    key: value.to("cuda")
    for key, value in inputs.items()
}

# 답변 생성
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

# 새롭게 생성된 부분만 가져오기
generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

# 숫자 → 글자
answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print("\n=== Qwen의 답변 ===")
print(answer)