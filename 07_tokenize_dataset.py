from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


question = "LoRA가 무엇인가요?"
answer = "LoRA는 기존 모델의 가중치를 동결하고 작은 저랭크 행렬을 추가하여 학습하는 파인튜닝 기법입니다."


# 1. Qwen 대화 형식 만들기
messages = [
    {
        "role": "user",
        "content": question
    },
    {
        "role": "assistant",
        "content": answer
    }
]


# 2. Chat Template 적용
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False
)

print("=== 1. Chat Template 적용 결과 ===")
print(text)


# 3. Tokenize
tokens = tokenizer(
    text,
    return_tensors="pt"
)

print("\n=== 2. Input IDs ===")
print(tokens["input_ids"])

print("\n=== 3. 토큰 개수 ===")
print(tokens["input_ids"].shape[1])


# 4. 다시 글자로 복원
decoded = tokenizer.decode(tokens["input_ids"][0])

print("\n=== 4. 다시 Decode ===")
print(decoded)