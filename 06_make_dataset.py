from datasets import Dataset


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


dataset = Dataset.from_list(data)


print(dataset)

print("\n=== 첫 번째 학습 데이터 ===")
print(dataset[0])

print("\n=== 전체 데이터 개수 ===")
print(len(dataset))