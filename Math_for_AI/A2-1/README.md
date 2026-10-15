# A2-1: AI가 어떻게 학습하는지 수학으로 직접 풀어보기

> **선형대수, 다변수 미적분, 역전파 수식 유도, 최적화 알고리즘, 확률적 손실함수의 수학적 원리를 코드로 검증합니다.**

---

## 1. 프로젝트 개요
- **목표**: Power Iteration 고유값 계산, SVD 이미지 압축, 중심차분 수치미분, 2층 신경망 역전파 수식 손계산 및 NumPy 검증, GD/Momentum 옵티마이저 구현
- **학습 시간**: 60시간
- **주요 기술**: Python, NumPy, Matplotlib, Linear Algebra, Calculus, Optimization

---

## 2. 디렉터리 구조
```text
A2-1/
├── src/
│   ├── linear_algebra.py       # 행렬 변환 시각화, Power Iteration 고유값/고유벡터, SVD 이미지 압축
│   ├── calculus.py             # 수치 미분(중심차분법), 그래디언트 시각화
│   └── optimizer.py            # VanillaGD, Momentum 옵티마이저 및 수렴 경로 시각화
├── notebooks/
│   ├── backprop_derivation.ipynb # 2층 신경망 순전파/역전파 수식 유도 및 손계산 검증
│   └── probability_loss.ipynb    # 정규/베르누이 분포 시각화, MSE/Cross-Entropy와 MLE 연결
└── README.md                   # 수학적 원리 정리, 실험 결과 요약, 실행 방법
```
