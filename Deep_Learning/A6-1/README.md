# A6-1: AI의 속 엔진(두뇌)을 내 손으로 직접 만들어보기

> **NumPy만을 활용하여 Tensor, Autograd(자동 미분 연산 그래프), Layer, Optimizer를 직접 구현한 미니 딥러닝 프레임워크입니다.**

---

## 1. 프로젝트 개요
- **목표**: 밑바닥부터 만드는 자동 미분 엔진, 수치 미분 대비 역전파 오차 검증(오차 < 1e-7), 가중치 초기화(Zero vs Random vs He) 비교 실험, XOR 문제 및 MNIST 분류 달성
- **학습 시간**: 80시간
- **주요 기술**: Python, NumPy from scratch, Autograd Engine, Computation Graph, He Initialization

---

## 2. 디렉터리 구조
```text
A6-1/
├── src/
│   └── framework/
│       ├── tensor.py           # 연산 그래프 및 그래디언트를 보유한 Tensor 클래스
│       ├── autograd.py         # 역전파(Backpropagation) 자동 미분 노드
│       ├── layers.py           # Linear, ReLU, Sigmoid, Softmax, CrossEntropy
│       ├── initializers.py     # Zero, Random, He (Kaiming) 초기화
│       └── optimizer.py        # SGD, Adam 옵티마이저
├── gradient_check.py           # 수치 미분과 역전파 결과 오차 검증 스크립트 (오차 < 1e-7)
├── train_xor.py                # XOR 문제 해결 학습 검증 스크립트
├── train_mnist.py              # MNIST 분류 1 Epoch 정확도 80% 이상 달성 스크립트
├── experiment_init.py          # 초기화 방법별 Loss 감소 비교 실험 스크립트
├── figures/                    # 초기화 비교 그래프 (loss_comparison.png 등)
│   └── .gitkeep
└── README.md                   # 프레임워크 아키텍처, 검증 리포트, 자가 체크리스트 충족 결과
```

---

## 3. 자가 체크리스트
- [x] `gradient_check.py` 실행 시 모든 레이어 오차가 1e-7 이하인가?
- [x] Zero Init으로 학습 실패를 재현할 수 있는가?
- [x] He Init으로 XOR 문제가 정상 동작하는가? (Loss < 0.1 또는 정확도 95% 이상)
- [x] 초기화 비교 그래프가 `figures/` 폴더에 저장되어 있는가?
- [x] MNIST 분류 시 1 Epoch에서 정확도 80% 이상을 달성하는가?
