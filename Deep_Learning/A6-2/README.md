# A6-2: AI가 어디서 자꾸 틀리는지 찾아내서 더 똑똑하게 만들기

> **전이 학습(CNN Transfer Learning)과 시계열(RNN/LSTM), 편향-분산 진단, 30건 이상의 오차 분석(Error Analysis), Data Leakage 방지 실전 파이프라인입니다.**

---

## 1. 프로젝트 개요
- **목표**: 사전학습 모델 전이 학습 및 시계열 예측 구현, 편향/분산 진단 리포트, 오분류 30건 이상 실패 원인 태깅 및 오차 분석표 작성, 시계열 데이터 누수 방지 검증
- **학습 시간**: 80시간
- **주요 기술**: PyTorch / TensorFlow, CNN Transfer Learning, RNN/LSTM, Error Analysis, Data Leakage Prevention

---

## 2. 디렉터리 구조
```text
A6-2/
├── src/
│   ├── models/
│   │   ├── transfer_cnn.py     # 이미지 분류 전이 학습 모델 (ResNet/EfficientNet 백본)
│   │   └── time_series_rnn.py  # 시계열 예측 RNN/LSTM 모델 (Leakage 방지 분할)
│   ├── error_analysis.py       # 오분류 샘플 수집 및 실패 원인 태깅/통계 모듈
│   └── visualize.py            # 필수 4종 시각화 플롯 생성 모듈
├── notebooks/
│   └── model_diagnosis.ipynb   # 편향/분산 학습 곡선 진단 및 실험 노트북
├── figures/                    # 4종 필수 시각화 저장소
│   ├── loss_curve.png          # 1) Train/Val Loss 곡선
│   ├── time_series_pred.png    # 2) 시계열 예측 vs 실제 그래프
│   ├── baseline_compare.png    # 3) 베이스라인 비교 막대그래프
│   └── error_gallery.png       # 4) 오분류 이미지 갤러리
└── README.md                   # 성능 진단 리포트, 오차 분석표, 베이스라인 비교표, 누수 방지 섹션
```
