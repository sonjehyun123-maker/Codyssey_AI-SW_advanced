# A5-1: 대출을 해줘도 될지 AI가 대신 판단해주는 시스템 만들기

> **금융 데이터(10,000건) 기반 규칙 기반 베이스라인 대비 ML 회귀(신용점수)/분류(연체예측) 모델링 파이프라인입니다.**

---

## 1. 프로젝트 개요
- **목표**: 규칙 기반 vs ML 모델 성능 비교(Accuracy, F1, AUC), 규제 선형회귀(Ridge/Lasso) 기반 신용점수 예측, 머신러닝 분류 기반 연체 위험 예측
- **학습 시간**: 80시간
- **주요 기술**: Python, Scikit-learn, XGBoost/LightGBM, Ridge/Lasso, Confusion Matrix, ROC-AUC

---

## 2. 디렉터리 구조
```text
A5-1/
├── src/
│   ├── data_loader.py          # finance_data.csv 로드 및 전처리
│   ├── baseline.py             # 규칙 기반(Rule-based) 베이스라인 모델
│   ├── models/
│   │   ├── regression.py       # 신용점수 예측 회귀 모델 (Alpha 규제 튜닝)
│   │   └── classification.py   # 연체 예측 분류 모델 (Confusion Matrix, ROC-AUC)
│   └── evaluate.py             # 베이스라인 vs ML 모델 성능 비교 리포트 산출
├── notebooks/
│   └── modeling_experiments.ipynb # 모델 튜닝 및 학습 실험 노트북
├── requirements.txt            # 라이브러리 및 버전 명세
└── README.md                   # 프로젝트 개요, 실행 방법, 모델 성능 비교표
```
