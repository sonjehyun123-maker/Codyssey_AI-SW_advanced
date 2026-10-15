# A5-2: AI가 왜 그런 결정을 했는지 이유를 사람에게 설명해주게 하기

> **고객 군집화(K-Means, PCA)와 XAI(SHAP)를 통한 머신러닝 모델의 글로벌/로컬 의사결정 해석 및 비즈니스 전략 제안입니다.**

---

## 1. 프로젝트 개요
- **목표**: 비지도학습 군집화 및 고객 페르소나 정의, SHAP Summary Plot(글로벌 해석), 승인/거절 케이스 Waterfall/Force Plot(로컬 해석), Dependence Plot 기반 비즈니스 제언
- **학습 시간**: 80시간
- **주요 기술**: Python, Scikit-learn, SHAP, PCA, K-Means Clustering, Explainable AI (XAI)

---

## 2. 디렉터리 구조
```text
A5-2/
├── src/
│   ├── clustering.py           # PCA 차원 축소 및 K-Means 군집화, 페르소나 통계 도출
│   └── explain.py              # SHAP Summary, Waterfall, Dependence Plot 생성 및 저장
├── figures/                    # SHAP 시각화 결과물 (PNG 필수 저장)
│   ├── shap_summary.png
│   ├── shap_waterfall_approved.png
│   ├── shap_waterfall_rejected.png
│   └── shap_dependence.png
├── notebooks/
│   └── xai_analysis.ipynb      # 군집화 시각화 및 SHAP 대화형 분석
└── README.md                   # 군집 정의, 페르소나, 글로벌/로컬 해석 결과, 비즈니스 제언
```
