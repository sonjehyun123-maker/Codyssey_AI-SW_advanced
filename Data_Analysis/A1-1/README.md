# A1-1: 쇼핑몰에서 누가 자주 오고 많이 사는지 분석해서 단골 찾기

> **이커머스 멀티모달 데이터(수치, 텍스트, 이미지 배열)를 NumPy/Pandas 기반 파이프라인으로 처리하고 RFM 분석으로 고객을 세분화합니다.**

---

## 1. 프로젝트 개요
- **목표**: 멀티모달 전처리 파이프라인 OOP 설계, EDA 및 IQR 이상치 처리, RFM 기반 고객 세분화 및 비즈니스 전략 도출
- **학습 시간**: 60시간
- **주요 기술**: Python, NumPy, Pandas, Matplotlib, Seaborn, RFM Segmentation

---

## 2. 디렉터리 구조
```text
A1-1/
├── src/
│   └── pipeline.py             # 객체 지향(OOP) 전처리 및 특성 공학 파이프라인 클래스
├── notebooks/
│   └── analysis_report.ipynb   # 6종 시각화, 통계 분석, RFM 세분화 리포트
├── data/                       # 샘플 데이터셋 (.gitignore 대상)
│   └── .gitkeep
└── README.md                   # 프로젝트 리포트 (데이터 설명, 인사이트, 실행 방법)
```

---

## 3. 실행 방법
```bash
python src/pipeline.py
```
