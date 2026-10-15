# A3-2: 영상 속 움직이는 사람·물건을 따라가며 표시해주기

> **배경 차분(Background Subtraction), 객체 추적(ID 할당/궤적 추적), 특징점 매칭 기반 실시간 영상 분석 시스템입니다.**

---

## 1. 프로젝트 개요
- **목표**: 배경 분리 기반 모션 감지, 객체 ID 유지 및 이동 경로 시각화, 특징점 매칭(ORB/SIFT) 기반 타겟 재인식, 고전 CV의 한계(겹침/가림/조명) 분석
- **학습 시간**: 80시간
- **주요 기술**: Python, OpenCV, MOG2, Optical Flow / SORT, ORB/SIFT Feature Matching

---

## 2. 디렉터리 구조
```text
A3-2/
├── src/
│   ├── tracker.py              # 모션 감지 및 고유 ID 객체 추적 모듈
│   ├── feature_matcher.py      # 특징점 검출 및 템플릿 매칭 모듈
│   └── main.py                 # 비디오/웹캠 실시간 스트림 처리 메인 루프
├── videos/                     # 테스트 비디오 클립
│   └── .gitkeep
├── REPORT.md                   # ID Switch/추적 실패율 정량 측정 및 딥러닝 필요성 분석 보고서
└── README.md                   # FPS 제어 및 단축키 안내, 성능 측정 결과
```
