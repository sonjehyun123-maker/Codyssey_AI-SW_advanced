# A3-1: 휴대폰으로 찍은 종이를 똑바르게 펴서 스캔하게 만들기

> **OpenCV를 활용한 엣지 검출, 윤곽선 추출, 원근 투영 변환(Perspective Transform) 기반 대화형 문서 스캐너입니다.**

---

## 1. 프로젝트 개요
- **목표**: 비스듬한 문서 자동 인식 및 직사각형 보정, 적응형 이진화 글씨 선명화, 트랙바 실시간 GUI 조절, 규칙 기반 CV의 실패 한계 분석
- **학습 시간**: 60시간
- **주요 기술**: Python, OpenCV (cv2), NumPy, Perspective Transform, Adaptive Threshold

---

## 2. 디렉터리 구조
```text
A3-1/
├── src/
│   ├── scanner.py              # 전처리, Canny 엣지, Contour, getPerspectiveTransform 스캔 엔진
│   └── gui.py                  # 트랙바 조절 및 단계별 윈도우 시각화 UI
├── test_images/                # 4가지 조건별 20장 테스트 이미지셋
│   └── .gitkeep
├── REPORT.md                   # 20장 정량 평가 성공률 표 및 실패 원인 분석 보고서
└── README.md                   # 실행 커맨드, 알고리즘 단계 요약, 규칙 기반의 한계
```
