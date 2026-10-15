# A4-1: 원하는 내용이 들어있는 문서를 똑똑하게 찾아주는 검색기 만들기

> **토큰화, 불용어 제거부터 NumPy 기반 TF-IDF 직접 구현, 코사인 유사도 검색 및 텍스트 분류 시스템입니다.**

---

## 1. 프로젝트 개요
- **목표**: NumPy 활용 TF-IDF 수식 직접 구현 및 Scikit-learn 결과 검증, 코사인 유사도 기반 문서 검색 엔진 구축, TF-IDF 기반 텍스트 분류 및 혼동 행렬 평가
- **학습 시간**: 70시간
- **주요 기술**: Python, NumPy, Scikit-learn, TF-IDF, Cosine Similarity, Information Retrieval

---

## 2. 디렉터리 구조
```text
A4-1/
├── src/
│   ├── preprocessor.py         # 토큰화 및 불용어 정제 파이프라인
│   ├── tfidf.py                # NumPy 기반 TF-IDF 직접 구현 및 Sklearn 검증 모듈
│   ├── search_engine.py        # 코사인 유사도 기반 상위 Top-K 문서 검색기
│   └── classifier.py           # TF-IDF 벡터 기반 텍스트 분류 모델
├── main.py                     # CLI 인터페이스 (`--query` `--topk`)
└── README.md                   # 구현 요약, 검증/실험 로그, BoW 한계와 후속 방향
```

---

## 3. 실행 방법
```bash
python main.py --query "검색어 입력" --topk 5
```
