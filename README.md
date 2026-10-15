# Codyssey_AI-SW_advanced

> **데이터 분석 → AI 수학 → 컴퓨터 비전 → 자연어 처리 → 머신러닝/XAI → 딥러닝 프레임워크 구현까지, AI 핵심 알고리즘과 내부 엔진을 직접 구현하며 학습한 기록입니다.**

라이브러리를 단순 호출하는 데 그치지 않고, **수학적 원리 증명, 밑바닥부터(Scratch) 프레임워크 구현, 규칙 기반 한계 분석과 딥러닝 전이, 모델 해석(XAI)**까지 시스템의 내부 동작을 깊이 이해하는 것을 목표로 합니다.

---

## 소개

이 저장소는 **"무엇을 공부했는가"보다 "직접 만들면서 무엇을 이해하게 되었는가"**를 보여주는 데 초점을 맞춥니다.

- **NumPy 기반 엔진 구현**: 외부 블랙박스 라이브러리에 의존하지 않고 Autograd, Tensor, Optimizer, TF-IDF 등을 직접 수식으로부터 코드로 구현했습니다.
- **실전 파이프라인 & OOP 설계**: 절차 지향 스크립트를 탈피하고 유지보수 가능한 객체 지향(OOP) 파이프라인을 구축했습니다.
- **한계 분석 & 실패 원인 진단**: 규칙 기반(Rule-based) 알고리즘의 실패 케이스를 정량적으로 분석하고 딥러닝/머신러닝 도입의 당위성을 입증했습니다.

---

## Featured Projects

기술적으로 가장 깊이 있고 의미 있었던 프로젝트입니다.

| 프로젝트 | 설명 | 핵심 기술 |
| :--- | :--- | :--- |
| **[Mini DL Framework](Deep_Learning/A6-1/README.md)** | Tensor, Autograd, 역전파, 초기화 기법을 직접 구현한 딥러닝 엔진 | Tensor · Autograd · Backprop · He Init |
| **[Interactive Document Scanner](Computer_Vision/A3-1/README.md)** | OpenCV 기반 지능형 윤곽선 검출 및 원근 투영 변환 스캐너 | Canny · Contour · Perspective Transform |
| **[Scratch Search & TF-IDF](NLP_Basics/A4-1/README.md)** | 수식 기반 TF-IDF 벡터라이저와 코사인 유사도 문서 검색기 | Vectorization · TF-IDF · Cosine Similarity |
| **[Explainable Credit Scoring](Machine_Learning/A5-2/README.md)** | 대출 심사 ML 분류/회귀 파이프라인 및 SHAP 기반 의사결정 설명(XAI) | Tabular ML · SHAP · PCA Clustering · XAI |

> 각 프로젝트 링크를 클릭하면 구현 과정과 기술적 내용을 확인할 수 있습니다.

---

## Curriculum & Missions

| 단계 | 미션 코드 | 미션 명 | 학습 시간 | 핵심 기술 스택 |
| :--- | :---: | :--- | :---: | :--- |
| **Data Analysis** | [A1-1](Data_Analysis/A1-1/README.md) | 쇼핑몰에서 누가 자주 오고 많이 사는지 분석해서 단골 찾기 | 60h | Multimodal Preprocessing · NumPy · Pandas EDA · RFM Segmentation |
| **Math for AI** | [A2-1](Math_for_AI/A2-1/README.md) | AI가 어떻게 학습하는지 수학으로 직접 풀어보기 | 60h | Linear Algebra · SVD · Calculus · Gradient Descent · Backprop Derivation |
| **Computer Vision** | [A3-1](Computer_Vision/A3-1/README.md) | 휴대폰으로 찍은 종이를 똑바르게 펴서 스캔하게 만들기 | 60h | OpenCV · Edge/Contour · Perspective Transform · Failure Analysis |
| | [A3-2](Computer_Vision/A3-2/README.md) | 영상 속 움직이는 사람·물건을 따라가며 표시해주기 | 80h | Motion Detection · Background Subtraction · Object Tracking · Feature Matching |
| **NLP Basics** | [A4-1](NLP_Basics/A4-1/README.md) | 원하는 내용이 들어있는 문서를 똑똑하게 찾아주는 검색기 만들기 | 70h | Tokenization · TF-IDF from scratch · Cosine Similarity · Text Classifier |
| | [A4-2](NLP_Basics/A4-2/README.md) | 글 속에 숨은 정보와 기분(좋음/나쁨)을 자동으로 뽑아내기 | 70h | Regex Entity Extraction · Lexicon Sentiment Analysis · Rule Evaluation |
| **Machine Learning**| [A5-1](Machine_Learning/A5-1/README.md) | 대출을 해줘도 될지 AI가 대신 판단해주는 시스템 만들기 | 80h | Baseline vs ML · Ridge/Lasso · Random Forest/XGBoost · ROC-AUC/F1 |
| | [A5-2](Machine_Learning/A5-2/README.md) | AI가 왜 그런 결정을 했는지 이유를 사람에게 설명해주게 하기 | 80h | K-Means · PCA · Persona · SHAP Summary/Waterfall Plot · Global/Local XAI |
| **Deep Learning** | [A6-1](Deep_Learning/A6-1/README.md) | AI의 속 엔진(두뇌)을 내 손으로 직접 만들어보기 | 80h | Scratch Framework (Tensor, Autograd, Layer, Optimizer) · XOR · MNIST |
| | [A6-2](Deep_Learning/A6-2/README.md) | AI가 어디서 자꾸 틀리는지 찾아내서 더 똑똑하게 만들기 | 80h | Transfer Learning · RNN/Time-Series · Bias-Variance · Error Analysis |
| **Term Project** | [A7-1](Term_Project/A7-1/README.md) | CV, NLP 자율 주제 프로젝트 | 80h | End-to-End AI Service · Full-stack / API Deployment · Capstone |
