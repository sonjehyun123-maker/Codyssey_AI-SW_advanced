"""
A1-1 객체지향(OOP) 데이터 분석 파이프라인 모듈
DataAnalyzer Class:
- load_data(): 거래 CSV 및 이미지 NumPy 배열 로드
- extract_multimodal_features(): NumPy 벡터 연산 기반 이미지 특성(평균, 표준편차) 및 텍스트 피처 추출
- handle_missing_values(): 그룹별(GroupBy) 통계 기반 결측치 대치
- handle_missing(): 요구서 표기와의 호환용 alias (handle_missing_values로 위임)
- detect_outliers(): IQR 기반 이상치 탐지 로직 직접 구현
- calculate_rfm(): 최종 거래일 기준 고객별 RFM 스코어 계산 및 세그먼트 분류
"""

import os
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd


class DataAnalyzer:
    """이커머스 멀티모달 EDA 및 RFM 고객 세분화 분석 파이프라인 클래스"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.df: Optional[pd.DataFrame] = None
        self.images: Optional[np.ndarray] = None
        self.rfm_df: Optional[pd.DataFrame] = None
        self.outlier_indices: Dict[str, np.ndarray] = {}
        self.imputation_stats: Dict[str, Any] = {}

    def load_data(
        self,
        csv_path: str,
        img_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """거래 데이터 CSV 및 상품 이미지 NumPy 배열을 로드합니다.

        Args:
            csv_path: 이커머스 거래 데이터 CSV 파일 경로
            img_path: 상품 이미지 NumPy .npy 파일 경로 (선택)

        Returns:
            Tuple[pd.DataFrame, Optional[np.ndarray]]: 로드된 데이터프레임과 이미지 배열
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

        self.df = pd.read_csv(csv_path)
        if "order_date" in self.df.columns:
            self.df["order_date"] = pd.to_datetime(self.df["order_date"])

        if img_path and os.path.exists(img_path):
            self.images = np.load(img_path)
        else:
            self.images = None

        return self.df, self.images

    def extract_multimodal_features(self) -> pd.DataFrame:
        """멀티모달 데이터(이미지 텐서 + 상품명 텍스트)로부터 파생 피처를 벡터 연산으로 추출합니다.

        1. 이미지 피처:
           - for loop 없이 NumPy 축(axis) 연산으로 이미지별 밝기 평균 및 표준편차 산출
        2. 텍스트 피처:
           - 상품명의 문자열 길이(char_length) 및 어절 수(word_count) 파생 변수 추가

        Returns:
            pd.DataFrame: 피처가 추가된 데이터프레임
        """
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data()를 먼저 호출하세요.")

        # 1. 이미지 데이터 피처 엔지니어링 (NumPy 벡터 연산)
        if self.images is not None and "image_index" in self.df.columns:
            # images shape: (N, H, W, C)
            # axis=(1, 2, 3)으로 각 이미지 전체 픽셀의 평균과 표준편차를 한 번에 계산
            img_mean_all = np.mean(self.images, axis=(1, 2, 3))
            img_std_all = np.std(self.images, axis=(1, 2, 3))

            # R, G, B 채널별 평균도 축 연산으로 추출: axis=(1, 2) -> shape: (N, 3)
            img_channel_means = np.mean(self.images, axis=(1, 2))
            img_r_mean = img_channel_means[:, 0]
            img_g_mean = img_channel_means[:, 1]
            img_b_mean = img_channel_means[:, 2]

            # 데이터프레임의 image_index를 기준으로 벡터 매핑
            indices = self.df["image_index"].values
            self.df["img_pixel_mean"] = img_mean_all[indices]
            self.df["img_pixel_std"] = img_std_all[indices]
            self.df["img_r_mean"] = img_r_mean[indices]
            self.df["img_g_mean"] = img_g_mean[indices]
            self.df["img_b_mean"] = img_b_mean[indices]

        # 2. 텍스트 데이터 피처 엔지니어링
        if "product_name" in self.df.columns:
            self.df["product_name_len"] = self.df["product_name"].astype(str).str.len()
            self.df["product_word_count"] = self.df["product_name"].astype(str).str.split().str.len()

        return self.df

    def handle_missing_values(
        self,
        strategy: str = "group_median",
        target_col: str = "customer_age",
        group_col: str = "category"
    ) -> pd.DataFrame:
        """결측치를 탐색하고 그룹별 통계치(평균 또는 중앙값)를 기반으로 대치합니다.

        Args:
            strategy: 'group_median' 또는 'group_mean', 'mode'
            target_col: 결측치를 채울 수치형/범주형 컬럼명 (기본: 'customer_age')
            group_col: 그룹화 기준 컬럼명 (기본: 'category')

        Returns:
            pd.DataFrame: 결측치가 대치된 데이터프레임
        """
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data()를 먼저 호출하세요.")

        # 1. 범주형 결측치(customer_gender)는 최빈값(Mode) 대치
        if "customer_gender" in self.df.columns and self.df["customer_gender"].isna().sum() > 0:
            gender_mode = self.df["customer_gender"].mode()[0]
            self.df["customer_gender"] = self.df["customer_gender"].fillna(gender_mode)
            self.imputation_stats["customer_gender_mode"] = gender_mode

        # 2. 수치형 결측치(customer_age)는 그룹별(GroupBy) 통계치로 대치
        if target_col in self.df.columns and self.df[target_col].isna().sum() > 0:
            if strategy == "group_median":
                group_stats = self.df.groupby(group_col)[target_col].transform("median")
                overall_stat = self.df[target_col].median()
            elif strategy == "group_mean":
                group_stats = self.df.groupby(group_col)[target_col].transform("mean")
                overall_stat = self.df[target_col].mean()
            else:
                overall_stat = self.df[target_col].median()
                group_stats = overall_stat

            # 그룹별 통계치로 우선 채우고, 남은 결측치는 전체 대푯값으로 안전 대치
            self.df[target_col] = self.df[target_col].fillna(group_stats).fillna(overall_stat)
            self.imputation_stats[f"{target_col}_{strategy}"] = {
                "group_col": group_col,
                "strategy": strategy,
                "overall_stat": overall_stat
            }

        return self.df

    def handle_missing(self, *args, **kwargs) -> pd.DataFrame:
        """`handle_missing_values()`의 호환 alias.

        요구서·평가 기준에는 `handle_missing()`으로 표기되어 있어, 기존 이름을
        유지하면서도 같은 진입점을 제공합니다. 동작과 반환값은 완전히 동일합니다.

        Returns:
            pd.DataFrame: 결측치가 대치된 데이터프레임
        """
        return self.handle_missing_values(*args, **kwargs)

    def detect_outliers(
        self,
        column: str = "amount",
        iqr_multiplier: float = 1.5,
        action: str = "flag"
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """IQR(사분위범위) 방식을 활용한 이상치 탐지 로직을 직접 계산합니다.

        IQR = Q3 - Q1
        Lower Bound = Q1 - (iqr_multiplier * IQR)
        Upper Bound = Q3 + (iqr_multiplier * IQR)

        Args:
            column: 이상치를 탐지할 대상 수치형 컬럼명
            iqr_multiplier: IQR 가중치 (기본 1.5, 극단치 탐지 시 3.0)
            action: 'flag' (불리언 마스크 반환 및 컬럼 생성) 또는 'clip' (경계값 클리핑)

        Returns:
            Tuple[pd.Series, Dict[str, float]]: 이상치 마스크(True=이상치) 및 통계 기준값 dict
        """
        if self.df is None or column not in self.df.columns:
            raise ValueError(f"데이터프레임에 {column} 컬럼이 존재하지 않습니다.")

        series = self.df[column].dropna()
        q1 = float(np.percentile(series, 25))
        q3 = float(np.percentile(series, 75))
        iqr = q3 - q1

        lower_bound = q1 - (iqr_multiplier * iqr)
        upper_bound = q3 + (iqr_multiplier * iqr)

        is_outlier = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        self.outlier_indices[column] = np.where(is_outlier)[0]

        stats = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": int(is_outlier.sum()),
            "outlier_ratio": float(is_outlier.mean())
        }

        flag_col = f"{column}_is_outlier"
        self.df[flag_col] = is_outlier

        if action == "clip":
            clipped_col = f"{column}_clipped"
            self.df[clipped_col] = np.clip(self.df[column], lower_bound, upper_bound)

        return is_outlier, stats

    @staticmethod
    def _percentile_score(
        series: pd.Series,
        ascending: bool = True,
        q: int = 4
    ) -> pd.Series:
        """평균 순위(average rank) 기반 분위 점수를 1~q 범위로 부여합니다.

        `pd.qcut` + `rank(method='first')` 조합은 동점자를 임의 순서로 잘라
        각 점수군을 무조건 25%씩 강제 분할합니다. 그 방식은 실제 빈도 분포가
        균등하지 않아도 "상위 25%"라는 라벨을 붙이게 되어 해석이 왜곡됩니다
        (예: 1회 구매 고객이 F_score=2를 받는 경우 발생).

        본 구현은 동점자에게 동일한 평균 백분위를 부여한 뒤 분위 경계로
        점수를 매기므로, 동점자는 반드시 같은 점수를 받고 점수군 크기는
        데이터 분포를 그대로 반영합니다(25%가 아닐 수 있음).

        Args:
            series: 점수를 매길 수치형 Series
            ascending: True면 값이 클수록 높은 점수, False면 작을수록 높은 점수
            q: 점수 개수(기본 4 → 1~4점)

        Returns:
            pd.Series: 1~q 정수 점수
        """
        # rank(pct=True): 최솟값 → 1/n, 최댓값 → 1.0. 동점자는 평균 순위를 공유
        percentile = series.rank(method="average", pct=True, ascending=ascending)
        # 0.25 경계는 1점, 1.0은 q점으로 매핑되도록 올림 적용 후 클리핑
        score = np.ceil(percentile * q).clip(1, q)
        return score.astype(int)

    def calculate_rfm(
        self,
        reference_date: Optional[str] = None,
        customer_id_col: str = "customer_id",
        order_date_col: str = "order_date",
        amount_col: str = "amount"
    ) -> pd.DataFrame:
        """고객별 Recency, Frequency, Monetary를 계산하고 4개 이상의 세그먼트로 분류합니다.

        - Recency: 분석 기준일(데이터의 최종 거래일 + 1일)로부터 마지막 구매일까지의 일수 (작을수록 좋음)
        - Frequency: 총 주문 횟수 (클수록 좋음)
        - Monetary: 총 구매 금액 (클수록 좋음)
        - 점수화: 평균 순위 백분위 기반 1~4점 (`_percentile_score`). 동점자는 동일 점수를 받으며
          점수군 크기는 데이터 분포를 따릅니다(강제 25% 분할 아님).

        Args:
            reference_date: 기준일자 문자열 (YYYY-MM-DD), None일 경우 데이터 최종 거래일 + 1일
            customer_id_col: 고객 고유 ID 컬럼명
            order_date_col: 주문 일시 컬럼명
            amount_col: 결제 금액 컬럼명

        Returns:
            pd.DataFrame: 고객별 RFM 지표 및 세그먼트가 포함된 데이터프레임
        """
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data()를 먼저 호출하세요.")

        temp_df = self.df.copy()
        if not np.issubdtype(temp_df[order_date_col].dtype, np.datetime64):
            temp_df[order_date_col] = pd.to_datetime(temp_df[order_date_col])

        # 기준일 정의 (미션 요구사항: 최종 거래일 또는 분석 기준일)
        if reference_date is None:
            ref_dt = temp_df[order_date_col].max() + pd.Timedelta(days=1)
        else:
            ref_dt = pd.to_datetime(reference_date)

        # 고객별 RFM 집계
        rfm = temp_df.groupby(customer_id_col).agg(
            recency=(order_date_col, lambda dates: (ref_dt - dates.max()).days),
            frequency=(order_date_col, "count"),
            monetary=(amount_col, "sum")
        ).reset_index()

        # 평균 순위 기반 분위수 점수 1~4점 부여 (4점이 최우수)
        # Recency는 일수가 작을수록 높은 점수 → ascending=False로 역순 매핑
        rfm["R_score"] = self._percentile_score(rfm["recency"], ascending=False)
        # Frequency & Monetary는 값이 클수록 높은 점수
        rfm["F_score"] = self._percentile_score(rfm["frequency"], ascending=True)
        rfm["M_score"] = self._percentile_score(rfm["monetary"], ascending=True)

        # 종합 RFM 스코어
        rfm["RFM_Score"] = rfm["R_score"] * 100 + rfm["F_score"] * 10 + rfm["M_score"]

        # 5대 핵심 고객 세그먼트 분류 로직
        # 주의: 규칙은 위에서부터 먼저 만족하는 조건에서 확정되므로 순서가 곧 우선순위입니다.
        # 특히 `New`는 "최근성 높음 + 최초 1회 구매"이므로, 구매 이력이 있는 고객을
        # 먼저 흡수하는 `Loyal`보다 앞에 두어야 도달 가능한 세그먼트가 됩니다.
        def assign_segment(row: pd.Series) -> str:
            r = row["R_score"]
            f = row["F_score"]
            m = row["M_score"]

            if r >= 3 and f >= 3 and m >= 3:
                return "VIP"
            elif r >= 3 and row["frequency"] == 1:
                # 실제 주문 1건 = 최초 구매 고객 (F 점수가 아닌 원시 빈도로 판정)
                return "New"
            elif r >= 3 and (f >= 2 or m >= 2):
                return "Loyal"
            elif r <= 2 and (f >= 3 or m >= 3):
                return "At_Risk"
            else:
                return "Churned"

        rfm["segment"] = rfm.apply(assign_segment, axis=1)
        self.rfm_df = rfm
        return rfm


if __name__ == "__main__":
    # 파이프라인 자가 진단 실행
    import os

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(base_dir, "data", "ecommerce_transactions.csv")
    img_file = os.path.join(base_dir, "data", "product_images.npy")

    analyzer = DataAnalyzer()
    df, imgs = analyzer.load_data(csv_file, img_file)
    print(f"[1] Data Loaded: df={df.shape}, images={imgs.shape if imgs is not None else None}")

    analyzer.extract_multimodal_features()
    print(f"[2] Multimodal Extracted: img_pixel_mean={df['img_pixel_mean'].iloc[0]:.2f}, word_count={df['product_word_count'].iloc[0]}")

    analyzer.handle_missing_values(strategy="group_median")
    print(f"[3] Missing Values Handled: age_nulls={df['customer_age'].isna().sum()}, gender_nulls={df['customer_gender'].isna().sum()}")

    _, outlier_stats = analyzer.detect_outliers(column="amount", iqr_multiplier=1.5, action="clip")
    print(f"[4] Outliers Detected: {outlier_stats['outlier_count']} items ({outlier_stats['outlier_ratio']*100:.1f}%)")

    rfm_result = analyzer.calculate_rfm()
    print(f"[5] RFM Calculated: {len(rfm_result)} customers segmented:")
    print(rfm_result["segment"].value_counts().to_string())
