"""A1-1 이커머스 멀티모달 EDA 및 RFM 고객 세분화 파이프라인."""

import os
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd


class DataAnalyzer:
    """이커머스 거래·이미지 데이터를 정제하고 RFM 세그먼트를 계산한다."""

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
        """거래 CSV와 이미지 배열을 로드한다."""
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
        """이미지 통계량과 상품명 텍스트 피처를 파생 변수로 추가한다."""
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data()를 먼저 호출하세요.")

        if self.images is not None and "image_index" in self.df.columns:
            if self.images.ndim != 4:
                raise ValueError(
                    f"이미지 배열은 (N, H, W, C) 4차원이어야 합니다. 현재 차원: {self.images.ndim}"
                )

            image_index = self.df["image_index"]
            if image_index.isna().any():
                raise ValueError(f"image_index에 결측치가 {int(image_index.isna().sum())}건 있습니다.")
            if not np.issubdtype(image_index.dtype, np.integer):
                raise ValueError(f"image_index는 정수형이어야 합니다. 현재 dtype: {image_index.dtype}")

            n_images = self.images.shape[0]
            if image_index.min() < 0 or image_index.max() >= n_images:
                raise ValueError(
                    f"image_index 범위를 벗어났습니다: [{image_index.min()}, {image_index.max()}], "
                    f"허용 범위는 [0, {n_images - 1}]입니다."
                )

            img_mean_all = np.mean(self.images, axis=(1, 2, 3))
            img_std_all = np.std(self.images, axis=(1, 2, 3))

            img_channel_means = np.mean(self.images, axis=(1, 2))
            img_r_mean = img_channel_means[:, 0]
            img_g_mean = img_channel_means[:, 1]
            img_b_mean = img_channel_means[:, 2]

            indices = self.df["image_index"].values
            self.df["img_pixel_mean"] = img_mean_all[indices]
            self.df["img_pixel_std"] = img_std_all[indices]
            self.df["img_r_mean"] = img_r_mean[indices]
            self.df["img_g_mean"] = img_g_mean[indices]
            self.df["img_b_mean"] = img_b_mean[indices]

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
        """범주형은 최빈값, 수치형은 그룹별 통계치로 결측치를 대치한다."""
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data()를 먼저 호출하세요.")

        valid_strategies = {"group_median", "group_mean", "mode"}
        if strategy not in valid_strategies:
            raise ValueError(
                f"지원하지 않는 strategy입니다: {strategy!r}. 사용 가능: {sorted(valid_strategies)}"
            )

        if "customer_gender" in self.df.columns and self.df["customer_gender"].isna().sum() > 0:
            gender_mode = self.df["customer_gender"].mode()[0]
            self.df["customer_gender"] = self.df["customer_gender"].fillna(gender_mode)
            self.imputation_stats["customer_gender_mode"] = gender_mode

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

            self.df[target_col] = self.df[target_col].fillna(group_stats).fillna(overall_stat)
            self.imputation_stats[f"{target_col}_{strategy}"] = {
                "group_col": group_col,
                "strategy": strategy,
                "overall_stat": overall_stat
            }

        return self.df

    def handle_missing(self, *args, **kwargs) -> pd.DataFrame:
        """`handle_missing_values()`와 동일한 동작을 하는 alias."""
        return self.handle_missing_values(*args, **kwargs)

    def detect_outliers(
        self,
        column: str = "amount",
        iqr_multiplier: float = 1.5,
        action: str = "flag"
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """IQR 기준으로 이상치를 탐지하고 선택적으로 경계값에 클리핑한다."""
        if self.df is None or column not in self.df.columns:
            raise ValueError(f"데이터프레임에 {column} 컬럼이 존재하지 않습니다.")

        if action not in {"flag", "clip"}:
            raise ValueError(f"지원하지 않는 action입니다: {action!r}. 사용 가능: ['clip', 'flag']")

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
        """평균 순위 백분위로 1~q 점수를 부여한다(동점자는 같은 점수, 강제 분할 없음)."""
        percentile = series.rank(method="average", pct=True, ascending=ascending)
        score = np.ceil(percentile * q).clip(1, q)
        return score.astype(int)

    def calculate_rfm(
        self,
        reference_date: Optional[str] = None,
        customer_id_col: str = "customer_id",
        order_date_col: str = "order_date",
        amount_col: str = "amount",
        allow_negative_amount: bool = False
    ) -> pd.DataFrame:
        """고객별 Recency·Frequency·Monetary를 계산해 5개 세그먼트로 분류한다.

        Frequency는 주문 건수, Monetary는 금액 합계다(환불을 허용하면 순액).
        """
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data()를 먼저 호출하세요.")

        required_cols = [customer_id_col, order_date_col, amount_col]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"RFM 계산에 필요한 컬럼이 없습니다: {missing_cols}")

        temp_df = self.df.copy()
        if temp_df[order_date_col].isna().any():
            raise ValueError(
                f"{order_date_col}에 결측치가 {int(temp_df[order_date_col].isna().sum())}건 있습니다."
            )
        if temp_df[amount_col].isna().any():
            raise ValueError(f"{amount_col}에 결측치가 {int(temp_df[amount_col].isna().sum())}건 있습니다.")

        amount_values = pd.to_numeric(temp_df[amount_col], errors="coerce")
        non_finite = int((~np.isfinite(amount_values)).sum())
        if non_finite:
            raise ValueError(f"{amount_col}에 무한대/비수치 값이 {non_finite}건 있습니다.")

        negative_count = int((amount_values < 0).sum())
        if negative_count and not allow_negative_amount:
            raise ValueError(
                f"{amount_col}에 음수 {negative_count}건이 있습니다(환불로 간주). "
                "순매출 기준으로 계산하려면 allow_negative_amount=True를 사용하세요."
            )

        if not np.issubdtype(temp_df[order_date_col].dtype, np.datetime64):
            temp_df[order_date_col] = pd.to_datetime(temp_df[order_date_col])

        if reference_date is None:
            ref_dt = temp_df[order_date_col].max() + pd.Timedelta(days=1)
        else:
            ref_dt = pd.to_datetime(reference_date)

        temp_df = temp_df[temp_df[order_date_col] < ref_dt]

        rfm = temp_df.groupby(customer_id_col).agg(
            recency=(order_date_col, lambda dates: (ref_dt - dates.max()).days),
            frequency=(order_date_col, "count"),
            monetary=(amount_col, "sum")
        ).reset_index()

        rfm["R_score"] = self._percentile_score(rfm["recency"], ascending=False)
        rfm["F_score"] = self._percentile_score(rfm["frequency"], ascending=True)
        rfm["M_score"] = self._percentile_score(rfm["monetary"], ascending=True)

        rfm["RFM_Score"] = rfm["R_score"] * 100 + rfm["F_score"] * 10 + rfm["M_score"]

        def assign_segment(row: pd.Series) -> str:
            r = row["R_score"]
            f = row["F_score"]
            m = row["M_score"]

            if r >= 3 and f >= 3 and m >= 3:
                return "VIP"
            elif r >= 3 and row["frequency"] == 1:
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
