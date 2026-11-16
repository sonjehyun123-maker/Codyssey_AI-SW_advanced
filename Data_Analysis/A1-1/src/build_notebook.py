"""A1-1 분석 리포트 노트북 빌더: 지표 산출 → 마크다운 주입 → 실제 실행 → 저장."""

import json
import os
import sys
import tempfile

import nbformat as nbf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
CSV_PATH = os.path.join(BASE_DIR, "data", "ecommerce_transactions.csv")
IMG_PATH = os.path.join(BASE_DIR, "data", "product_images.npy")
NB_DIR = os.path.join(BASE_DIR, "notebooks")
NB_PATH = os.path.join(NB_DIR, "analysis_report.ipynb")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline import DataAnalyzer  # noqa: E402

FONT_CANDIDATES = [
    "Malgun Gothic",          # Windows
    "AppleGothic",            # macOS (구버전)
    "Apple SD Gothic Neo",    # macOS
    "NanumGothic",            # Linux (fonts-nanum)
    "NanumBarunGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
]

SEGMENT_ORDER = ["VIP", "Loyal", "At_Risk", "New", "Churned"]
SEGMENT_LABEL = {
    "VIP": "VIP",
    "Loyal": "Loyal",
    "At_Risk": "At_Risk (이탈 우려)",
    "New": "New (신규)",
    "Churned": "Churned (이탈)",
}


def collect_metrics() -> dict:
    """파이프라인을 실제로 실행해 리포트에 필요한 지표를 모두 계산합니다."""
    analyzer = DataAnalyzer(random_state=42)
    df, images = analyzer.load_data(CSV_PATH, IMG_PATH)

    n_rows, n_cols_raw = df.shape

    null_age_raw = int(df["customer_age"].isna().sum())
    null_gender_raw = int(df["customer_gender"].isna().sum())

    analyzer.extract_multimodal_features()
    analyzer.handle_missing_values(strategy="group_median")
    mask, outlier_stats = analyzer.detect_outliers(column="amount", iqr_multiplier=1.5, action="clip")
    rfm = analyzer.calculate_rfm()

    n_cols_total = df.shape[1]
    total_amount = float(df["amount"].sum())
    clipped_loss_share = float((df["amount"] - df["amount_clipped"]).sum() / total_amount * 100)

    outliers = df[mask]
    if outliers.empty:
        raise ValueError(
            "IQR 기준으로 탐지된 이상치가 없어 이상치 해석 문장을 만들 수 없습니다. "
            "iqr_multiplier 조정 또는 데이터 분포를 확인하세요."
        )
    outlier_by_cat = outliers["category"].value_counts()
    top_outlier_cat = outlier_by_cat.index[0]
    top_outlier_n = int(outlier_by_cat.iloc[0])
    top_outlier_cat_total = int((df["category"] == top_outlier_cat).sum())

    category_counts = df["category"].value_counts()
    top2 = category_counts.head(2)

    monthly = df.groupby(df["order_date"].dt.to_period("M"))["amount"].sum() / 1e6

    seg = rfm.groupby("segment").agg(
        n=("customer_id", "count"),
        rec=("recency", "mean"),
        freq=("frequency", "mean"),
        mon=("monetary", "mean"),
        tot=("monetary", "sum"),
    )
    seg["share_cust"] = seg["n"] / seg["n"].sum() * 100
    seg["share_rev"] = seg["tot"] / seg["tot"].sum() * 100

    missing_segments = [name for name in SEGMENT_ORDER if name not in seg.index]
    if missing_segments:
        raise ValueError(
            f"필수 세그먼트가 생성되지 않았습니다: {missing_segments}. "
            "세그먼트 규칙이나 데이터 분포를 확인하세요(리포트 문장이 이 세그먼트를 전제로 합니다)."
        )

    segments = {}
    for name in SEGMENT_ORDER:
        if name in seg.index:
            row = seg.loc[name]
            segments[name] = {
                "n": int(row["n"]),
                "share_cust": float(row["share_cust"]),
                "rec": float(row["rec"]),
                "freq": float(row["freq"]),
                "mon": float(row["mon"]),
                "tot": float(row["tot"]),
                "share_rev": float(row["share_rev"]),
            }

    vip = segments.get("VIP")
    at_risk = segments.get("At_Risk")
    churned = segments.get("Churned")

    vip_uplift_pct = (vip["share_rev"] * 1.0 / vip["freq"]) if vip else 0.0  # VIP 빈도 +1회 시 매출 증가율
    at_risk_winback_20 = (at_risk["tot"] * 0.20 / 1e4) if at_risk else 0.0   # 재진입 20% 가정 시 방어 규모(만원)

    multi_order = df[df["quantity"] >= 2]
    return {
        "n_rows": int(n_rows),
        "n_cols_raw": int(n_cols_raw),
        "n_cols_total": int(n_cols_total),
        "n_customers": int(df["customer_id"].nunique()),
        "n_products": int(df["product_id"].nunique()),
        "n_categories": int(df["category"].nunique()),
        "n_images": int(images.shape[0]) if images is not None else 0,
        "img_hw": f"{images.shape[1]}x{images.shape[2]}" if images is not None else "-",
        "date_min": df["order_date"].min().strftime("%Y-%m-%d"),
        "date_max": df["order_date"].max().strftime("%Y-%m-%d"),
        "null_age": null_age_raw,
        "null_gender": null_gender_raw,
        "null_age_pct": null_age_raw / n_rows * 100,
        "null_gender_pct": null_gender_raw / n_rows * 100,
        "nulls_after": int(df[["customer_age", "customer_gender"]].isna().sum().sum()),
        "amt_mean": float(df["amount"].mean()),
        "amt_median": float(df["amount"].median()),
        "amt_max": float(df["amount"].max()),
        "amt_clip_mean": float(df["amount_clipped"].mean()),
        "q1": outlier_stats["q1"],
        "q3": outlier_stats["q3"],
        "iqr": outlier_stats["iqr"],
        "upper": outlier_stats["upper_bound"],
        "lower": outlier_stats["lower_bound"],
        "n_outlier": outlier_stats["outlier_count"],
        "outlier_pct": outlier_stats["outlier_ratio"] * 100,
        "clipped_loss_share": clipped_loss_share,
        "top_outlier_cat": top_outlier_cat,
        "top_outlier_n": top_outlier_n,
        "top_outlier_cat_total": top_outlier_cat_total,
        "corr_price": float(df["amount_clipped"].corr(df["unit_price"])),
        "corr_qty": float(df["amount_clipped"].corr(df["quantity"])),
        "corr_age": float(df["amount_clipped"].corr(df["customer_age"])),
        "corr_img": float(df["amount_clipped"].corr(df["img_pixel_mean"])),
        "cat1": str(top2.index[0]),
        "cat1_n": int(top2.iloc[0]),
        "cat2": str(top2.index[1]),
        "cat2_n": int(top2.iloc[1]),
        "monthly_min": float(monthly.min()),
        "monthly_max": float(monthly.max()),
        "segments": segments,
        "seg_total": int(seg["n"].sum()),
        "rec_mean": float(rfm["recency"].mean()),
        "freq_mean": float(rfm["frequency"].mean()),
        "mon_mean": float(rfm["monetary"].mean()),
        "vip_uplift_pct": vip_uplift_pct,
        "at_risk_winback_20": at_risk_winback_20,
        "vip_freq_over_churned": (vip["freq"] / churned["freq"]) if (vip and churned and churned["freq"]) else 0.0,
        "single_order_pct": float((df["quantity"] == 1).mean() * 100),
        "multi_order_pct": float((df["quantity"] >= 2).mean() * 100),
        "multi_order_avg_qty": float(multi_order["quantity"].mean()) if len(multi_order) else 0.0,
        "avg_qty": float(df["quantity"].mean()),
        "aov": float(df["amount_clipped"].mean()),
        "img_mean_min": float(df["img_pixel_mean"].min()),
        "img_mean_max": float(df["img_pixel_mean"].max()),
    }


def build_cells(m: dict) -> list:
    seg = m["segments"]
    missing = [name for name in SEGMENT_ORDER if name not in seg]
    if missing:
        raise ValueError(f"리포트 구성에 필요한 세그먼트가 없습니다: {missing}")
    vip, at_risk, new, churned = (seg[k] for k in ("VIP", "At_Risk", "New", "Churned"))

    def won(value: float) -> str:
        return f"{value:,.0f}"

    seg_table_rows = []
    for name in SEGMENT_ORDER:
        row = seg.get(name)
        if not row:
            continue
        seg_table_rows.append(
            "| **{label}** | {n}명 ({sc:.1f}%) | {rec:.1f}일 | {freq:.1f}회 | {mon}원 | {share:.1f}% |".format(
                label=SEGMENT_LABEL[name],
                n=row["n"],
                sc=row["share_cust"],
                rec=row["rec"],
                freq=row["freq"],
                mon=won(row["mon"]),
                share=row["share_rev"],
            )
        )
    seg_table = "\n".join(seg_table_rows)

    md_rows = []
    for name in SEGMENT_ORDER:
        row = seg.get(name)
        if row:
            md_rows.append(f"| {SEGMENT_LABEL[name]} | {row['n']} | {row['share_rev']:.1f} |")
    md_table = "\n".join(md_rows)

    cells = []

    cells.append(nbf.v4.new_markdown_cell(f"""# A1-1 이커머스 멀티모달 EDA 및 RFM 고객 세분화 분석 리포트

## 1. 프로젝트 개요
이커머스 거래 데이터(수치·범주·날짜·텍스트)와 상품 이미지 텐서(배열)를 결합한 멀티모달 데이터를
객체지향 `DataAnalyzer` 파이프라인으로 정제하고, 통계 분석과 RFM 고객 세분화로 비즈니스 전략을 도출합니다.

### 데이터 규모
- 거래 로그: **{m['n_rows']:,}행 × {m['n_cols_raw']}컬럼**(파생 피처 포함 {m['n_cols_total']}컬럼), 고객 {m['n_customers']}명, 상품 {m['n_products']}종
- 상품 이미지: **{m['n_images']:,}장 × {m['img_hw']} × 3(RGB) uint8 텐서**

### 분석 목표
1. **멀티모달 피처 엔지니어링**: NumPy 축(axis) 벡터 연산으로 이미지 통계량 추출 + 상품명 텍스트 특성 생성
2. **데이터 정제**: 그룹별(GroupBy) 결측치 대치 및 IQR 기반 이상치 탐지/클리핑
3. **EDA**: 히스토그램·박스플롯·막대·히트맵·산점도·라인차트 6종 시각화
4. **RFM 세분화**: 5개 고객군(VIP·Loyal·At_Risk·New·Churned) 발굴

> 본 노트북의 해석 문장에 등장하는 모든 수치는 `src/build_notebook.py`가 파이프라인을 실제 실행해
> 산출한 값이 자동 주입된 것입니다(수기 입력 아님)."""))

    cells.append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

FONT_CANDIDATES = [
    "Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo",
    "NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR", "Noto Sans KR",
]
installed_fonts = {f.name for f in fm.fontManager.ttflist}
font_name = next((c for c in FONT_CANDIDATES if c in installed_fonts), "DejaVu Sans")

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False

sys.path.append(os.path.abspath("../src"))
from pipeline import DataAnalyzer

print(f"pandas {pd.__version__} / matplotlib {matplotlib.__version__} / seaborn {sns.__version__}")
print(f"시각화 한글 폰트: {font_name}")
if font_name == "DejaVu Sans":
    print("경고: 한글 폰트를 찾지 못했습니다. 차트의 한글이 네모로 보이면 폰트를 설치하세요.")"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. 데이터 로드 및 기본 탐색
`DataAnalyzer.load_data()`로 CSV와 이미지 배열을 함께 로드하고, `head()`·`info()`·`describe()`로
구조와 결측치 현황을 확인합니다."""))

    cells.append(nbf.v4.new_code_cell("""analyzer = DataAnalyzer(random_state=42)
df, images = analyzer.load_data("../data/ecommerce_transactions.csv", "../data/product_images.npy")

print(f"거래 데이터: {df.shape}")
print(f"이미지 텐서: {images.shape if images is not None else None} / dtype: {images.dtype if images is not None else None}")
print(f"기간: {df['order_date'].min()} ~ {df['order_date'].max()}")
df.head(5)"""))

    cells.append(nbf.v4.new_code_cell("""df.info()"""))
    cells.append(nbf.v4.new_code_cell("""df.describe().T"""))

    cells.append(nbf.v4.new_markdown_cell(f"""### 기본 구조 해석
- **규모**: 원본 {m['n_rows']:,}행 × {m['n_cols_raw']}컬럼, 고객 {m['n_customers']}명, 상품 {m['n_products']}종
  (멀티모달·텍스트 피처를 추출하면 {m['n_cols_total']}컬럼이 됩니다).
  기간은 {m['date_min']} ~ {m['date_max']}입니다.
- **결측치**: `customer_age` {m['null_age']}건({m['null_age_pct']:.1f}%),
  `customer_gender` {m['null_gender']}건({m['null_gender_pct']:.1f}%)이 존재합니다.
- **결제 금액 분포**: 평균 {won(m['amt_mean'])}원, 중앙값 {won(m['amt_median'])}원, 최댓값 {won(m['amt_max'])}원.
  평균이 중앙값의 {m['amt_mean'] / m['amt_median']:.2f}배로 **오른쪽으로 긴 꼬리(right-skewed)** 분포입니다.
- **시사점**: 평균이 중앙값을 크게 웃도는 형태는 소수 대형 결제가 평균을 끌어올린다는 뜻이므로,
  대푯값으로 중앙값을 함께 보고하고 IQR 기반 경계 처리가 필요합니다. 나이 결측치는 카테고리별 구매 성향이
  다르므로 그룹별 중앙값 대치가 적절합니다."""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. 멀티모달 피처 엔지니어링
- **이미지**: `for` 루프 없이 NumPy 축 연산으로 이미지별 픽셀 평균/표준편차, R·G·B 채널 평균을 계산합니다.
- **텍스트**: 상품명의 문자 수(`product_name_len`)와 어절 수(`product_word_count`)를 파생 변수로 만듭니다."""))

    cells.append(nbf.v4.new_code_cell("""df_feat = analyzer.extract_multimodal_features()
df_feat[["product_name", "product_name_len", "product_word_count",
         "img_pixel_mean", "img_pixel_std", "img_r_mean", "img_g_mean", "img_b_mean"]].head(5)"""))

    cells.append(nbf.v4.new_markdown_cell(f"""### 멀티모달 피처 해석
- 이미지 픽셀 평균은 {m['img_mean_min']:.1f} ~ {m['img_mean_max']:.1f} 구간에 분포합니다.
- **주의**: 이 데이터셋의 이미지는 카테고리별 대표 색상에 노이즈를 더해 합성되었고 `image_index`가 행 인덱스와
  1:1로 매핑됩니다. 따라서 `img_pixel_*` 피처는 **이미지 내용 정보라기보다 카테고리의 대리변수**에 가깝습니다.
  실제 서비스라면 상품 이미지에서 CNN 임베딩을 추출해 넣어야 이 피처가 독립적인 정보를 갖습니다."""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. 결측치 및 이상치 처리
- **결측치**: 성별은 최빈값, 나이는 `category`별 중앙값(`group_median`)으로 대치합니다.
- **이상치**: 결제 금액에 IQR(Tukey's fence) 방식을 직접 적용하고, `np.clip`으로 경계값에 눌러 담습니다(행 삭제 없음)."""))

    cells.append(nbf.v4.new_code_cell("""print("대치 전 결측치:")
print(df[["customer_gender", "customer_age"]].isna().sum())

analyzer.handle_missing_values(strategy="group_median", target_col="customer_age", group_col="category")
print(f"\\n대치 후 결측치 합계: {df[['customer_gender', 'customer_age']].isna().sum().sum()}")"""))

    cells.append(nbf.v4.new_code_cell("""is_outlier, outlier_stats = analyzer.detect_outliers(column="amount", iqr_multiplier=1.5, action="clip")
print(f"Q1 = {outlier_stats['q1']:,.0f}원, Q3 = {outlier_stats['q3']:,.0f}원, IQR = {outlier_stats['iqr']:,.0f}원")
print(f"정상 범위: [{outlier_stats['lower_bound']:,.0f}, {outlier_stats['upper_bound']:,.0f}]원")
print(f"탐지된 이상치: {outlier_stats['outlier_count']}건 ({outlier_stats['outlier_ratio']*100:.2f}%)")

print("\\n이상치 카테고리 분포:")
print(df.loc[is_outlier, "category"].value_counts().to_string())"""))

    cells.append(nbf.v4.new_markdown_cell(f"""### 결측치·이상치 처리 해석
- **결측치**: 대치 후 결측치는 {m['nulls_after']}건입니다. 나이는 상품 카테고리별 중앙값으로 채워
  카테고리별 구매 연령대 차이를 보존했습니다.
- **이상치 규모**: Q1 {won(m['q1'])}원, Q3 {won(m['q3'])}원, IQR {won(m['iqr'])}원이며 상한은 **{won(m['upper'])}원**입니다.
  이를 넘는 결제는 **{m['n_outlier']}건({m['outlier_pct']:.1f}%)**입니다.
- **탐지 결과의 성격**: 이상치 중 가장 많은 카테고리는 **{m['top_outlier_cat']} {m['top_outlier_n']}건**입니다
  (해당 카테고리 전체 {m['top_outlier_cat_total']}건 중 {m['top_outlier_n'] / m['top_outlier_cat_total'] * 100:.0f}%).
  단가가 높은 카테고리는 구조적으로 상한을 넘기므로, 이 {m['n_outlier']}건은 "전부 오류값"이 아니라
  **고가 상품의 정상 결제가 다수 포함된 집합**으로 읽어야 합니다.
- **비용**: 클리핑으로 줄어드는 금액은 전체 결제액의 **{m['clipped_loss_share']:.1f}%**입니다.
  분포를 안정화하는 대신 매출 규모는 축소되므로, 매출 집계에는 원본 `amount`를, 분포·상관 분석에는
  `amount_clipped`를 사용합니다(본 리포트의 RFM Monetary는 원본 `amount` 기준)."""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. 정밀 통계 분석
주요 수치형 변수의 평균·중앙값·표준편차·사분위수를 계산하고, 피어슨 상관계수로 변수 간 관계를 확인합니다."""))

    cells.append(nbf.v4.new_code_cell("""stat_cols = ["amount", "amount_clipped", "unit_price", "quantity",
             "customer_age", "img_pixel_mean", "product_name_len"]
summary_stats = pd.DataFrame({
    "Mean": df[stat_cols].mean(),
    "Median": df[stat_cols].median(),
    "Std": df[stat_cols].std(),
    "Q1": df[stat_cols].quantile(0.25),
    "Q3": df[stat_cols].quantile(0.75),
    "IQR": df[stat_cols].quantile(0.75) - df[stat_cols].quantile(0.25),
})
summary_stats.round(2)"""))

    cells.append(nbf.v4.new_code_cell("""corr_matrix = df[stat_cols].corr()

corr_df = pd.DataFrame(
    [("amount_clipped", "unit_price", corr_matrix.loc["amount_clipped", "unit_price"]),
     ("amount_clipped", "quantity", corr_matrix.loc["amount_clipped", "quantity"]),
     ("customer_age", "amount_clipped", corr_matrix.loc["customer_age", "amount_clipped"]),
     ("img_pixel_mean", "unit_price", corr_matrix.loc["img_pixel_mean", "unit_price"])],
    columns=["변수 1", "변수 2", "피어슨 상관계수"],
)
corr_df.round(4)"""))

    cells.append(nbf.v4.new_markdown_cell(f"""### 통계·상관 분석 해석
- `amount_clipped` ↔ `unit_price`: **{m['corr_price']:.2f}**, `amount_clipped` ↔ `quantity`: **{m['corr_qty']:.2f}**.
- `customer_age` ↔ `amount_clipped`: **{m['corr_age']:.2f}**로 사실상 무상관입니다. 연령대에 따라 결제 규모가
  달라지지 않으므로 연령 기반 타기팅보다 구매 이력 기반 타기팅이 근거가 강합니다.
- `img_pixel_mean` ↔ `unit_price`: {m['corr_img']:.2f}. 앞서 밝힌 대로 이미지가 카테고리 색상의 대리변수이므로
  이 값은 이미지 "내용"의 효과가 아니라 **카테고리 구성 차이**를 반영합니다.
- **해석 주의**: 결제 금액은 정의상 `unit_price × quantity`(에 더해 생성기에서 주입된 일부 오류 배수)이므로
  두 상관계수는 **구조적으로 양(+)이 나올 수밖에 없는 값**입니다. 이를 "발견"으로 제시하기보다
  "객단가를 올리는 레버는 단가와 수량 두 축"이라는 **정의 수준의 결론**으로 다루는 것이 정확합니다.""" ))

    cells.append(nbf.v4.new_markdown_cell("""## 6. 탐색적 데이터 분석 6종 시각화
모든 차트는 제목과 x·y축 레이블을 명시합니다.
1. 히스토그램 — 결제 금액/연령 분포
2. 박스플롯 — 이상치 처리 전후 비교
3. 막대그래프 — 카테고리별 주문 빈도
4. 히트맵 — 상관행렬
5. 산점도 — 단가 vs 결제 금액
6. 라인차트 — 월별 매출 추이"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df["amount_clipped"], bins=30, kde=True, ax=axes[0], color="#2b5c8f")
axes[0].set_title("결제 금액 분포 (IQR 클리핑 후)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("결제 금액 (원)")
axes[0].set_ylabel("주문 건수")

sns.histplot(df["customer_age"], bins=20, kde=True, ax=axes[1], color="#e27c38")
axes[1].set_title("고객 연령 분포", fontsize=13, fontweight="bold")
axes[1].set_xlabel("고객 나이 (세)")
axes[1].set_ylabel("주문 건수")

plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(y=df["amount"], ax=axes[0], color="#d95f02")
axes[0].set_title("처리 전 결제 금액 (Raw)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("결제 금액 (원)")

sns.boxplot(y=df["amount_clipped"], ax=axes[1], color="#1b9e77")
axes[1].set_title("IQR 클리핑 후 결제 금액", fontsize=13, fontweight="bold")
axes[1].set_ylabel("결제 금액 (원)")

plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""cat_order = df["category"].value_counts()
plt.figure(figsize=(9, 4.5))
sns.barplot(x=cat_order.index, y=cat_order.values, hue=cat_order.index, palette="Blues_r", legend=False)
plt.title("상품 카테고리별 주문 빈도", fontsize=13, fontweight="bold")
plt.xlabel("상품 카테고리")
plt.ylabel("주문 건수")
for i, v in enumerate(cat_order.values):
    plt.text(i, v + 5, f"{v}건", ha="center", fontweight="bold")
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
plt.title("주요 수치형·멀티모달 변수 상관행렬", fontsize=13, fontweight="bold")
plt.xlabel("피처")
plt.ylabel("피처")
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(9, 5))
sns.scatterplot(data=df, x="unit_price", y="amount_clipped", hue="category", alpha=0.7, s=40, palette="tab10")
plt.title("상품 단가 vs 결제 금액 (카테고리별)", fontsize=13, fontweight="bold")
plt.xlabel("상품 단가 (원)")
plt.ylabel("결제 금액 (원)")
plt.legend(title="카테고리", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
monthly_sales = df.groupby("order_month")["amount"].sum() / 1e6

plt.figure(figsize=(10, 4.5))
plt.plot(monthly_sales.index, monthly_sales.values, marker="o", linewidth=2, color="#d95f02")
plt.title("2025년 월별 총 매출 추이", fontsize=13, fontweight="bold")
plt.xlabel("주문 연월")
plt.ylabel("총 매출액 (백만 원)")
plt.xticks(rotation=45)
for i, val in enumerate(monthly_sales.values):
    plt.text(i, val + 0.3, f"{val:.1f}M", ha="center", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell(f"""### 시각화 해석
- **카테고리 구성**: {m['cat1']} {m['cat1_n']}건, {m['cat2']} {m['cat2_n']}건 순으로 많고 {m['n_categories']}개 카테고리가
  고르게 분포합니다(합계 {m['n_rows']:,}건).
- **결제 금액**: 클리핑 전에는 최댓값 {won(m['amt_max'])}원의 극단값이 축을 지배하지만, 클리핑 후에는
  상한 {won(m['upper'])}원에서 눌린 분포로 바뀌어 나머지 구간의 형태를 읽을 수 있습니다.
- **월별 매출**: 원본 `amount` 기준 월 {m['monthly_min']:.1f}백만 ~ {m['monthly_max']:.1f}백만 원 범위로,
  뚜렷한 추세 없이 연중 유사한 수준을 유지합니다. 이는 합성 데이터가 날짜를 균등 분포로 생성했기 때문이며,
  실데이터라면 계절성(프로모션·명절) 분석이 필요합니다.
- **산점도**: 단가와 결제 금액이 계단형 띠를 이루는 것은 수량이 1~5(드물게 10)의 이산값이기 때문입니다.""" ))

    cells.append(nbf.v4.new_markdown_cell("""## 7. RFM 기반 고객 세분화
- 기준일: 데이터 최종 거래일 **+1일**
- R(최근성) = 기준일 − 마지막 구매일 (작을수록 우수), F(빈도) = 주문 건수, M(금액) = 누적 결제액
- 점수화: 평균 순위 백분위 기반 **1~4점**. 동점자는 동일 점수를 받고, 점수군 크기는 데이터 분포를 따릅니다
  (강제 25% 분할 아님).
- 세그먼트 규칙(위에서부터 먼저 만족하는 조건에서 확정):
  1. `VIP`: R≥3 & F≥3 & M≥3
  2. `New`: R≥3 & 주문 1건 (최초 구매 고객) — `Loyal`보다 **먼저** 판정
  3. `Loyal`: R≥3 & (F≥2 | M≥2)
  4. `At_Risk`: R≤2 & (F≥3 | M≥3)
  5. `Churned`: 그 외"""))

    cells.append(nbf.v4.new_code_cell("""rfm_df = analyzer.calculate_rfm()
rfm_df.head(10)"""))

    cells.append(nbf.v4.new_code_cell("""score_dist = pd.DataFrame({
    "R_score": rfm_df["R_score"].value_counts().sort_index(),
    "F_score": rfm_df["F_score"].value_counts().sort_index(),
    "M_score": rfm_df["M_score"].value_counts().sort_index(),
}).fillna(0).astype(int)
score_dist"""))

    cells.append(nbf.v4.new_code_cell("""segment_summary = rfm_df.groupby("segment").agg(
    customer_count=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean"),
    total_monetary=("monetary", "sum"),
).reset_index()
segment_summary["customer_share"] = segment_summary["customer_count"] / segment_summary["customer_count"].sum() * 100
segment_summary["monetary_share"] = segment_summary["total_monetary"] / segment_summary["total_monetary"].sum() * 100
segment_summary.sort_values("total_monetary", ascending=False).round(1)"""))

    cells.append(nbf.v4.new_code_cell("""order = ["VIP", "Loyal", "At_Risk", "New", "Churned"]
plot_df = segment_summary.set_index("segment").loc[order].reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=plot_df, x="segment", y="customer_count", hue="segment", palette="viridis", legend=False, ax=axes[0])
axes[0].set_title("RFM 세그먼트별 고객 수", fontsize=13, fontweight="bold")
axes[0].set_xlabel("고객 세그먼트")
axes[0].set_ylabel("고객 수 (명)")
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height())}명", (p.get_x() + p.get_width() / 2, p.get_height() + 1),
                     ha="center", fontweight="bold")

sns.barplot(data=plot_df, x="segment", y="monetary_share", hue="segment", palette="magma", legend=False, ax=axes[1])
axes[1].set_title("RFM 세그먼트별 매출 기여도", fontsize=13, fontweight="bold")
axes[1].set_xlabel("고객 세그먼트")
axes[1].set_ylabel("매출 기여율 (%)")
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height() + 0.5),
                     ha="center", fontweight="bold")

plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell(f"""### RFM 세분화 결과

| 세그먼트 | 고객 수 | 평균 Recency | 평균 Frequency | 평균 Monetary | 매출 기여도 |
| :-- | :--: | :--: | :--: | :--: | :--: |
{seg_table}
| **전체** | **{m['seg_total']}명** | **{m['rec_mean']:.1f}일** | **{m['freq_mean']:.1f}회** | **{won(m['mon_mean'])}원** | **100.0%** |

- **{SEGMENT_LABEL['VIP']}**: {vip['n']}명({vip['share_cust']:.1f}%)이 전체 매출의 **{vip['share_rev']:.1f}%**를 차지합니다.
  평균 구매 {vip['freq']:.1f}회, 1인 누적 {won(vip['mon'])}원으로, {SEGMENT_LABEL['Churned']}의 평균 구매
  {churned['freq']:.1f}회 대비 **{m['vip_freq_over_churned']:.1f}배**입니다.
- **{SEGMENT_LABEL['At_Risk']}**: {at_risk['n']}명({at_risk['share_cust']:.1f}%)이 매출의 {at_risk['share_rev']:.1f}%를
  차지하는 고가치군이지만 평균 Recency가 {at_risk['rec']:.1f}일로, VIP({vip['rec']:.1f}일)의 약
  {at_risk['rec'] / vip['rec']:.1f}배입니다. 과거 가치가 높고 최근성이 낮은 전형적인 윈백 대상입니다.
- **{SEGMENT_LABEL['New']}**: {new['n']}명({new['share_cust']:.1f}%)이 매출의 {new['share_rev']:.1f}%로 아직 규모는 작지만,
  평균 Recency {new['rec']:.1f}일로 최근 유입된 고객입니다. 2차 구매 전환이 KPI입니다.
- **{SEGMENT_LABEL['Churned']}**: {churned['n']}명({churned['share_cust']:.1f}%)이 매출의 {churned['share_rev']:.1f}%를
  차지하며 평균 Recency {churned['rec']:.1f}일로 가장 오래 방문하지 않았습니다."""))

    cells.append(nbf.v4.new_markdown_cell(f"""## 8. 정리와 한계

### 세그먼트별 매출 기여 요약
| 세그먼트 | 고객 수 | 매출 기여율(%) |
| :-- | :--: | :--: |
{md_table}

### 분석의 한계 (먼저 밝혀야 할 것들)
1. **데이터가 합성**입니다. 구매 가중치·결측·이상치를 설계해 주입했으므로, 세그먼트 비율은 시장의 실측치가
   아니라 **파이프라인 검증용 분포**입니다.
2. **이미지 피처는 카테고리 대리변수**입니다(합성 이미지 + 행 인덱스 매핑). 멀티모달 융합의 효과를 주장하려면
   실제 상품 이미지에서 추출한 임베딩이 필요합니다.
3. **클리핑 기준이 두 갈래**입니다. 분포·상관 분석은 `amount_clipped`, 매출·RFM Monetary는 원본 `amount`를
   사용합니다. 의도된 선택이지만 두 수치를 같은 문장에서 비교하면 안 됩니다.
4. **인과가 아닙니다.** 세그먼트와 매출의 관계는 관측적 연관이며, 프로모션 효과는 A/B 테스트로 검증해야 합니다.
5. **시계열 구조가 없습니다.** 날짜를 균등 분포로 생성해 계절성·추세가 없으므로 월별 매출 차이는 노이즈입니다."""))

    return cells


def _resolve_kernel_name() -> str:
    """현재 파이썬 환경에서 사용 가능한 ipykernel 커널 이름을 반환합니다."""
    from jupyter_client.kernelspec import KernelSpecManager

    if "python3" in KernelSpecManager().find_kernel_specs():
        return "python3"

    tmp_dir = tempfile.mkdtemp(prefix="a1-1-kernel-")
    spec_dir = os.path.join(tmp_dir, "kernels", "python3")
    os.makedirs(spec_dir, exist_ok=True)
    with open(os.path.join(spec_dir, "kernel.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "A1-1 (current python)",
                "language": "python",
            },
            f,
        )
    os.environ["JUPYTER_PATH"] = tmp_dir + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    return "python3"


def execute_notebook(nb_path: str, nb_dir: str, timeout: int = 900) -> None:
    """노트북을 실제 실행해 출력(이미지/표/로그)이 포함된 상태로 저장합니다."""
    from nbclient import NotebookClient

    nb = nbf.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=_resolve_kernel_name(),
        resources={"metadata": {"path": nb_dir}},
        allow_errors=False,
    )
    client.execute()
    nbf.write(nb, nb_path)


def create_and_execute_notebook() -> str:
    """지표 산출 → 셀 구성(수치 주입) → 노트북 실행 → 저장까지 수행합니다."""
    os.makedirs(NB_DIR, exist_ok=True)

    metrics = collect_metrics()
    cells = build_cells(metrics)

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nbf.write(nb, NB_PATH)
    print(f"노트북 셀 구성 완료: {NB_PATH} (code {sum(c['cell_type'] == 'code' for c in cells)} / "
          f"markdown {sum(c['cell_type'] == 'markdown' for c in cells)})")

    execute_notebook(NB_PATH, NB_DIR)
    print("노트북 실행 완료: 모든 셀이 오류 없이 실행되어 출력이 저장되었습니다.")

    print("\n[README 동기화용 핵심 지표]")
    print(json.dumps(_readme_metrics(metrics), ensure_ascii=False, indent=2))
    return NB_PATH


def _readme_metrics(m: dict) -> dict:
    """README 표에 옮겨 적을 핵심 지표만 추립니다."""
    return {
        "segments": {
            k: {
                "n": v["n"],
                "share_cust": round(v["share_cust"], 1),
                "recency": round(v["rec"], 1),
                "frequency": round(v["freq"], 1),
                "monetary": int(round(v["mon"])),
                "share_rev": round(v["share_rev"], 1),
            }
            for k, v in m["segments"].items()
        },
        "overall": {
            "n": m["seg_total"],
            "recency": round(m["rec_mean"], 1),
            "frequency": round(m["freq_mean"], 1),
            "monetary": int(round(m["mon_mean"])),
        },
        "amount": {"mean": int(m["amt_mean"]), "median": int(m["amt_median"]), "max": int(m["amt_max"])},
        "iqr": {"q1": int(m["q1"]), "q3": int(m["q3"]), "iqr": int(m["iqr"]), "upper": int(m["upper"]),
                "n_outlier": m["n_outlier"], "pct": round(m["outlier_pct"], 1),
                "clipped_loss_share": round(m["clipped_loss_share"], 1)},
        "corr": {"price": round(m["corr_price"], 3), "qty": round(m["corr_qty"], 3), "age": round(m["corr_age"], 3)},
        "categories": {m["cat1"]: m["cat1_n"], m["cat2"]: m["cat2_n"]},
        "scenarios": {
            "vip_uplift_pct": round(
                round(m["segments"]["VIP"]["share_rev"], 1) / round(m["segments"]["VIP"]["freq"], 1), 2
            ) if "VIP" in m["segments"] else 0.0,
            "at_risk_winback_20man": int(round(m["at_risk_winback_20"])),
            "single_order_pct": round(m["single_order_pct"], 1),
            "multi_order_pct": round(m["multi_order_pct"], 1),
            "avg_qty": round(m["avg_qty"], 2),
        },
    }


if __name__ == "__main__":
    create_and_execute_notebook()
