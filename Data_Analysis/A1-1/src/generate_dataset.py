"""A1-1 멀티모달 이커머스 데이터셋 생성기."""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_multimodal_dataset(n_samples=1500, random_seed=42):
    np.random.seed(random_seed)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    n_customers = 350
    customer_ids = [f"CUST_{i:04d}" for i in range(1, n_customers + 1)]
    genders = np.random.choice(["Male", "Female"], size=n_customers, p=[0.48, 0.52])
    ages = np.random.normal(35, 10, size=n_customers).astype(int)
    ages = np.clip(ages, 19, 70)
    customer_meta = {cid: {"gender": g, "age": a} for cid, g, a in zip(customer_ids, genders, ages)}
    
    categories = ["Fashion", "Electronics", "Beauty", "Home", "Sports"]
    cat_price_ranges = {
        "Fashion": (25000, 150000),
        "Electronics": (80000, 850000),
        "Beauty": (15000, 90000),
        "Home": (30000, 250000),
        "Sports": (40000, 300000)
    }
    
    products = [
        ("P01", "오버핏 베이직 코튼 라운드 티셔츠", "Fashion", 35000),
        ("P02", "슬림핏 스트레치 데님 팬츠 청바지", "Fashion", 59000),
        ("P03", "헤비웨이트 기모 후드 집업", "Fashion", 78000),
        ("P04", "클래식 캐시미어 블렌드 싱글 코트", "Fashion", 189000),
        ("P05", "데일리 캔버스 스니커즈 단화", "Fashion", 45000),
        ("P06", "미니멀 레더 크로스백 숄더백", "Fashion", 92000),
        ("P07", "울 터틀넥 니트 스웨터", "Fashion", 64000),
        ("P08", "이지 린넨 밴딩 슬랙스", "Fashion", 42000),
        ("P09", "노이즈캔슬링 무선 블루투스 헤드폰", "Electronics", 299000),
        ("P10", "초경량 마그네틱 보조배터리 10000mAh", "Electronics", 39000),
        ("P11", "기계식 RGB 게이밍 적축 키보드", "Electronics", 129000),
        ("P12", "초고속 무소음 무선 버티컬 마우스", "Electronics", 55000),
        ("P13", "4K UHD 스마트 휴대용 빔프로젝터", "Electronics", 450000),
        ("P14", "스마트 LED 무선 센서 스탠드 조명", "Electronics", 48000),
        ("P15", "듀얼 드라이버 고음질 인이어 무선 이어폰", "Electronics", 119000),
        ("P16", "저자극 시카 히알루론산 수분 진정 크림", "Beauty", 28000),
        ("P17", "순수 비타민C 미백 톤업 앰플 세럼", "Beauty", 36000),
        ("P18", "소프트 매트 롱래스팅 립 틴트", "Beauty", 18000),
        ("P19", "약산성 티트리 버블 클렌징 폼", "Beauty", 15000),
        ("P20", "워터프루프 슬림 아이라이너 펜슬", "Beauty", 14000),
        ("P21", "안티에이징 펩타이드 아이크림", "Beauty", 45000),
        ("P22", "향기로운 천연 소이캔들 아로마 디퓨저 세트", "Home", 32000),
        ("P23", "초극세사 고밀도 호텔식 침구 커버 세트", "Home", 89000),
        ("P24", "친환경 대나무 원목 도마 키친웨어", "Home", 27000),
        ("P25", "스테인리스 3단 밀폐 보온 도시락통", "Home", 38000),
        ("P26", "스마트 전동 자동 센서 쓰레기통 20L", "Home", 52000),
        ("P27", "인체공학 메모리폼 경추 기능성 베개", "Home", 49000),
        ("P28", "고밀도 친환경 NBR 와이드 요가매트 10mm", "Sports", 35000),
        ("P29", "무게조절 덤벨 홈트레이닝 아령 세트", "Sports", 85000),
        ("P30", "러닝용 통기성 메쉬 기능성 스포츠 양말", "Sports", 12000),
        ("P31", "충격흡수 카본 런닝화 쿠셔닝 마라톤화", "Sports", 139000),
        ("P32", "스테인리스 진공 단열 대용량 스포츠 텀블러", "Sports", 29000),
        ("P33", "스마트 무선 줄넘기 홈트 카운팅", "Sports", 24000),
    ]
    
    records = []
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    days_span = (end_date - start_date).days
    
    image_tensors = np.zeros((n_samples, 64, 64, 3), dtype=np.uint8)
    
    for idx in range(n_samples):
        order_id = f"ORD_{idx+1:05d}"
        
        cust_weights = np.ones(n_customers)
        cust_weights[:50] = 4.0   # 상위 50명은 단골 (구매 빈도 높음)
        cust_weights[50:150] = 2.0 # 중간 고객
        cust_weights /= cust_weights.sum()
        
        cid = np.random.choice(customer_ids, p=cust_weights)
        c_info = customer_meta[cid]
        
        prod_idx = np.random.randint(0, len(products))
        pid, pname, cat, base_price = products[prod_idx]
        
        qty = np.random.choice([1, 2, 3, 4, 5, 10], p=[0.60, 0.22, 0.10, 0.05, 0.02, 0.01])
        
        rand_days = np.random.randint(0, days_span)
        rand_secs = np.random.randint(0, 86400)
        order_time = start_date + timedelta(days=rand_days, seconds=rand_secs)
        
        discount_rate = np.random.choice([1.0, 0.95, 0.9, 0.85], p=[0.5, 0.25, 0.15, 0.1])
        unit_price = int(base_price * discount_rate)
        amount = int(unit_price * qty)
        
        if np.random.rand() < 0.008:
            amount = amount * 15  # 비정상 대량 결제
        
        age_val = c_info["age"] if np.random.rand() > 0.04 else np.nan
        gender_val = c_info["gender"] if np.random.rand() > 0.02 else np.nan
        
        cat_base_color = {
            "Fashion": [180, 80, 100],
            "Electronics": [40, 90, 200],
            "Beauty": [230, 140, 160],
            "Home": [140, 110, 80],
            "Sports": [60, 190, 110]
        }[cat]
        
        img = np.full((64, 64, 3), cat_base_color, dtype=np.float32)
        noise = np.random.normal(0, 25, (64, 64, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        image_tensors[idx] = img
        
        records.append({
            "order_id": order_id,
            "customer_id": cid,
            "order_date": order_time.strftime("%Y-%m-%d %H:%M:%S"),
            "product_id": pid,
            "product_name": pname,
            "category": cat,
            "unit_price": unit_price,
            "quantity": qty,
            "amount": amount,
            "customer_gender": gender_val,
            "customer_age": age_val,
            "image_index": idx
        })
        
    df = pd.DataFrame(records)
    
    csv_path = os.path.join(data_dir, "ecommerce_transactions.csv")
    img_path = os.path.join(data_dir, "product_images.npy")
    
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    np.save(img_path, image_tensors)
    
    print(f"Data generation complete!")
    print(f"- Transactions CSV saved: {csv_path} (Shape: {df.shape})")
    print(f"- Product Images NPY saved: {img_path} (Shape: {image_tensors.shape})")
    print(f"- Total customers: {df['customer_id'].nunique()}")
    print(f"- Date range: {df['order_date'].min()} ~ {df['order_date'].max()}")
    print(f"- Missing values: age={df['customer_age'].isna().sum()}, gender={df['customer_gender'].isna().sum()}")
    return df, image_tensors

if __name__ == "__main__":
    generate_multimodal_dataset()
