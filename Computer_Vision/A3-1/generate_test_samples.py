"""Generate 20 test benchmark images for A3-1 Document Scanner.

4개 조건별 각 5장 (총 20장):
1. condition1_simple_front: 단순 배경 + 정면 (5장)
2. condition2_shadows: 그림자/조명 얼룩 있음 (5장)
3. condition3_tilted_45deg: 45도 이상 심하게 기울어짐/사다리꼴 투영 (5장)
4. condition4_complex_background: 복잡한 배경 (텍스처, 여러 사각형 물체 등) (5장)
"""

import os
import cv2
import numpy as np

def create_base_document(width=400, height=550, title="RECEIPT / INVOICE"):
    """글씨와 표가 들어간 가상 영수증/문서 생성."""
    doc = np.full((height, width, 3), 250, dtype=np.uint8)
    # 테두리 라인
    cv2.rectangle(doc, (10, 10), (width - 10, height - 10), (50, 50, 50), 2)
    # 헤더
    cv2.putText(doc, title, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.line(doc, (30, 65), (width - 30, 65), (0, 0, 0), 1)

    # 본문 텍스트 라인들
    items = [
        "Item 01: Deep Learning Book    $45.00",
        "Item 02: OpenCV Vision Guide   $38.50",
        "Item 03: Python AI CookBook    $52.00",
        "Item 04: Linear Algebra Math   $30.00",
        "Item 05: Cloud Infrastructure  $60.00",
        "Subtotal:                      $225.50",
        "Tax (10%):                     $22.55",
        "Total Amount:                  $248.05",
    ]
    y = 110
    for it in items:
        cv2.putText(doc, it, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
        y += 40

    # 바코드 시뮬레이션
    for bx in range(40, width - 40, 6):
        thickness = 1 if (bx % 12 == 0) else 3
        cv2.line(doc, (bx, y + 20), (bx, y + 60), (0, 0, 0), thickness)
    return doc

def generate_benchmark_dataset(out_dir="test_images"):
    os.makedirs(out_dir, exist_ok=True)
    canvas_w, canvas_h = 800, 800

    print("Generating 20 benchmark test images...")

    # 1. 단순 배경 + 정면 (5장)
    for i in range(1, 6):
        canvas = np.full((canvas_h, canvas_w, 3), (40, 45, 50), dtype=np.uint8) # 짙은 단색 책상
        doc = create_base_document(title=f"OFFICIAL REPORT #{i}")
        dw, dh = doc.shape[1], doc.shape[0]

        # 정면 배치 (미세한 각도만)
        x_offset = 200 + (i * 5)
        y_offset = 120 + (i * 5)
        canvas[y_offset:y_offset+dh, x_offset:x_offset+dw] = doc

        path = os.path.join(out_dir, f"cond1_simple_{i}.jpg")
        cv2.imwrite(path, canvas)
        print(f"Generated: {path}")

    # 2. 그림자 있음 (5장)
    for i in range(1, 6):
        canvas = np.full((canvas_h, canvas_w, 3), (50, 55, 60), dtype=np.uint8)
        doc = create_base_document(title=f"SHADOW INVOICE #{i}")
        dw, dh = doc.shape[1], doc.shape[0]
        x_offset, y_offset = 200, 120
        canvas[y_offset:y_offset+dh, x_offset:x_offset+dw] = doc

        # 대각선 방향 강한 그림자 그라데이션 추가
        shadow_mask = np.ones((canvas_h, canvas_w), dtype=np.float32)
        for y in range(canvas_h):
            for x in range(canvas_w):
                grad = (x + y) / (canvas_w + canvas_h)
                # 그림자 강도 (i가 커질수록 더 짙은 그림자)
                shadow_mask[y, x] = max(0.2, 1.0 - (grad * 0.15 * i))

        shadowed = (canvas.astype(np.float32) * shadow_mask[:, :, np.newaxis]).astype(np.uint8)
        path = os.path.join(out_dir, f"cond2_shadow_{i}.jpg")
        cv2.imwrite(path, shadowed)
        print(f"Generated: {path}")

    # 3. 45도 이상 심하게 기울어짐 (사다리꼴 투영 왜곡) (5장)
    for i in range(1, 6):
        canvas = np.full((canvas_h, canvas_w, 3), (35, 40, 45), dtype=np.uint8)
        doc = create_base_document(title=f"PERSPECTIVE TILT #{i}")
        dh, dw = doc.shape[:2]

        src_pts = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype=np.float32)
        # 45도 이상 기울어진 사다리꼴 왜곡 꼭짓점
        skew = 40 + (i * 25)
        dst_pts = np.array([
            [150 + skew, 100],
            [650 - skew, 140 + skew],
            [720, 680],
            [80 + skew, 620]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_doc = cv2.warpPerspective(doc, M, (canvas_w, canvas_h))

        mask = cv2.warpPerspective(np.ones((dh, dw), dtype=np.uint8)*255, M, (canvas_w, canvas_h))
        canvas[mask > 0] = warped_doc[mask > 0]

        path = os.path.join(out_dir, f"cond3_tilted_{i}.jpg")
        cv2.imwrite(path, canvas)
        print(f"Generated: {path}")

    # 4. 복잡한 배경 (텍스처, 신문지, 여러 사각 박스 등) (5장)
    for i in range(1, 6):
        # 복잡한 줄무늬 및 격자 패턴의 배경
        canvas = np.random.randint(40, 160, (canvas_h, canvas_w, 3), dtype=np.uint8)
        for line_y in range(0, canvas_h, 30):
            cv2.line(canvas, (0, line_y), (canvas_w, line_y), (200, 200, 200), 2)
        # 여러 개의 방해 사각형(노트, 명함 등) 배치
        cv2.rectangle(canvas, (50, 50), (280, 220), (180, 100, 80), -1)
        cv2.rectangle(canvas, (500, 450), (750, 700), (90, 150, 120), -1)

        doc = create_base_document(title=f"COMPLEX BG #{i}")
        dh, dw = doc.shape[:2]
        src_pts = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype=np.float32)
        dst_pts = np.array([[220, 180], [580, 190], [560, 650], [240, 640]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_doc = cv2.warpPerspective(doc, M, (canvas_w, canvas_h))
        mask = cv2.warpPerspective(np.ones((dh, dw), dtype=np.uint8)*255, M, (canvas_w, canvas_h))
        canvas[mask > 0] = warped_doc[mask > 0]

        path = os.path.join(out_dir, f"cond4_complex_{i}.jpg")
        cv2.imwrite(path, canvas)
        print(f"Generated: {path}")

    print("All 20 test benchmark images successfully created!")

if __name__ == "__main__":
    generate_benchmark_dataset()
