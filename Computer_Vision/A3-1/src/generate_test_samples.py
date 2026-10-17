"""Generate realistic smartphone-photographed benchmark dataset (20 images).

A3-1: 휴대폰으로 찍은 종이를 똑바르게 펴서 스캔하게 만들기 20장 실사풍 테스트셋.
4가지 조건별 각 5장 (총 20장):
1. cond1_simple: 원목 책상 위, 손으로 위에서 찍은 자연스러운 미세 각도와 원근감 (5장)
2. cond2_shadow: 스탠드 조명 및 강한 그림자가 드리워진 환경 (5장)
3. cond3_tilted: 45도 이상 사선으로 비스듬히 누워 있는 원근 단축 환경 (5장)
4. cond4_complex: 책상 위에 다른 서류, 노트, 패턴 매트가 함께 있는 복잡한 배경 (5장)
"""

import os
from pathlib import Path
import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

try:
    from scanner import imwrite_unicode
except ImportError:
    from src.scanner import imwrite_unicode


def create_wood_desk_background(h: int, w: int, tone: str = "warm") -> np.ndarray:
    """나뭇결과 조명 그라데이션이 들어간 사실적인 책상 텍스처 생성."""
    if tone == "warm":
        base_color = (35, 65, 110)  # 따뜻한 브라운 원목 (BGR)
    elif tone == "dark":
        base_color = (30, 40, 50)   # 다크 월넛
    else:
        base_color = (60, 75, 90)   # 라이트 오크

    desk = np.full((h, w, 3), base_color, dtype=np.uint8)
    # 나뭇결 물결 노이즈 합성
    noise = np.random.normal(0, 5, (h, w, 1)).astype(np.float32)
    for x in range(w):
        wave = int(np.sin(x / 40.0) * 10 + np.sin(x / 9.0) * 4)
        desk[:, x] = np.clip(desk[:, x].astype(np.float32) + noise[:, x] + wave, 0, 255).astype(np.uint8)

    # 전체 카메라 비네팅(Vignetting, 가장자리 어두움) 추가
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    vignette = 1.0 - (dist_from_center / max_dist) * 0.25
    desk = (desk.astype(np.float32) * vignette[:, :, None]).astype(np.uint8)

    return desk


def create_realistic_document(doc_w: int = 440, doc_h: int = 620, title: str = "OFFICIAL INVOICE") -> np.ndarray:
    """종이 질감과 텍스트, 바코드가 담긴 현실적인 A4 영수증 문서 생성."""
    # 미세한 아이보리 톤 종이 바탕
    doc = np.full((doc_h, doc_w, 3), (245, 248, 250), dtype=np.uint8)
    # 종이 질감 미세 노이즈
    paper_noise = np.random.normal(0, 2, (doc_h, doc_w, 3)).astype(np.float32)
    doc = np.clip(doc.astype(np.float32) + paper_noise, 0, 255).astype(np.uint8)

    # 외곽 테두리
    cv2.rectangle(doc, (18, 18), (doc_w - 18, doc_h - 18), (60, 60, 60), 2)
    cv2.putText(doc, title, (36, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2)
    cv2.line(doc, (36, 82), (doc_w - 36, 82), (40, 40, 40), 1)

    items = [
        "Item 01: Deep Learning Engine       $45.00",
        "Item 02: OpenCV Computer Vision     $38.50",
        "Item 03: PyTorch Framework Core     $52.00",
        "Item 04: Linear Algebra Calculus    $30.00",
        "Item 05: Cloud Infrastructure API   $60.00",
        "-------------------------------------------",
        "Subtotal:                           $225.50",
        "Tax (10%):                          $22.55",
        "Total Amount:                       $248.05",
    ]
    y = 135
    for it in items:
        cv2.putText(doc, it, (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
        y += 42

    # 하단 바코드
    for bx in range(40, doc_w - 40, 6):
        thickness = 1 if (bx % 12 == 0) else 3
        cv2.line(doc, (bx, y + 25), (bx, y + 75), (20, 20, 20), thickness)

    return doc


def apply_perspective_with_shadow(
    background: np.ndarray,
    doc: np.ndarray,
    dst_corners: np.ndarray,
    shadow_offset: Tuple[int, int] = (15, 20),
    shadow_strength: float = 0.45,
) -> np.ndarray:
    """문서를 4점 원근 투영 변환하고, 바닥에 자연스러운 소프트 그림자를 합성."""
    canvas_h, canvas_w = background.shape[:2]
    dh, dw = doc.shape[:2]

    src_corners = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_corners, dst_corners.astype(np.float32))

    # 종이 마스크
    doc_mask = np.ones((dh, dw), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(doc_mask, M, (canvas_w, canvas_h))
    warped_doc = cv2.warpPerspective(doc, M, (canvas_w, canvas_h))

    # 그림자 좌표 생성 (살짝 아래쪽으로 오프셋)
    shadow_corners = dst_corners + np.array(shadow_offset, dtype=np.float32)
    M_shadow = cv2.getPerspectiveTransform(src_corners, shadow_corners.astype(np.float32))
    warped_shadow = cv2.warpPerspective(doc_mask, M_shadow, (canvas_w, canvas_h))
    # 가우시안 블러로 소프트 그림자 생성
    blurred_shadow = cv2.GaussianBlur(warped_shadow, (41, 41), 0).astype(np.float32) / 255.0

    # 배경에 그림자 투영
    result = background.astype(np.float32) * (1.0 - blurred_shadow[:, :, None] * shadow_strength)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 종이 합성
    result[warped_mask > 0] = warped_doc[warped_mask > 0]
    return result


def generate_benchmark_dataset(out_dir=None):
    if out_dir is None:
        out_dir = PROJECT_ROOT / "test_images"
    os.makedirs(out_dir, exist_ok=True)
    canvas_w, canvas_h = 900, 900

    print("Generating 20 realistic smartphone-photographed benchmark images...")

    # =========================================================================
    # 조건 1. 단순 배경 + 정면 (5장)
    # 책상 위에서 손으로 위에서 내려다보며 찍은 탑뷰.
    # 미세한 손 회전각 (2~9도)과 사다리꼴 원근감, 자연스러운 그림자 반영.
    # =========================================================================
    for i in range(1, 6):
        desk = create_wood_desk_background(canvas_h, canvas_w, tone="warm")
        doc = create_realistic_document(title=f"STORE RECEIPT - #{i:03d}")

        # 회전 및 미세 원근 꼭짓점 계산
        rot = (i - 3) * 3.5  # -7도 ~ +7도 자연스러운 손 각도
        rad = np.radians(rot)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        base_corners = np.array([
            [230, 140],
            [670, 150],
            [630, 780],
            [190, 760],
        ], dtype=np.float32)

        # 회전 중심 기준 변환
        center = np.mean(base_corners, axis=0)
        rotated_corners = []
        for pt in base_corners:
            dx, dy = pt[0] - center[0], pt[1] - center[1]
            rx = center[0] + dx * cos_a - dy * sin_a + (i * 3)
            ry = center[1] + dx * sin_a + dy * cos_a + (i * 2)
            rotated_corners.append([rx, ry])
        rotated_corners = np.array(rotated_corners, dtype=np.float32)

        final_img = apply_perspective_with_shadow(desk, doc, rotated_corners, shadow_offset=(12, 18), shadow_strength=0.42)

        path = os.path.join(out_dir, f"cond1_simple_{i}.jpg")
        imwrite_unicode(path, final_img)
        print(f"[조건 1: 단순 정면] 생성 완료 -> {path}")

    # =========================================================================
    # 조건 2. 그림자 / 불균일 조명 있음 (5장)
    # 스탠드 조명 및 작업자의 손/휴대폰 그림자가 종이 위로 짙게 드리워진 환경.
    # =========================================================================
    for i in range(1, 6):
        desk = create_wood_desk_background(canvas_h, canvas_w, tone="dark")
        doc = create_realistic_document(title=f"SHADOWED REPORT #{i}")

        corners = np.array([
            [220 + i * 5, 130 + i * 4],
            [680 - i * 4, 160 + i * 6],
            [640, 790],
            [180 + i * 6, 750],
        ], dtype=np.float32)

        img_with_doc = apply_perspective_with_shadow(desk, doc, corners, shadow_offset=(15, 20), shadow_strength=0.5)

        # 대각선 방향 강한 스탠드 조명 및 스마트폰 손 그림자 마스크 생성
        light_gradient = np.ones((canvas_h, canvas_w), dtype=np.float32)
        for y in range(canvas_h):
            for x in range(canvas_w):
                # 대각선 조명 감쇠
                diag = (x * 0.7 + y * 0.9) / (canvas_w + canvas_h)
                light_gradient[y, x] = max(0.25, 1.0 - (diag * 0.18 * i))

        # 중앙 스마트폰 그림자 타원 추가
        shadow_ellipse = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        cv2.ellipse(shadow_ellipse, (380 + i * 20, 420 + i * 15), (180, 240), 25, 0, 360, 0.45, -1)
        shadow_ellipse = cv2.GaussianBlur(shadow_ellipse, (71, 71), 0)

        total_shadow = np.clip(light_gradient - shadow_ellipse, 0.22, 1.0)
        shadowed_result = (img_with_doc.astype(np.float32) * total_shadow[:, :, None]).astype(np.uint8)

        path = os.path.join(out_dir, f"cond2_shadow_{i}.jpg")
        imwrite_unicode(path, shadowed_result)
        print(f"[조건 2: 그림자/조명] 생성 완료 -> {path}")

    # =========================================================================
    # 조건 3. 45도 이상 기울어짐 (5장)
    # 비스듬히 책상에 앉아서 스마트폰을 사선(45~60도)으로 기울여 촬영한 심한 원근 왜곡.
    # =========================================================================
    for i in range(1, 6):
        desk = create_wood_desk_background(canvas_h, canvas_w, tone="light")
        doc = create_realistic_document(title=f"TILTED SCAN #{i:02d}")

        # 원근 단축이 극심한 사다리꼴 꼭짓점
        skew_top = 80 + i * 25
        skew_side = 40 + i * 15
        corners = np.array([
            [120 + skew_top, 100 + i * 10],       # 상단 원거리 (폭이 매우 좁음)
            [780 - skew_top, 130 + i * 15],
            [840 - skew_side, 820],               # 하단 근거리 (폭이 매우 넓음)
            [60 + skew_side, 780],
        ], dtype=np.float32)

        tilted_result = apply_perspective_with_shadow(desk, doc, corners, shadow_offset=(10, 25), shadow_strength=0.55)

        path = os.path.join(out_dir, f"cond3_tilted_{i}.jpg")
        imwrite_unicode(path, tilted_result)
        print(f"[조건 3: 45도 왜곡] 생성 완료 -> {path}")

    # =========================================================================
    # 조건 4. 복잡한 배경 및 방해 객체 (5장)
    # 책상 위에 줄무늬 패턴 매트, 다른 서류 봉투, 사각 노트, 펜이 함께 놓인 환경.
    # =========================================================================
    for i in range(1, 6):
        desk = create_wood_desk_background(canvas_h, canvas_w, tone="warm")

        # 1. 책상 위 격자/줄무늬 식탁보 매트
        mat = np.random.randint(60, 110, (canvas_h, canvas_w, 3), dtype=np.uint8)
        for ly in range(0, canvas_h, 35):
            cv2.line(desk, (0, ly), (canvas_w, ly), (45, 60, 80), 3)
        for lx in range(0, canvas_w, 35):
            cv2.line(desk, (lx, 0), (lx, canvas_h), (45, 60, 80), 2)

        # 2. 주변 방해 객체: 사각 메모 패드, 펜, 명함
        cv2.rectangle(desk, (40, 60), (320, 280), (70, 95, 140), -1)  # 파란색 노트
        cv2.rectangle(desk, (580, 520), (860, 820), (40, 80, 50), -1) # 초록색 다이어리
        cv2.line(desk, (100, 650), (280, 780), (20, 20, 20), 10)       # 볼펜

        doc = create_realistic_document(title=f"COMPLEX DESK #{i}")

        corners = np.array([
            [230 + (i * 8), 170 + (i * 6)],
            [660 - (i * 6), 190 + (i * 8)],
            [610, 750],
            [200 + (i * 4), 720],
        ], dtype=np.float32)

        complex_result = apply_perspective_with_shadow(desk, doc, corners, shadow_offset=(15, 20), shadow_strength=0.48)

        path = os.path.join(out_dir, f"cond4_complex_{i}.jpg")
        imwrite_unicode(path, complex_result)
        print(f"[조건 4: 복잡한 배경] 생성 완료 -> {path}")

    print("\nAll 20 realistic smartphone benchmark images successfully generated!")


if __name__ == "__main__":
    generate_benchmark_dataset()
