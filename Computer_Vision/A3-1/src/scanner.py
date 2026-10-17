"""Document Scanner Engine using OpenCV.

A3-1: 휴대폰으로 찍은 종이를 똑바르게 펴서 스캔하게 만들기
- Canny Edge Detection
- Contour Extraction & 4-Corner Approximation (RDP Algorithm)
- 4-Corner Points Ordering (Sum/Difference technique)
- Homography Perspective Transform (Warp Perspective)
- Adaptive Thresholding for B&W scanned output
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """4개의 꼭짓점 좌표를 [좌상(TL), 우상(TR), 우하(BR), 좌하(BL)] 순서로 정렬.

    Args:
        pts: (4, 2) 크기의 꼭짓점 좌표 배열

    Returns:
        정렬된 (4, 2) 크기의 float32 좌표 배열
    """
    rect = np.zeros((4, 2), dtype="float32")

    # x + y 합 계산: 좌상단(TL)은 합이 최소, 우하단(BR)은 합이 최대
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # y - x 차 계산 (또는 diff): 우상단(TR)은 diff가 최소, 좌하단(BL)은 diff가 최대
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """정렬된 4개 꼭짓점을 기준으로 이미지를 반듯한 직사각형으로 투시 변환(Warp Perspective).

    Args:
        image: 원본 이미지 (RGB 또는 Grayscale)
        pts: 4개의 꼭짓점 좌표 (4, 2)

    Returns:
        보정된 직사각형 이미지
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 1. 새 이미지의 너비(Width) 계산 (상단 너비와 하단 너비 중 최댓값)
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # 2. 새 이미지의 높이(Height) 계산 (우측 높이와 좌측 높이 중 최댓값)
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # 비정상적인 작은 크기 방어
    max_width = max(max_width, 100)
    max_height = max(max_height, 100)

    # 3. 투시 변환 대상 좌표(dst) 설정
    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    # 4. 투시 변환 행렬 M 계산 및 워핑 수행
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))

    return warped


class DocumentScanner:
    """문서 검출, 윤곽선 추출, 원근 투영 변환 및 적응형 이진화 파이프라인 클래스."""

    def __init__(
        self,
        blur_ksize: int = 5,
        canny_low: int = 75,
        canny_high: int = 200,
        approx_eps_ratio: float = 0.02,
        adaptive_block_size: int = 21,
        adaptive_c: int = 10,
    ):
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.approx_eps_ratio = approx_eps_ratio
        self.adaptive_block_size = (
            adaptive_block_size
            if adaptive_block_size % 2 == 1
            else adaptive_block_size + 1
        )
        self.adaptive_c = adaptive_c

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """이미지를 Grayscale 변환 후 가우시안 블러 적용."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        ksize = (
            self.blur_ksize
            if self.blur_ksize % 2 == 1
            else self.blur_ksize + 1
        )
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        return gray, blurred

    def detect_edges(self, blurred: np.ndarray) -> np.ndarray:
        """Canny 에지 검출기를 적용하여 경계선 맵 생성."""
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        # 끊어진 에지를 살짝 메우기 위한 모폴로지 닫기(선택)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return edges

    def find_document_contour(
        self, edges: np.ndarray, min_area_ratio: float = 0.05
    ) -> Optional[np.ndarray]:
        """에지 맵에서 면적이 가장 크고 꼭짓점이 4개인 문서 사각형 윤곽선 탐색.

        Args:
            edges: Canny 이진 에지 이미지
            min_area_ratio: 전체 이미지 면적 대비 최소 문서 면적 비율

        Returns:
            검출된 4개의 꼭짓점 좌표 (4, 2) 또는 검출 실패 시 None
        """
        contours, _ = cv2.findContours(
            edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # 면적 내림차순 정렬
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        img_area = edges.shape[0] * edges.shape[1]

        for c in contours:
            area = cv2.contourArea(c)
            if area < (img_area * min_area_ratio):
                break

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, self.approx_eps_ratio * peri, True)

            # 꼭짓점이 정확히 4개인 경우 문서로 판단
            if len(approx) == 4:
                return approx.reshape(4, 2)

        return None

    def postprocess_scan(self, warped_image: np.ndarray) -> np.ndarray:
        """원근 보정된 이미지에서 그림자를 지우고 글씨를 선명하게 하는 적응형 이진화 적용."""
        if len(warped_image.shape) == 3:
            gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = warped_image.copy()

        block_size = (
            self.adaptive_block_size
            if self.adaptive_block_size % 2 == 1
            else self.adaptive_block_size + 1
        )
        block_size = max(block_size, 3)

        scanned = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self.adaptive_c,
        )
        return scanned

    def scan(
        self,
        image: np.ndarray,
        custom_pts: Optional[np.ndarray] = None,
    ) -> Dict[str, Optional[np.ndarray]]:
        """전체 파이프라인을 실행하고 각 단계별 시각화 이미지를 반환."""
        orig = image.copy()
        h, w = orig.shape[:2]

        # 1. 전처리 (그레이스케일 + 블러)
        gray, blurred = self.preprocess(orig)

        # 2. 에지 검출
        edges = self.detect_edges(blurred)

        # 3. 문서 윤곽선 탐지
        doc_pts = (
            custom_pts
            if custom_pts is not None
            else self.find_document_contour(edges)
        )

        contour_drawn = orig.copy()
        warped = None
        scanned = None
        success = False

        if doc_pts is not None:
            success = True
            # 윤곽선 그리기
            cv2.drawContours(
                contour_drawn, [doc_pts.astype(int)], -1, (0, 255, 0), 3
            )
            for pt in doc_pts:
                cv2.circle(contour_drawn, tuple(pt.astype(int)), 7, (0, 0, 255), -1)

            # 4. 원근 투영 변환
            warped = four_point_transform(orig, doc_pts)

            # 5. 적응형 이진화 스캔본 생성
            scanned = self.postprocess_scan(warped)
        else:
            # 문서 윤곽선을 못 찾은 경우 전체 이미지를 대상으로 스캔 처리
            warped = orig.copy()
            scanned = self.postprocess_scan(warped)

        return {
            "success": success,
            "original": orig,
            "gray": gray,
            "blurred": blurred,
            "edges": edges,
            "contour_drawn": contour_drawn,
            "doc_pts": doc_pts,
            "warped": warped,
            "scanned": scanned,
        }
