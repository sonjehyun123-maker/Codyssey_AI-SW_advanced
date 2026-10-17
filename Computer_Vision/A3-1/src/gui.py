"""Interactive Document Scanner GUI with OpenCV Trackbars.

A3-1 실시간 트랙바 조절 및 스캔 결과 시각화 도구.
- 트랙바를 통한 실시간 파라미터 조절 (블러, Canny Low/High, 적응형 이진화 BlockSize, C)
- 4분할 화면 시각화: [원본/윤곽선 | Canny 에지] / [원근 보정본 | 최종 스캔본]
- 키보드 단축키 지원 ('s': 저장, 'r': 리셋, 'q': 종료)
- 웹캠 모드 지원 (--webcam)
"""

import argparse
import os
import sys
from typing import Optional
import cv2
import numpy as np

# 같은 디렉터리 내 scanner 모듈 임포트
try:
    from .scanner import DocumentScanner, order_points, four_point_transform
except ImportError:
    from scanner import DocumentScanner, order_points, four_point_transform


class InteractiveScannerGUI:
    """OpenCV 윈도우 및 트랙바를 이용한 실시간 인터랙티브 문서 스캐너."""

    WINDOW_NAME = "Interactive Document Scanner (A3-1)"

    def __init__(self, image: np.ndarray, output_dir: str = "output"):
        self.orig_image = image
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 화면 크기에 맞게 이미지 다운스케일링 (화면 밖으로 나가지 않도록 최대 높이 800 기준)
        self.display_scale = 1.0
        h, w = self.orig_image.shape[:2]
        if h > 800 or w > 1200:
            scale_h = 800.0 / h
            scale_w = 1200.0 / w
            self.display_scale = min(scale_h, scale_w)

        self.proc_image = cv2.resize(
            self.orig_image,
            (int(w * self.display_scale), int(h * self.display_scale)),
            interpolation=cv2.INTER_AREA,
        )

        # 기본 파라미터 설정
        self.blur_ksize = 5
        self.canny_low = 50
        self.canny_high = 150
        self.adaptive_block = 21
        self.adaptive_c = 10

        self.last_results = None

    def _on_trackbar_change(self, _=None):
        """트랙바 값이 변경될 때마다 화면 갱신."""
        self._update_display()

    def _setup_trackbars(self):
        """OpenCV 트랙바 생성."""
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        cv2.createTrackbar(
            "Blur Kernel (Odd)", self.WINDOW_NAME, (self.blur_ksize - 1) // 2, 10, self._on_trackbar_change
        )
        cv2.createTrackbar(
            "Canny Low", self.WINDOW_NAME, self.canny_low, 255, self._on_trackbar_change
        )
        cv2.createTrackbar(
            "Canny High", self.WINDOW_NAME, self.canny_high, 255, self._on_trackbar_change
        )
        cv2.createTrackbar(
            "Adaptive Block (Odd)",
            self.WINDOW_NAME,
            (self.adaptive_block - 3) // 2,
            25,
            self._on_trackbar_change,
        )
        cv2.createTrackbar(
            "Adaptive C", self.WINDOW_NAME, self.adaptive_c, 30, self._on_trackbar_change
        )

    def _read_trackbar_values(self):
        """현재 트랙바에서 파라미터 읽기."""
        blur_idx = cv2.getTrackbarPos("Blur Kernel (Odd)", self.WINDOW_NAME)
        self.blur_ksize = blur_idx * 2 + 1

        self.canny_low = cv2.getTrackbarPos("Canny Low", self.WINDOW_NAME)
        self.canny_high = cv2.getTrackbarPos("Canny High", self.WINDOW_NAME)

        adapt_block_idx = cv2.getTrackbarPos("Adaptive Block (Odd)", self.WINDOW_NAME)
        self.adaptive_block = adapt_block_idx * 2 + 3

        self.adaptive_c = cv2.getTrackbarPos("Adaptive C", self.WINDOW_NAME)

    def _update_display(self):
        """현재 파라미터로 스캔 알고리즘을 수행하고 4분할 격자 윈도우 표시."""
        self._read_trackbar_values()

        scanner = DocumentScanner(
            blur_ksize=self.blur_ksize,
            canny_low=self.canny_low,
            canny_high=self.canny_high,
            adaptive_block_size=self.adaptive_block,
            adaptive_c=self.adaptive_c,
        )

        results = scanner.scan(self.proc_image)
        self.last_results = results

        # 4개 뷰 준비
        # 1. 윤곽선 감지 뷰
        contour_view = results["contour_drawn"].copy()
        status_text = "Status: Document Found" if results["success"] else "Status: Looking for Document..."
        status_color = (0, 255, 0) if results["success"] else (0, 0, 255)
        cv2.putText(contour_view, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(
            contour_view,
            f"Canny: [{self.canny_low}, {self.canny_high}] | Blur: {self.blur_ksize}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # 2. Canny 에지 뷰
        edge_bgr = cv2.cvtColor(results["edges"], cv2.COLOR_GRAY2BGR)
        cv2.putText(edge_bgr, "Canny Edges", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 3. 원근 보정본 (Warped)
        h, w = contour_view.shape[:2]
        if results["warped"] is not None:
            warped_view = cv2.resize(results["warped"], (w, h))
        else:
            warped_view = np.zeros_like(contour_view)
        cv2.putText(warped_view, "Warped Perspective", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        # 4. 적응형 이진화 최종본 (Scanned)
        if results["scanned"] is not None:
            scanned_bgr = cv2.cvtColor(results["scanned"], cv2.COLOR_GRAY2BGR)
            scanned_view = cv2.resize(scanned_bgr, (w, h))
        else:
            scanned_view = np.zeros_like(contour_view)
        cv2.putText(scanned_view, "Final Scanned Output (Adaptive B&W)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(
            scanned_view,
            f"BlockSize: {self.adaptive_block} | C: {self.adaptive_c}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

        # 상하좌우 격자 합성 (2x2 Grid)
        top_row = np.hstack([contour_view, edge_bgr])
        bottom_row = np.hstack([warped_view, scanned_view])
        grid = np.vstack([top_row, bottom_row])

        # 하단 단축키 가이드 바
        guide_bar = np.zeros((40, grid.shape[1], 3), dtype=np.uint8)
        guide_text = "Hotkeys: [S] Save Scan Result | [R] Reset Trackbars | [Q / ESC] Quit"
        cv2.putText(guide_bar, guide_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        grid_with_guide = np.vstack([grid, guide_bar])

        cv2.imshow(self.WINDOW_NAME, grid_with_guide)

    def run(self):
        """GUI 메인 이벤트 루프."""
        self._setup_trackbars()
        self._update_display()

        save_count = 0
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in [ord("q"), 27]:  # 'q' or ESC
                break
            elif key == ord("s"):  # 's' to save
                if self.last_results and self.last_results["scanned"] is not None:
                    # 원본 스케일 기준으로 고해상도 재스캔
                    full_scanner = DocumentScanner(
                        blur_ksize=self.blur_ksize,
                        canny_low=self.canny_low,
                        canny_high=self.canny_high,
                        adaptive_block_size=self.adaptive_block,
                        adaptive_c=self.adaptive_c,
                    )
                    # 스케일 보정된 doc_pts 계산
                    if self.last_results["doc_pts"] is not None:
                        orig_pts = self.last_results["doc_pts"] / self.display_scale
                        full_res = full_scanner.scan(self.orig_image, custom_pts=orig_pts)
                    else:
                        full_res = full_scanner.scan(self.orig_image)

                    save_count += 1
                    save_path = os.path.join(self.output_dir, f"scan_result_{save_count}.png")
                    cv2.imwrite(save_path, full_res["scanned"])
                    print(f"[저장 완료] 스캔 결과가 저장되었습니다: {save_path}")
            elif key == ord("r"):  # 'r' to reset
                cv2.setTrackbarPos("Blur Kernel (Odd)", self.WINDOW_NAME, 2)
                cv2.setTrackbarPos("Canny Low", self.WINDOW_NAME, 50)
                cv2.setTrackbarPos("Canny High", self.WINDOW_NAME, 150)
                cv2.setTrackbarPos("Adaptive Block (Odd)", self.WINDOW_NAME, 9)
                cv2.setTrackbarPos("Adaptive C", self.WINDOW_NAME, 10)

        cv2.destroyAllWindows()


def run_webcam_mode(camera_id: int = 0, output_dir: str = "output"):
    """웹캠 실시간 프레임 스캐너 모드."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[오류] 웹캠(ID {camera_id})을 열 수 없습니다.")
        return

    scanner = DocumentScanner(blur_ksize=5, canny_low=60, canny_high=160, adaptive_block_size=21, adaptive_c=10)
    print("=== 웹캠 문서 스캐너 실행 중 ===")
    print("- 's' 키: 현재 프레임 문서 스캔 및 저장")
    print("- 'q' 키: 종료")

    save_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = scanner.scan(frame)
        display_frame = results["contour_drawn"]

        status = "Document Detected [Press 'S' to scan]" if results["success"] else "Searching for document..."
        color = (0, 255, 0) if results["success"] else (0, 0, 255)
        cv2.putText(display_frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Webcam Document Scanner (Press 's' to save, 'q' to quit)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), 27]:
            break
        elif key == ord("s"):
            save_idx += 1
            path_warp = os.path.join(output_dir, f"webcam_scanned_{save_idx}.png")
            cv2.imwrite(path_warp, results["scanned"])
            print(f"[저장 완료] 웹캠 스캔본 저장: {path_warp}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="A3-1 대화형 OpenCV 문서 스캐너 GUI")
    parser.add_argument("--image", type=str, default=None, help="스캔할 이미지 파일 경로")
    parser.add_argument("--webcam", action="store_true", help="웹캠 실시간 스트림 모드 실행")
    parser.add_argument("--camera-id", type=int, default=0, help="웹캠 디바이스 ID (기본값: 0)")
    parser.add_argument("--output-dir", type=str, default="output", help="스캔본 저장 디렉터리")
    args = parser.parse_args()

    if args.webcam:
        run_webcam_mode(camera_id=args.camera_id, output_dir=args.output_dir)
    elif args.image:
        if not os.path.exists(args.image):
            print(f"[오류] 이미지 파일을 찾을 수 없습니다: {args.image}")
            sys.exit(1)
        image = cv2.imread(args.image)
        if image is None:
            print(f"[오류] 유효한 이미지 형식이 아닙니다: {args.image}")
            sys.exit(1)
        gui = InteractiveScannerGUI(image, output_dir=args.output_dir)
        gui.run()
    else:
        print("사용법 안내:")
        print("1) 이미지 파일 스캔 GUI 실행: python src/gui.py --image test_images/sample.jpg")
        print("2) 실시간 웹캠 스캔 실행:       python src/gui.py --webcam")


if __name__ == "__main__":
    main()
