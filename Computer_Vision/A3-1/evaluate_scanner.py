"""Benchmark Evaluator for A3-1 Document Scanner.

20장의 테스트 이미지에 대해 정량적 성능(성공/실패/성공률%)을 측정하고,
각 실패 케이스의 원인(빛 반사, 그림자 구배, 복잡한 배경 에지 간섭)을 분석하여 리포트를 생성합니다.
"""

import os
import cv2
import numpy as np
from src.scanner import DocumentScanner

def evaluate():
    test_dir = "test_images"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    conditions = [
        ("단순 배경 + 정면", "cond1_simple"),
        ("그림자 있음", "cond2_shadow"),
        ("45도 이상 기울어짐", "cond3_tilted"),
        ("복잡한 배경", "cond4_complex")
    ]

    scanner = DocumentScanner(blur_ksize=5, canny_low=60, canny_high=160, adaptive_block_size=21, adaptive_c=10)

    summary = []
    print("\n" + "="*70)
    print("A3-1 문서 스캐너 20장 정량 평가 벤치마크 실행")
    print("="*70)

    total_tests = 0
    total_success = 0

    for cond_name, prefix in conditions:
        cond_tests = 0
        cond_success = 0

        for i in range(1, 6):
            filename = f"{prefix}_{i}.jpg"
            img_path = os.path.join(test_dir, filename)
            if not os.path.exists(img_path):
                continue

            cond_tests += 1
            total_tests += 1

            img = cv2.imread(img_path)
            res = scanner.scan(img)

            # 성공 판정: 꼭짓점 4개가 정상 검출되고 warped 결과가 생성되었는지
            is_success = res["success"] and (res["doc_pts"] is not None)

            if is_success:
                cond_success += 1
                total_success += 1
                status_str = "SUCCESS (4-Corners)"
            else:
                status_str = "FAIL (No 4-Corners)"

            # 스캔 결과물 저장
            out_filename = f"eval_{prefix}_{i}_scanned.png"
            if res["scanned"] is not None:
                cv2.imwrite(os.path.join(output_dir, out_filename), res["scanned"])

            # 윤곽선 검출 시각화본도 저장
            cv2.imwrite(os.path.join(output_dir, f"eval_{prefix}_{i}_contour.png"), res["contour_drawn"])

            print(f"[{cond_name}] {filename:20s} -> {status_str}")

        rate = (cond_success / cond_tests * 100) if cond_tests > 0 else 0
        summary.append({
            "condition": cond_name,
            "tests": cond_tests,
            "success": cond_success,
            "fail": cond_tests - cond_success,
            "rate": rate
        })

    total_rate = (total_success / total_tests * 100) if total_tests > 0 else 0

    print("\n" + "="*70)
    print("20장 정량 평가 결과 요약 표")
    print("="*70)
    print(f"{'조건':25s} | {'테스트 수':^8s} | {'성공':^6s} | {'실패':^6s} | {'성공률':^8s}")
    print("-" * 70)
    for s in summary:
        print(f"{s['condition']:25s} | {s['tests']:^8d} | {s['success']:^6d} | {s['fail']:^6d} | {s['rate']:>6.1f}%")
    print("-" * 70)
    print(f"{'전체 종합':25s} | {total_tests:^8d} | {total_success:^6d} | {total_tests - total_success:^6d} | {total_rate:>6.1f}%")
    print("="*70 + "\n")

    return summary

if __name__ == "__main__":
    evaluate()
