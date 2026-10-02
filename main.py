"""DAY2 전체 분석 실행: python main.py (Python 3.11 권장)."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="ESS 배터리 DAY2: 학습 → 평가 → 그래프 → 오류 분석")
    parser.add_argument("--rebuild-features", action="store_true", help="원본 MAT에서 피처를 다시 추출")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw", help="MAT 파일 3개가 있는 폴더")
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    try:
        from src.features import build_features
        from src.train import run_training
        from src.plots import make_figures
        from src.diagnostics import diagnose
    except ModuleNotFoundError as exc:
        print(f"필요한 패키지가 없습니다: {exc.name}\n"
              "선택한 Python 환경에서 python -m pip install -r requirements.txt 를 실행하세요.",
              file=sys.stderr)
        return 1

    if args.rebuild_features:
        print("[1/4] 원본 MAT에서 피처 추출", flush=True)
        build_features(args.raw_dir.resolve(), ROOT / "data/processed")
    else:
        print("[1/4] 포함된 셀별 피처 사용", flush=True)
    feature_path = ROOT / "data/processed/features.csv"
    if not feature_path.is_file():
        parser.error("data/processed/features.csv가 없습니다. README.md의 환경 설정을 확인하세요.")

    print("[2/4] Batch 1 모델 선택 및 Batch 2·3 평가", flush=True)
    summary, _, _ = run_training(feature_path, ROOT / "results")
    print("[3/4] 결과 그래프 생성", flush=True)
    make_figures(ROOT)
    print("[4/4] 입력 분포 변화와 기준 모델 진단", flush=True)
    diagnose(ROOT)
    print(f"\n완료: {summary['model']} | Batch 2 MAPE {summary['test_mape']:.2f}%")
    print(f"결과 폴더: {ROOT / 'results'}")
    print("노트북 출력 갱신은 노트북에서 Run All을 실행하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
