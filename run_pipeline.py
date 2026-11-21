"""Chạy toàn bộ quy trình bằng một lệnh: python run_pipeline.py --config configs/cape_town.yaml [--skip-llm]
--skip-llm: không gọi API, chỉ dùng kết quả LLM đã cache (không cần khoá API)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import baseline, download, evaluate, kpi, llm, qa, viz  # noqa: E402
from common import load_cfg  # noqa: E402

STEPS = [download, qa, llm, baseline, kpi, viz, evaluate]  # evaluate cuối: dừng để điền nhãn tay nếu chưa có


def main(args):
    cfg = load_cfg(args.config)
    extra = {llm: {"skip_llm": args.skip_llm}}
    for step in STEPS:
        print(f"\n=== {step.__name__} ===")
        if step.main(cfg, **extra.get(step, {})):
            sys.exit(f"Bước {step.__name__} chưa hoàn tất: chạy lại để tiếp tục.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="đường dẫn YAML, ví dụ configs/cape_town.yaml")
    ap.add_argument("--skip-llm", action="store_true")
    main(ap.parse_args())