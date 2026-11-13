"""Phương pháp đối chứng không LLM: VADER + luật ngưỡng, chỉ gán nhãn cảm xúc positive / mixed / negative.

Chạy: python src/baseline.py configs/cape_town.yaml
Ra:   data/processed/baseline_results.csv (review_id, sentiment, status)
Cùng mẫu «Mẫu LLM» và cùng clean() như llm.py. Không cần khoá API hay mạng.
Tham số (ngưỡng) đọc từ mục `baseline:` của YAML, cố định trước, không chỉnh trên nhãn tay."""
import re
import sys

import pandas as pd
import yaml
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from download import ROOT
from llm import clean  # dùng chung bước làm sạch -> hai bên chấm trên cùng văn bản
from qa import load_refs

VADER = SentimentIntensityAnalyzer()
CLAUSE = re.compile(r"(?<=[.!?])\s+|\b(?:but|however|although|though)\b", re.I)  # tách câu và mệnh đề đối lập


def score(s):
    return VADER.polarity_scores(s)["compound"]  # [-1, 1]; VADER đã xử lý phủ định và «but»


def label(text, B):
    parts = [p.strip() for p in CLAUSE.split(text) if p and p.strip()]
    sc = [score(p) for p in parts] or [0]
    total, thr = score(text), B["thr"]
    if max(sc) >= B["mixed_min"] and min(sc) <= -B["mixed_min"]:  # cả khen lẫn chê rõ rệt (không để triệt tiêu khi cộng)
        return "mixed"
    if total >= thr:
        return "positive"
    if total <= -thr:
        return "negative"
    return B["default"]  # không có tín hiệu: đoán nhãn phổ biến nhất


def main(config_path):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    root, snap0, out = cfg["data_root"].strip("/ "), str(cfg["snapshots"][0]), ROOT / "data/processed"
    ref = load_refs(cfg)["review"]
    ids = [int(x) for x in ref.loc[ref["Mẫu LLM"].str.lower() == "x", "review_id"]]
    comments = pd.read_parquet(out / root / snap0 / "reviews.parquet").set_index("id")["comments"]

    rows = []
    for rid in ids:
        text = clean(comments.get(rid), cfg["llm"]["max_chars"])
        rows.append({"review_id": rid, "sentiment": label(text, cfg["baseline"]) if text else None,
                     "status": "ok" if text else "empty"})
    pd.DataFrame(rows).to_csv(out / "baseline_results.csv", index=False)
    print(f"Done: {len(rows)} reviews, {sum(r['status'] == 'ok' for r in rows)} ok -> baseline_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/cape_town.yaml"))