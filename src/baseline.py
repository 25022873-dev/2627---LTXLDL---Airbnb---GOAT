"""Phương pháp đối chứng không LLM (VADER): python src/baseline.py configs/cape_town.yaml
Ra: data/processed/baseline_results.csv (review_id, sentiment, status). Ngưỡng đọc từ mục `baseline:` của YAML, cố định trước, không chỉnh trên nhãn tay."""
import re
import sys

import pandas as pd
import yaml
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from download import ROOT
from llm import clean
from qa import load_refs

VADER = SentimentIntensityAnalyzer()  # từ điển có sẵn: không huấn luyện, không cần mạng
CLAUSE = re.compile(r"(?<=[.!?])\s+|\b(?:but|however|although|though|except)\b", re.I)  # tách câu (giữ dấu câu) và mệnh đề đối lập


def score(text):
    """Điểm tổng hợp (compound) của VADER, từ -1 đến +1."""
    return VADER.polarity_scores(text)["compound"]


def label(text, B):
    """mixed nếu có mệnh đề khen rõ (>= mixed_min) và mệnh đề chê rõ (<= -mixed_min); không thì so điểm cả văn bản với thr."""
    s = [score(c) for c in CLAUSE.split(text) if c.strip()]  # không có mệnh đề nào chấm được: s rỗng, không lỗi
    if max(s, default=0) >= B["mixed_min"] and min(s, default=0) <= -B["mixed_min"]:
        return "mixed"
    t = score(text)
    return "positive" if t >= B["thr"] else "negative" if t <= -B["thr"] else B["default"]  # không có tín hiệu: nhãn mặc định


def main(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ref = load_refs(cfg)["review"]  # cùng mẫu «Mẫu LLM» và cùng clean() như llm.py
    sample = [int(x) for x in ref.loc[ref["Mẫu LLM"].str.lower().eq("x"), "review_id"]]
    proc = ROOT / "data" / "processed"
    rv = pd.read_csv(proc / "reviews_clean.csv", usecols=["id", "comments"]).set_index("id")["comments"]
    texts = {i: clean(rv[i], cfg["llm"]["max_chars"]) for i in sample if i in rv.index}
    rows = [{"review_id": i, "sentiment": label(t, cfg["baseline"]) if t else None, "status": "ok" if t else "empty"}
            for i, t in texts.items()]
    out = pd.DataFrame(rows)
    out.to_csv(proc / "baseline_results.csv", index=False)
    print(f"{len(out)} review -> baseline_results.csv; {out['sentiment'].value_counts().to_dict()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/cape_town.yaml")