"""Phương pháp đối chứng không LLM (phương án B): VADER + luật ngưỡng + từ khoá khía cạnh + langdetect.

Chạy: python src/baseline.py configs/cape_town.yaml [--limit N]
Ra:   data/processed/baseline_results.csv (cùng cột với llm_results.csv để evaluate.py chấm chung)
Cùng mẫu «Mẫu LLM», cùng clean() và cờ review tự động như llm.py. Không cần khoá API hay mạng.
Tham số (ngưỡng, từ khoá) đọc từ mục `baseline:` của YAML, cố định trước, không chỉnh trên nhãn tay."""
import argparse
import re
import sys

import pandas as pd
import yaml
from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from download import ROOT
from llm import AUTO, clean  # dùng chung bước làm sạch -> hai bên chấm trên cùng văn bản
from qa import load_refs

VADER = SentimentIntensityAnalyzer()
CLAUSE = re.compile(r"(?<=[.!?])\s+|\b(?:but|however|although|though)\b", re.I)  # tách câu và mệnh đề đối lập


def score(s):
    return VADER.polarity_scores(s)["compound"]  # [-1, 1]; VADER đã xử lý phủ định và «but»


def language(text, B):
    """ISO 639-1; «unknown» nếu quá ngắn hoặc độ tin cậy thấp."""
    if len(text) < B["lang_min_chars"]:
        return "unknown"
    try:
        top = detect_langs(text)[0]
    except LangDetectException:
        return "unknown"
    return top.lang.split("-")[0] if top.prob >= B["lang_min_prob"] else "unknown"


def label(text, B, kw):
    """-> (sentiment, {aspect: polarity | None})."""
    parts = [p.strip() for p in CLAUSE.split(text) if p and p.strip()]
    sc = [score(p) for p in parts]
    total, thr = score(text), B["thr"]
    if max(sc) >= B["mixed_min"] and min(sc) <= -B["mixed_min"]:  # cả khen lẫn chê rõ rệt (không để triệt tiêu khi cộng)
        sent = "mixed"
    elif total >= thr:
        sent = "positive"
    elif total <= -thr:
        sent = "negative"
    else:
        sent = B["default"]  # không có tín hiệu: đoán nhãn phổ biến nhất
    asp = {}
    for a, pat in kw.items():  # phân cực khía cạnh = điểm trung bình các câu chứa từ khoá
        hits = [s for p, s in zip(parts, sc) if pat.search(p)]
        m = sum(hits) / len(hits) if hits else 0
        asp[a] = "positive" if m >= thr else "negative" if m <= -thr else None
    return sent, asp


def main(config_path, limit):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    B, L, root, snap0 = cfg["baseline"], cfg["llm"], cfg["data_root"].strip("/ "), str(cfg["snapshots"][0])
    DetectorFactory.seed = cfg["seed"]
    out = ROOT / "data/processed"
    kw = {a: re.compile(r"\b(?:%s)" % "|".join(map(re.escape, B["keywords"][a])), re.I) for a in L["aspects"]}
    ref = load_refs(cfg)["review"]
    ids = [int(x) for x in ref.loc[ref["Mẫu LLM"].str.lower() == "x", "review_id"]][:limit]
    comments = pd.read_parquet(out / root / snap0 / "reviews.parquet").set_index("id")["comments"]

    rows = []
    for rid in ids:
        text, _ = clean(comments.get(rid), L["max_chars"])
        row = {"review_id": rid, "sentiment": None, "language": None, "status": "empty",
               "auto_generated": bool(AUTO.match(text)), **{f"aspect_{a}": None for a in L["aspects"]}}
        if text:
            sent, asp = label(text, B, kw)
            row.update(sentiment=sent, language=language(text, B), status="ok",
                       **{f"aspect_{a}": p for a, p in asp.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "baseline_results.csv", index=False)
    print(f"Done: {len(rows)} reviews, {sum(r['status'] == 'ok' for r in rows)} ok -> baseline_results.csv")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/cape_town.yaml")
    ap.add_argument("--limit", type=int, help="chỉ xử lý N review đầu (thử nhanh)")
    a = ap.parse_args()
    sys.exit(main(a.config, a.limit))
