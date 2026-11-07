"""Baseline không dùng LLM cho «Mẫu LLM»: VADER (cảm xúc) + từ khoá tự lập (khía cạnh) + langdetect (ngôn ngữ).

Chạy:  python src/baseline.py configs/cape_town.yaml [--limit N] [--input data/samples/x.csv] [--out FILE]
Ra:    data/processed/baseline_results.csv, cùng cột với llm_results.csv
       (review_id, sentiment, language, status, auto_generated, aspect_<tên>) + cột sentiment_floor (mức A).
Không cần khoá API, không cần mạng.

Ngưỡng và từ khoá nằm ở khối `baseline:` trong YAML. Ngưỡng được CỐ ĐỊNH TỪ TRƯỚC, không chỉnh trên 100 nhãn tay.
Nếu muốn chỉnh, dùng tập phát triển riêng (không giao với 100 nhãn tay):
  python src/baseline.py configs/cape_town.yaml --sample-dev 50   # sinh tệp để tự gán nhãn
  python src/baseline.py configs/cape_town.yaml --dev data/samples/dev_labeled.csv [--tune]
"""
import argparse
import itertools
import re
import sys

import pandas as pd
import yaml

from download import ROOT
from qa import load_refs

# --- Hai đoạn dưới đây PHẢI giống llm.py để hai bên chấm trên cùng tập và cùng văn bản đã làm sạch. ---
AUTO = re.compile(r"^(the host canceled this reservation|the reservation was canceled|this is an automated posting)", re.I)
LABELS = ["positive", "mixed", "negative"]


def clean(text, max_chars):
    """Bỏ thẻ HTML, gọn khoảng trắng, cắt độ dài -> (văn bản, có bị cắt?). Bản sao của llm.clean."""
    t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text if isinstance(text, str) else "")).strip()
    return t[:max_chars], len(t) > max_chars
# ------------------------------------------------------------------------------------------------------


def load_tools(seed):
    """Nạp VADER và langdetect; báo lỗi rõ nếu chưa cài."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        from langdetect import DetectorFactory, detect_langs
        from langdetect.lang_detect_exception import LangDetectException
    except ImportError as e:
        sys.exit(f"Thiếu thư viện ({e.name}). Chạy: pip install -r requirements.txt")
    DetectorFactory.seed = seed  # langdetect có yếu tố ngẫu nhiên: đặt seed để lần chạy nào cũng ra cùng kết quả
    return SentimentIntensityAnalyzer(), detect_langs, LangDetectException


def aspect_regexes(keywords):
    """{khía cạnh: regex}. 'clean*' khớp clean/cleaned/cleanliness; từ không có '*' khớp nguyên từ."""
    out = {}
    for name, words in keywords.items():
        parts = [r"\b" + re.escape(w.lower()).replace(r"\*", r"\w*") + r"\b" for w in words]
        out[name] = re.compile("|".join(parts), re.I)
    return out


def split_clauses(text, contrast_words):
    """Tách thành câu (dấu . ! ? ;), rồi tách tiếp tại từ tương phản (but, however...)."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?;…])\s+", text) if p.strip()]
    if contrast_words:
        rx = re.compile(r"\b(?:" + "|".join(map(re.escape, contrast_words)) + r")\b", re.I)
        parts = [c.strip(" ,-") for p in parts for c in rx.split(p)]
    return [p for p in parts if p]


class Baseline:
    def __init__(self, cfg):
        self.B = B = cfg["baseline"]
        self.vader, self._detect, self._LangErr = load_tools(cfg["seed"])
        self.rx = aspect_regexes(B["aspect_keywords"])
        self.aspects = cfg["llm"]["aspects"]  # cùng danh sách khía cạnh với LLM
        missing = [a for a in self.aspects if a not in self.rx]
        if missing:
            sys.exit(f"baseline.aspect_keywords thiếu khía cạnh: {missing}")

    def analyze(self, text):
        """Phần tốn thời gian (chạy VADER): điểm cả review và từng mệnh đề. Tách khỏi luật để chỉnh ngưỡng nhanh."""
        clauses = [(c, self.vader.polarity_scores(c)["compound"])
                   for c in split_clauses(text, self.B["contrast_words"] if self.B["split_on_contrast"] else [])]
        return {"compound": self.vader.polarity_scores(text)["compound"], "clauses": clauses}

    def sentiment(self, an, P=None):
        """Luật nhãn. Tách tổng dương / tổng âm riêng để định nghĩa mixed (tổng gộp sẽ triệt tiêu nhau)."""
        P = P or self.B
        sc = [s for _, s in an["clauses"]]
        pos = sum(s for s in sc if s >= P["pos_threshold"])
        neg = -sum(s for s in sc if s <= P["neg_threshold"])
        if pos >= P["mixed_min_pos"] and neg >= P["mixed_min_neg"]:
            return "mixed"
        if an["compound"] >= P["pos_threshold"]:
            return "positive"
        if an["compound"] <= P["neg_threshold"]:
            return "negative"
        return P["majority_label"]  # điểm quanh 0 và không đủ hai phía: nghiêng về nhãn phổ biến nhất

    def aspect_polarity(self, an):
        """Với mỗi khía cạnh: tổng điểm các mệnh đề chứa từ khoá. None = không nhắc tới HOẶC không rõ cực."""
        t, out = self.B["aspect_threshold"], {}
        for a in self.aspects:
            hits = [s for c, s in an["clauses"] if self.rx[a].search(c)]
            tot = sum(hits)
            out[a] = "positive" if hits and tot >= t else "negative" if hits and tot <= -t else None
        return out

    def language(self, text):
        """Mã ISO 639-1; 'unknown' khi quá ngắn hoặc độ tin cậy thấp (không đoán bừa)."""
        if len(text) < self.B["lang_min_chars"]:
            return "unknown"
        try:
            top = self._detect(text)[0]
        except self._LangErr:
            return "unknown"
        return top.lang.split("-")[0] if top.prob >= self.B["lang_min_prob"] else "unknown"


def f1_report(y, p):
    """accuracy, macro-F1 (trên các nhãn xuất hiện ở y hoặc p), F1 từng lớp, ma trận nhầm lẫn (hàng = nhãn tay)."""
    labels = [l for l in LABELS if l in set(y) | set(p)]
    f1 = {}
    for l in labels:
        tp = sum(a == l and b == l for a, b in zip(y, p))
        fp = sum(a != l and b == l for a, b in zip(y, p))
        fn = sum(a == l and b != l for a, b in zip(y, p))
        f1[l] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    cm = pd.DataFrame([[sum(a == r and b == c for a, b in zip(y, p)) for c in labels] for r in labels],
                      index=labels, columns=labels)
    return {"n": len(y), "accuracy": sum(a == b for a, b in zip(y, p)) / len(y),
            "macro_f1": sum(f1.values()) / len(f1), "f1": f1, "cm": cm}


def load_sample(cfg, args):
    """-> (ids, {id: văn bản đã làm sạch}, ref). Mặc định: 1.000 review «Mẫu LLM» đọc từ reviews.parquet của mốc đầu."""
    L, ref = cfg["llm"], load_refs(cfg)["review"]
    if args.input:
        df = pd.read_csv(ROOT / args.input if not str(args.input).startswith("/") else args.input,
                         dtype={"review_id": "int64"})
        ids = [int(i) for i in df["review_id"]][:args.limit]
        comments = df.set_index("review_id")["comments"]
    else:
        root, snap0 = cfg["data_root"].strip("/ "), str(cfg["snapshots"][0])
        ids = [int(x) for x in ref.loc[ref["Mẫu LLM"].str.lower() == "x", "review_id"]][:args.limit]
        comments = pd.read_parquet(ROOT / "data/processed" / root / snap0 / "reviews.parquet").set_index("id")["comments"]
    cut = 0
    texts = {}
    for rid in ids:
        texts[rid], c = clean(comments.get(rid), L["max_chars"])
        cut += c
    return ids, texts, ref


def run_dev(bl, cfg, ids, texts, ref, args):
    """Đánh giá (và tuỳ chọn chỉnh ngưỡng) trên tập phát triển tự gán nhãn, tách khỏi 100 nhãn tay."""
    dev = pd.read_csv(ROOT / args.dev if not str(args.dev).startswith("/") else args.dev, dtype={"review_id": "int64"})
    dev = dev[dev["sentiment"].isin(LABELS)]
    hand = set(int(i) for i in ref.loc[ref["Nhãn tay"].str.lower() == "x", "review_id"])
    overlap = hand & set(dev["review_id"])
    if overlap:
        sys.exit(f"Tập phát triển giao với 100 nhãn tay ({len(overlap)} review): sẽ gây rò rỉ. Loại chúng ra khỏi {args.dev}.")
    an = {rid: bl.analyze(texts[rid]) for rid in dev["review_id"] if rid in texts}
    dev = dev[dev["review_id"].isin(an)]
    y = list(dev["sentiment"])

    def score(P):
        return f1_report(y, [bl.sentiment(an[r], P) for r in dev["review_id"]])

    cur = score(bl.B)
    print(f"Tập phát triển: n={cur['n']}  accuracy={cur['accuracy']:.3f}  macro-F1={cur['macro_f1']:.3f} (ngưỡng hiện tại trong YAML)")
    print(cur["cm"])
    if args.tune:
        grid = {"pos_threshold": [0.05, 0.1, 0.2, 0.3], "neg_threshold": [-0.05, -0.1, -0.2],
                "mixed_min_pos": [0.3, 0.5, 0.8], "mixed_min_neg": [0.2, 0.4, 0.6]}
        res = []
        for vals in itertools.product(*grid.values()):
            P = {**bl.B, **dict(zip(grid, vals))}
            r = score(P)
            res.append((r["macro_f1"], r["accuracy"], dict(zip(grid, vals))))
        res.sort(key=lambda t: (-t[0], -t[1]))
        print("\n5 bộ ngưỡng tốt nhất trên tập phát triển (macro-F1, accuracy, tham số):")
        for m, a, p in res[:5]:
            print(f"  {m:.3f}  {a:.3f}  {p}")
        print(f"\nTập chỉ {cur['n']} review: chênh lệch nhỏ giữa các bộ là nhiễu. Nếu đổi, sửa tay vào YAML, "
              "ghi rõ trong báo cáo là ngưỡng chỉnh trên tập phát triển, rồi chạy lại không kèm --dev.")


def main(config_path, args):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    if "baseline" not in cfg:
        sys.exit(f"{config_path} thiếu khối `baseline:` (ngưỡng và từ khoá khía cạnh). "
                 "Chép bản configs/cape_town.yaml mới có khối này, hoặc dán khối đó vào trước dòng `seed:`.")
    B, L = cfg["baseline"], cfg["llm"]
    ids, texts, ref = load_sample(cfg, args)
    bl = Baseline(cfg)

    if args.sample_dev:
        hand = set(int(i) for i in ref.loc[ref["Nhãn tay"].str.lower() == "x", "review_id"])
        pool = pd.Series([i for i in ids if i not in hand and texts[i] and not AUTO.match(texts[i])])
        pick = pool.sample(min(args.sample_dev, len(pool)), random_state=cfg["seed"])
        out = ROOT / "data/samples/dev_to_label.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"review_id": pick, "comments": [texts[i] for i in pick], "sentiment": ""}).to_csv(out, index=False)
        print(f"Đã ghi {len(pick)} review (không thuộc 100 nhãn tay) vào {out}. Điền cột sentiment (positive/mixed/negative), "
              "lưu thành dev_labeled.csv rồi chạy với --dev.")
        return 0
    if args.dev:
        run_dev(bl, cfg, ids, texts, ref, args)
        return 0

    rows = []
    for rid in ids:
        t = texts[rid]
        if not t:  # review rỗng: giống llm.py (status = empty, không có nhãn)
            rows.append({"review_id": rid, "sentiment": None, "language": None, "status": "empty",
                         "auto_generated": False, "sentiment_floor": None, **{f"aspect_{a}": None for a in bl.aspects}})
            continue
        an = bl.analyze(t)
        rows.append({"review_id": rid, "sentiment": bl.sentiment(an), "language": bl.language(t), "status": "ok",
                     "auto_generated": bool(AUTO.match(t)), "sentiment_floor": B["majority_label"],
                     **{f"aspect_{a}": p for a, p in bl.aspect_polarity(an).items()}})
    df = pd.DataFrame(rows)
    if args.out:
        out = ROOT / args.out
    elif args.input:  # chạy thử trên tệp nhỏ: không ghi đè kết quả chính
        out = ROOT / args.input
        out = out.with_name(out.stem + "_baseline.csv")
    else:
        out = ROOT / "data/processed/baseline_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    ok = df[df["status"] == "ok"]
    print(f"Done: {len(ok)} review ok, {(df['status'] == 'empty').sum()} rỗng, {int(ok['auto_generated'].sum())} do hệ thống sinh -> {out}")
    print("sentiment:", ok["sentiment"].value_counts().to_dict())
    print("language :", ok["language"].value_counts().head(6).to_dict(), f"| unknown={(ok['language'] == 'unknown').mean():.0%}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/cape_town.yaml")
    ap.add_argument("--limit", type=int, help="chỉ xử lý N review đầu (thử nhanh)")
    ap.add_argument("--input", help="CSV nhỏ (cột review_id, comments), ví dụ data/samples/x.csv; thay cho reviews.parquet")
    ap.add_argument("--out", help="đường dẫn CSV đầu ra (tính từ gốc repo)")
    ap.add_argument("--sample-dev", type=int, metavar="N", help="chọn N review ngoài 100 nhãn tay để tự gán nhãn làm tập phát triển")
    ap.add_argument("--dev", help="CSV tập phát triển đã gán nhãn (review_id, sentiment): in accuracy/macro-F1")
    ap.add_argument("--tune", action="store_true", help="với --dev: dò lưới ngưỡng trên tập phát triển")
    a = ap.parse_args()
    sys.exit(main(a.config, a))