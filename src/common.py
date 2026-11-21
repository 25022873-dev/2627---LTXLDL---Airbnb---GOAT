"""Tiện ích dùng chung: đường dẫn, cấu hình, bảng tham chiếu Excel, làm sạch văn bản, điểm vào dòng lệnh."""
import argparse
import html
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(main):
    """Chạy một mô-đun từ dòng lệnh: <script> <config.yaml> [--co-nao]; cờ --a-b thành tham số a_b=True của main."""
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args, flags = ap.parse_known_args()
    sys.exit(main(load_cfg(args.config), **{f[2:].replace("-", "_"): True for f in flags}))


def load_refs(cfg):
    """ID/vùng của nhóm từ Excel, cache CSV nhỏ (có thể commit); có cache thì không cần Excel."""
    base, g = PROC / "refs", cfg["group_code"]
    base.mkdir(parents=True, exist_ok=True)
    refs = {}
    for sheet, key in (("Listing", "listing"), ("Review", "review"), ("Vùng", "area")):
        cache = base / f"{key}_{g}.csv"
        if not cache.exists():
            df = pd.read_excel(ROOT / cfg["ids_file"], sheet_name=sheet, dtype=str)
            mine = df["Thành phố"].str.lower().str.replace(" ", "_") == cfg["city"] if key == "area" else df["Mã dữ liệu"] == g
            df[mine].to_csv(cache, index=False)
        refs[key] = pd.read_csv(cache, dtype=str)
    return refs


def clean(text, max_chars):
    """Bỏ HTML, gộp khoảng trắng, cắt độ dài: mọi phương pháp (LLM, đối chứng) dùng chung."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()[:max_chars]


def sample_ids(cfg, col="Mẫu LLM"):
    """review_id được đánh dấu «x» ở cột col của sheet Review."""
    ref = load_refs(cfg)["review"]
    return ref.loc[ref[col].str.lower().eq("x"), "review_id"].astype(int).tolist()


def review_texts(cfg, ids=None):
    """{review_id: văn bản đã clean()}; mặc định là mẫu LLM. Review không có/rỗng cho chuỗi rỗng."""
    rv = pd.read_csv(PROC / "reviews_clean.csv", usecols=["id", "comments"]).set_index("id")["comments"]
    return {i: clean(rv.get(i), cfg["llm"]["max_chars"]) for i in (sample_ids(cfg) if ids is None else ids)}