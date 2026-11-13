"""Chấm LLM vs baseline (và mức sàn A) trên nhãn tay. Chỉ đọc kết quả có sẵn, không gọi API.

Chạy: python src/evaluate.py configs/cape_town.yaml
Tệp nhãn tay mặc định data/samples/hand_labels.csv (review_id, comments, label ∈ positive/mixed/negative).
Chưa có thì tự tạo từ «Mẫu LLM» (n_labels dòng, seed từ YAML), bạn điền cột label rồi chạy lại.
YAML: eval.labels (đường dẫn), eval.n_labels (số nhãn cần tạo), eval.bootstrap (số lần lấy mẫu lại).
Ra:   data/processed/eval_metrics.csv (method, group, metric, value, n), eval_confusion.csv, eval_errors.csv
Quy ước: review mà phương pháp không trả nhãn hợp lệ (status != ok, thiếu trong kết quả) tính là SAI."""
import sys

import numpy as np
import pandas as pd
import yaml

from download import ROOT
from llm import clean
from qa import load_refs

CLASSES = ["positive", "mixed", "negative"]


def f1(y, p, c):
    d = np.sum(y == c) + np.sum(p == c)
    return 2 * np.sum((y == c) & (p == c)) / d if d else 0.0


def macro_f1(y, p, classes):
    return np.mean([f1(y, p, c) for c in classes])


def metrics(y, p, per_class):
    classes = [c for c in CLASSES if (y == c).any()]  # chỉ lớp có trong nhãn tay
    rows = [("accuracy", np.mean(y == p), len(y)), ("macro_f1", macro_f1(y, p, classes), len(y))]
    for c in classes if per_class else []:
        tp, ns, npred = np.sum((y == c) & (p == c)), np.sum(y == c), np.sum(p == c)
        rows += [(f"precision_{c}", tp / npred if npred else 0.0, ns), (f"recall_{c}", tp / ns, ns),
                 (f"f1_{c}", f1(y, p, c), ns)]  # n = số review thật của lớp
    return rows


def kappa(a, b):
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in CLASSES)
    return (po - pe) / (1 - pe)


def make_sheet(cfg, path, comments):
    """Tạo tệp nhãn tay (ngẫu nhiên từ «Mẫu LLM», cố định theo seed) rồi dừng để người dùng điền nhãn."""
    ref = load_refs(cfg)["review"]
    ids = [int(x) for x in ref.loc[ref["Mẫu LLM"].str.lower() == "x", "review_id"]]
    ids = np.random.default_rng(cfg["seed"]).choice(ids, cfg["eval"]["n_labels"], replace=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"review_id": ids, "comments": [clean(comments.get(i), cfg["llm"]["max_chars"]) for i in ids],
                  "label": ""}).to_csv(path, index=False, encoding="utf-8-sig")
    sys.exit(f"Đã tạo {path}. Điền cột label (positive/mixed/negative) rồi chạy lại.\n"
             "Mở bằng VS Code hoặc Excel (Data > From Text/CSV, cột review_id chọn Text); mở thẳng bằng Excel sẽ làm hỏng ID.")


def main(config_path):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    E, root, snap0 = cfg["eval"], cfg["data_root"].strip("/ "), str(cfg["snapshots"][0])
    out = ROOT / "data/processed"
    comments = pd.read_parquet(out / root / snap0 / "reviews.parquet").set_index("id")["comments"]

    path = ROOT / E["labels"]
    if not path.exists():
        make_sheet(cfg, path, comments)
    lab = pd.read_csv(path, dtype={"review_id": "int64"}, usecols=["review_id", "label"])
    lab["label"] = lab["label"].astype(str).str.strip().str.lower()
    if lab["review_id"].duplicated().any() or not lab["label"].isin(CLASSES).all():
        sys.exit(f"{path.name}: review_id trùng, còn {int((~lab['label'].isin(CLASSES)).sum())} nhãn trống hoặc ngoài positive/mixed/negative.")
    df = lab.set_index("review_id")

    methods = ["llm", "baseline"]
    res = {m: pd.read_csv(out / f"{m}_results.csv", dtype={"review_id": "int64"}).set_index("review_id") for m in methods}
    for m, r in res.items():
        df[m] = r["sentiment"].where(r["status"] == "ok").reindex(df.index).fillna("none")
        print(f"{m}: thiếu/lỗi {np.sum(df[m] == 'none')}/{len(df)} review (tính là sai)")
    df["floor"] = df["label"].mode()[0]  # mức sàn A: luôn đoán lớp phổ biến nhất của nhãn tay

    txt = pd.Series({rid: clean(comments.get(rid), cfg["llm"]["max_chars"]) for rid in df.index})
    df["lang"] = np.where(res["llm"]["language"].reindex(df.index) == "en", "en", "other")
    df["length"] = pd.cut(txt.str.len(), [-1, 150, 400, 1e9], labels=["short", "mid", "long"]).astype(str)
    df["auto"] = res["llm"]["auto_generated"].reindex(df.index).fillna(False).astype(bool).map({True: "yes", False: "no"})

    groups = [("all", np.ones(len(df), bool))] + [(f"{c}={v}", (df[c] == v).to_numpy())
                                                 for c in ("lang", "length", "auto") for v in sorted(df[c].unique())]
    rows = []
    for g, mask in groups:
        d = df[mask]
        for m in methods + ["floor"]:
            rows += [(m, g, k, v, n) for k, v, n in metrics(d["label"].to_numpy(), d[m].to_numpy(), g == "all")]

    y, B = df["label"].to_numpy(), E["bootstrap"]  # khoảng tin cậy 95% bằng bootstrap
    idx = np.random.default_rng(cfg["seed"]).integers(0, len(df), (B, len(df)))
    cl = [c for c in CLASSES if (y == c).any()]
    for m in methods + ["floor"]:
        p = df[m].to_numpy()
        for k, fn in (("accuracy", lambda a, b: np.mean(a == b)), ("macro_f1", lambda a, b: macro_f1(a, b, cl))):
            lo, hi = np.percentile([fn(y[i], p[i]) for i in idx], [2.5, 97.5])
            rows += [(m, "all", f"{k}_ci_lo", lo, len(df)), (m, "all", f"{k}_ci_hi", hi, len(df))]

    # hai con số bất đồng (không kiểm định McNemar) + độ đồng thuận trên cả mẫu
    ok = {m: df[m] == df["label"] for m in methods}
    pair = {"both_right": ok["llm"] & ok["baseline"], "only_llm_right": ok["llm"] & ~ok["baseline"],
            "only_baseline_right": ~ok["llm"] & ok["baseline"], "both_wrong": ~ok["llm"] & ~ok["baseline"]}
    rows += [("llm_vs_baseline", "all", k, int(v.sum()), len(df)) for k, v in pair.items()]
    both = pd.concat({m: res[m]["sentiment"].where(res[m]["status"] == "ok") for m in methods}, axis=1).dropna()
    a, b = both["llm"].to_numpy(), both["baseline"].to_numpy()
    rows += [("llm_vs_baseline", "agreement_all_sample", "agreement", np.mean(a == b), len(both)),
             ("llm_vs_baseline", "agreement_all_sample", "cohen_kappa", kappa(a, b), len(both))]
    pd.DataFrame(rows, columns=["method", "group", "metric", "value", "n"]).to_csv(out / "eval_metrics.csv", index=False)

    conf = [df.groupby(["label", m]).size().rename("count").reset_index().rename(columns={m: "pred"}).assign(method=m)
            for m in methods + ["floor"]]
    pd.concat(conf)[["method", "label", "pred", "count"]].to_csv(out / "eval_confusion.csv", index=False)

    wrong = pd.concat({m: df[m] != df["label"] for m in methods}, axis=1)
    err = df[wrong.any(axis=1)].assign(wrong_in=wrong.apply(lambda r: "+".join(m for m in methods if r[m]), axis=1),
                                       excerpt=txt.str[:200])
    err[["label", *methods, "wrong_in", "lang", "length", "auto", "excerpt"]].reset_index().to_csv(
        out / "eval_errors.csv", index=False)
    print(f"Done: {len(rows)} số đo, {len(err)} review sai -> eval_metrics.csv, eval_confusion.csv, eval_errors.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/cape_town.yaml"))