"""Chấm LLM và baseline trên nhãn tay: accuracy, F1 theo lớp, macro-F1, ma trận nhầm lẫn, khoảng tin cậy bootstrap,
McNemar, tách theo ngôn ngữ / độ dài, phân tích lỗi. Chỉ đọc kết quả có sẵn: không gọi API, không cần mạng.

Chạy: python src/evaluate.py configs/cape_town.yaml
      python src/evaluate.py configs/cape_town.yaml --template   (sinh tệp mẫu 100 review để gán nhãn tay)
Vào:  data/processed/llm_results.csv, baseline_results.csv; tệp nhãn tay (eval.hand_labels trong YAML: cột review_id, sentiment,
      tuỳ chọn aspect_<tên> nếu có nhãn khía cạnh); comments lấy từ reviews.parquet như llm.py.
Ra:   data/processed/eval_metrics.csv    (scope, method, group, metric, value, n)
      data/processed/eval_confusion.csv  (scope, method, true, pred, count)
      data/processed/eval_errors.csv     (review sai + cột `cause` để điền tay khi phân tích lỗi)

Quy ước chấm:
- Mọi phương pháp được chấm trên CÙNG một tập review: có nhãn tay, baseline ok, không phải review do hệ thống sinh
  (nếu eval.exclude_auto) và, ở scope `same_set`, LLM ok. Scope `llm_error_as_wrong` giữ review LLM lỗi và tính là đoán sai
  (chỉ có khi có review LLM lỗi). Hai scope cho hai phiên bản kết quả.
- floor = mức A: luôn đoán baseline.majority_label.
- macro-F1 tính trên các lớp có mặt trong nhãn tay của nhóm đang chấm (lớp vắng thì không tính).
- Nhóm ngôn ngữ lấy từ LLM (không có thì từ baseline); `non_en` gồm cả ngôn ngữ chưa xác định.
- Bootstrap: lấy mẫu lại có hoàn lại, seed từ YAML. McNemar: kiểm định chính xác hai phía trên cặp LLM - baseline.
"""
import math
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from baseline import LABELS, load_sample
from download import ROOT

LABEL_COLS = ["sentiment", "label", "nhan", "nhãn", "nhãn tay"]


def read_hand(path, what):
    """Đọc và kiểm tra tệp nhãn tay -> DataFrame (index review_id, cột sentiment [+ aspect_*])."""
    p = ROOT / path
    if not p.exists():
        sys.exit(f"Chưa có {what}: {p}. Tạo tệp CSV (cột review_id, sentiment) và khai báo đường dẫn ở eval trong YAML.")
    df = pd.read_csv(p, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    col = next((c for c in LABEL_COLS if c in df.columns), None)
    if "review_id" not in df.columns or col is None:
        sys.exit(f"{p} cần cột review_id và một trong {LABEL_COLS}. Hiện có: {list(df.columns)}")
    df = df.rename(columns={col: "sentiment"})
    df["review_id"] = pd.to_numeric(df["review_id"], errors="coerce")
    if df["review_id"].isna().any():
        sys.exit(f"{p}: có review_id trống hoặc không phải số.")
    df["review_id"] = df["review_id"].astype("int64")
    df["sentiment"] = df["sentiment"].fillna("").str.strip().str.lower()
    bad = df[~df["sentiment"].isin(LABELS)]
    if len(bad):
        sys.exit(f"{p}: {len(bad)} nhãn trống/sai (chỉ nhận {LABELS}). Ví dụ: "
                 + ", ".join(f"{r.review_id}={r.sentiment!r}" for r in bad.head(5).itertuples()))
    dup = df[df["review_id"].duplicated()]["review_id"].tolist()
    if dup:
        sys.exit(f"{p}: review_id bị trùng: {dup[:5]}")
    return df.set_index("review_id")


def read_results(name):
    f = ROOT / "data/processed" / name
    if not f.exists():
        return None
    df = pd.read_csv(f)
    return df.set_index("review_id")


def codes(s):
    """positive/mixed/negative -> 0/1/2; mọi giá trị khác ('error', ...) -> 3 (luôn tính là sai)."""
    m = {l: i for i, l in enumerate(LABELS)}
    return np.array([m.get(v, 3) for v in s])


def f1_arr(tp, fp, fn):
    return np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def class_counts(y, p, k, axis=None):
    return ((y == k) & (p == k)).sum(axis), ((y != k) & (p == k)).sum(axis), ((y == k) & (p != k)).sum(axis)


def cohen_kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def mcnemar_exact(b, c):
    """p hai phía, kiểm định nhị thức chính xác trên b + c cặp bất đồng."""
    m = b + c
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(min(b, c) + 1)) / 2 ** m
    return min(1.0, 2 * tail)


class Out:
    def __init__(self):
        self.rows, self.conf, self.scope = [], [], None

    def add(self, method, group, metric, value, n):
        self.rows.append({"scope": self.scope, "method": method, "group": group, "metric": metric,
                          "value": value, "n": n})


def point_metrics(out, method, group, y, p):
    present = [k for k in range(3) if (y == k).any()]
    f1s = []
    out.add(method, group, "accuracy", float((y == p).mean()), len(y))
    for k in present:
        tp, fp, fn = class_counts(y, p, k)
        prec, rec = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn)
        f1s.append(float(f1_arr(np.array(tp), np.array(fp), np.array(fn))))
        for name, v in (("precision", prec), ("recall", rec), ("f1", f1s[-1])):
            out.add(method, group, f"{name}_{LABELS[k]}", float(v), int((y == k).sum()))
    out.add(method, group, "macro_f1", float(np.mean(f1s)), len(y))


def bootstrap(out, y, preds, E, rng):
    """CI cho accuracy, macro-F1 của từng phương pháp, và CI chênh lệch LLM - baseline (cùng mẫu lại)."""
    B, lvl = E["bootstrap"], E["ci_level"]
    lo, hi = (1 - lvl) / 2 * 100, (1 + lvl) / 2 * 100
    present = [k for k in range(3) if (y == k).any()]
    idx = rng.integers(0, len(y), (B, len(y)))
    Y, res = y[idx], {}
    for m, p in preds.items():
        P = p[idx]
        f1 = np.mean([f1_arr(*class_counts(Y, P, k, axis=1)) for k in present], axis=0)
        res[m] = {"accuracy": (Y == P).mean(1), "macro_f1": f1}
        for met, arr in res[m].items():
            out.add(m, "all", f"{met}_ci_low", float(np.percentile(arr, lo)), len(y))
            out.add(m, "all", f"{met}_ci_high", float(np.percentile(arr, hi)), len(y))
    if "llm" in res and "baseline" in res:
        for met in ("accuracy", "macro_f1"):
            d = res["llm"][met] - res["baseline"][met]
            out.add("llm_minus_baseline", "all", f"{met}_diff_ci_low", float(np.percentile(d, lo)), len(y))
            out.add("llm_minus_baseline", "all", f"{met}_diff_ci_high", float(np.percentile(d, hi)), len(y))


def evaluate_scope(out, scope, d, methods, E, rng):
    out.scope = scope
    y = codes(d["true"])
    preds = {m: codes(d[f"pred_{m}"]) for m in methods}
    for m, p in preds.items():
        point_metrics(out, m, "all", y, p)
        for (t, pr), c in Counter(zip(d["true"], d[f"pred_{m}"])).items():
            out.conf.append({"scope": scope, "method": m, "true": t, "pred": pr, "count": c})
    bootstrap(out, y, preds, E, rng)
    if "llm" in preds and "baseline" in preds:
        l_ok, b_ok = preds["llm"] == y, preds["baseline"] == y
        b, c = int((l_ok & ~b_ok).sum()), int((~l_ok & b_ok).sum())  # b: chỉ LLM đúng; c: chỉ baseline đúng
        out.add("llm_vs_baseline", "all", "only_llm_correct", b, len(y))
        out.add("llm_vs_baseline", "all", "only_baseline_correct", c, len(y))
        out.add("llm_vs_baseline", "all", "both_wrong", int((~l_ok & ~b_ok).sum()), len(y))
        out.add("llm_vs_baseline", "all", "mcnemar_p", mcnemar_exact(b, c), b + c)
    lang = d["language_eval"].map(lambda v: "en" if v == "en" else "non_en")
    length = pd.cut(d["n_chars"], [-1, *E["length_bins"], 10 ** 9], labels=["short", "medium", "long"])
    for gname, series in (("language", lang), ("length", length)):
        for gv in series.dropna().unique():
            mask = (series == gv).to_numpy()
            for m, p in preds.items():
                point_metrics_short(out, m, f"{gname}={gv}", y[mask], p[mask])


def point_metrics_short(out, method, group, y, p):
    present = [k for k in range(3) if (y == k).any()]
    f1 = np.mean([float(f1_arr(*map(np.array, class_counts(y, p, k)))) for k in present])
    out.add(method, group, "accuracy", float((y == p).mean()), len(y))
    out.add(method, group, "macro_f1", float(f1), len(y))


def evaluate_aspects(out, hand, same, results, aspects):
    """Chỉ chạy khi nhãn tay có cột aspect_<tên>. Phát hiện khía cạnh (P/R/F1) và độ đúng phân cực khi cả hai cùng nhắc."""
    out.scope = "same_set"
    for a in aspects:
        col = f"aspect_{a}"
        if col not in hand.columns:
            continue
        h = hand.loc[same, col].fillna("").str.strip().str.lower().replace("", np.nan)
        for m, res in results.items():
            pr = res[col].reindex(same)
            hm, pm = h.notna().to_numpy(), pr.notna().to_numpy()
            tp, fp, fn = int((hm & pm).sum()), int((~hm & pm).sum()), int((hm & ~pm).sum())
            out.add(m, f"aspect={a}", "aspect_detect_precision", tp / (tp + fp) if tp + fp else 0.0, int(hm.sum()))
            out.add(m, f"aspect={a}", "aspect_detect_recall", tp / (tp + fn) if tp + fn else 0.0, int(hm.sum()))
            out.add(m, f"aspect={a}", "aspect_detect_f1", float(f1_arr(np.array(tp), np.array(fp), np.array(fn))), int(hm.sum()))
            both = hm & pm
            if both.any():
                out.add(m, f"aspect={a}", "aspect_polarity_acc", float((h[both].to_numpy() == pr[both].to_numpy()).mean()), int(both.sum()))


def show(out, scope, methods):
    df = pd.DataFrame(out.rows)
    df = df[(df["scope"] == scope) & (df["group"] == "all")].set_index(["method", "metric"])["value"]
    n = int(pd.DataFrame(out.rows).query("scope == @scope and metric == 'accuracy'")["n"].iloc[0])
    print(f"\n== scope {scope} (n={n}) ==")
    for m in methods:
        g = lambda k: df.get((m, k), float("nan"))
        ci = (f"  CI95 acc [{g('accuracy_ci_low'):.3f}, {g('accuracy_ci_high'):.3f}] macro-F1 [{g('macro_f1_ci_low'):.3f}, {g('macro_f1_ci_high'):.3f}]"
              if ("accuracy_ci_low" in df.xs(m, level=0).index) else "")
        print(f"{m:9s} accuracy={g('accuracy'):.3f} macro-F1={g('macro_f1'):.3f}{ci}")
    if ("llm_vs_baseline", "mcnemar_p") in df.index:
        print(f"LLM đúng/baseline sai={int(df[('llm_vs_baseline', 'only_llm_correct')])}, "
              f"ngược lại={int(df[('llm_vs_baseline', 'only_baseline_correct')])}, "
              f"cả hai sai={int(df[('llm_vs_baseline', 'both_wrong')])}, McNemar p={df[('llm_vs_baseline', 'mcnemar_p')]:.3f}")
        print(f"Chênh lệch LLM - baseline (CI): accuracy [{df[('llm_minus_baseline', 'accuracy_diff_ci_low')]:+.3f}, "
              f"{df[('llm_minus_baseline', 'accuracy_diff_ci_high')]:+.3f}], macro-F1 [{df[('llm_minus_baseline', 'macro_f1_diff_ci_low')]:+.3f}, "
              f"{df[('llm_minus_baseline', 'macro_f1_diff_ci_high')]:+.3f}]")


def main(config_path):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    if "eval" not in cfg or "baseline" not in cfg:
        sys.exit("YAML thiếu khối `eval:` hoặc `baseline:`. Chép bản configs/cape_town.yaml mới.")
    E, aspects = cfg["eval"], cfg["llm"]["aspects"]
    rng = np.random.default_rng(cfg["seed"])
    if "--template" in sys.argv:
        _, texts, ref = load_sample(cfg, SimpleNamespace(input=None, limit=None))
        ids = [int(i) for i in ref.loc[ref["Nhãn tay"].str.lower() == "x", "review_id"]]
        dest = ROOT / "data/samples/hand_labels_template.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"review_id": ids, "comments": [texts.get(i, "") for i in ids], "sentiment": ""}).to_csv(
            dest, index=False, encoding="utf-8-sig")
        print(f"Đã ghi {len(ids)} review cần gán nhãn vào {dest}.\nĐiền cột sentiment (positive/mixed/negative) trước khi xem "
              f"kết quả LLM, rồi lưu thành {E['hand_labels']} (có thể giữ hoặc xoá cột comments).")
        return 0
    hand_df = read_hand(E["hand_labels"], "tệp nhãn tay")
    base, llm = read_results("baseline_results.csv"), read_results("llm_results.csv")
    if base is None:
        sys.exit("Thiếu data/processed/baseline_results.csv. Chạy: python src/baseline.py configs/cape_town.yaml")
    if llm is None:
        print("CẢNH BÁO: chưa có llm_results.csv (chạy src/llm.py trước): chỉ chấm baseline và mức sàn A.")
    _, texts, ref = load_sample(cfg, SimpleNamespace(input=None, limit=None))

    # --- kiểm tra đầu vào ---
    ref_hand = set(int(i) for i in ref.loc[ref["Nhãn tay"].str.lower() == "x", "review_id"])
    if set(hand_df.index) != ref_hand:
        print(f"CẢNH BÁO: nhãn tay ({len(hand_df)}) khác danh sách «Nhãn tay» trong tệp phân công ({len(ref_hand)}): "
              f"thiếu {len(ref_hand - set(hand_df.index))}, thừa {len(set(hand_df.index) - ref_hand)}.")
    outside = [i for i in hand_df.index if i not in base.index]
    if outside:
        print(f"CẢNH BÁO: {len(outside)} review có nhãn tay nhưng không nằm trong mẫu LLM (bỏ qua): {outside[:5]}")
    hand_df = hand_df.loc[[i for i in hand_df.index if i in base.index]]
    if len(hand_df) < 100:
        print(f"CẢNH BÁO: chỉ {len(hand_df)} nhãn tay dùng được (đề yêu cầu ít nhất 100).")

    # --- bảng chung: nhãn tay + dự đoán của từng phương pháp ---
    d = pd.DataFrame({"true": hand_df["sentiment"]})
    d["text"] = [texts.get(i, "") for i in d.index]
    d["n_chars"] = d["text"].str.len()
    d["b_status"], d["b_auto"] = base["status"].reindex(d.index), base["auto_generated"].reindex(d.index).fillna(False).astype(bool)
    d["pred_baseline"] = base["sentiment"].reindex(d.index).where(d["b_status"] == "ok")
    d["pred_floor"] = np.where(d["b_status"] == "ok", cfg["baseline"]["majority_label"], None)
    d["language_eval"], d["l_auto"] = base["language"].reindex(d.index), False
    methods = ["floor", "baseline"]
    if llm is not None:
        d["l_status"] = llm["status"].reindex(d.index).fillna("missing")
        d["l_auto"] = llm["auto_generated"].reindex(d.index).fillna(False).astype(bool)
        d["language_eval"] = llm["language"].reindex(d.index).fillna(d["language_eval"])
        d["pred_llm"] = llm["sentiment"].reindex(d.index).where(d["l_status"] == "ok", "error")
        d.loc[d["l_status"] == "empty", "pred_llm"] = np.nan
        methods = ["floor", "baseline", "llm"]
        n_miss = int((d["l_status"] == "missing").sum())
        if n_miss:
            print(f"CẢNH BÁO: {n_miss} review có nhãn tay nhưng không có kết quả LLM (tính là lỗi ở scope llm_error_as_wrong).")
    keep = d["b_status"] == "ok"
    if E["exclude_auto"]:
        auto = d["b_auto"] | d["l_auto"]
        print(f"Loại {int((keep & auto).sum())} review do hệ thống sinh khỏi cả hai phương pháp.")
        keep &= ~auto
    if llm is not None:
        keep &= d["pred_llm"].notna()  # review rỗng ở phía LLM: không có gì để chấm
    d = d[keep]

    out = Out()
    if llm is not None:
        n_err = int((d["pred_llm"] == "error").sum())
        print(f"Review LLM lỗi/thiếu trong tập chấm: {n_err}/{len(d)}")
        same = d[d["pred_llm"].isin(LABELS)]
        evaluate_scope(out, "same_set", same, methods, E, rng)
        if n_err:
            evaluate_scope(out, "llm_error_as_wrong", d, methods, E, rng)
        else:
            print("Không có review LLM lỗi: hai scope trùng nhau, chỉ báo cáo same_set.")
    else:
        same = d
        evaluate_scope(out, "same_set", d, methods, E, rng)
    if cfg["baseline"]["majority_label"] != d["true"].mode().iloc[0]:
        print(f"LƯU Ý: lớp phổ biến nhất trong nhãn tay là {d['true'].mode().iloc[0]!r}, khác majority_label trong YAML "
              f"({cfg['baseline']['majority_label']!r}); mức sàn A vẫn dùng giá trị trong YAML (cố định từ trước).")

    # --- khía cạnh (nếu có nhãn tay khía cạnh) ---
    if any(f"aspect_{a}" in hand_df.columns for a in aspects):
        evaluate_aspects(out, hand_df, same.index, {"baseline": base, **({"llm": llm} if llm is not None else {})}, aspects)
    else:
        print("Nhãn tay không có cột aspect_<tên>: khía cạnh chưa được kiểm chứng bằng nhãn tay.")

    # --- độ đồng thuận giữa người, giữa LLM và baseline trên cả mẫu ---
    out.scope = "extra"
    if E.get("hand_labels_2"):
        h2 = read_hand(E["hand_labels_2"], "tệp nhãn tay của người thứ hai")["sentiment"]
        common = hand_df.index.intersection(h2.index)
        out.add("annotator1_vs_annotator2", "all", "agreement", float((hand_df.loc[common, "sentiment"] == h2[common]).mean()), len(common))
        out.add("annotator1_vs_annotator2", "all", "cohen_kappa", cohen_kappa(hand_df.loc[common, "sentiment"], h2[common]), len(common))
    if llm is not None:
        both = pd.DataFrame({"l": llm["sentiment"].where(llm["status"] == "ok"), "b": base["sentiment"].where(base["status"] == "ok"),
                             "auto": base["auto_generated"].fillna(False).astype(bool) | llm["auto_generated"].fillna(False).astype(bool)}).dropna(subset=["l", "b"])
        if E["exclude_auto"]:
            both = both[~both["auto"]]
        out.add("llm_vs_baseline", "all_sample", "agreement", float((both["l"] == both["b"]).mean()), len(both))
        out.add("llm_vs_baseline", "all_sample", "cohen_kappa", cohen_kappa(both["l"], both["b"]), len(both))

    # --- danh sách review sai (để phân tích lỗi) ---
    if llm is not None:
        w = d.copy()
        lw, bw = w["pred_llm"] != w["true"], w["pred_baseline"] != w["true"]
        w["error_group"] = np.select([lw & bw, lw, bw], ["both_wrong", "llm_only_wrong", "baseline_only_wrong"], "both_right")
    else:
        w = d.copy()
        w["pred_llm"], w["error_group"] = np.nan, np.where(w["pred_baseline"] != w["true"], "baseline_wrong", "both_right")
    w = w[w["error_group"] != "both_right"]
    err = pd.DataFrame({"review_id": w.index.to_numpy(), "error_group": w["error_group"].to_numpy(), "true": w["true"].to_numpy(), "llm": w["pred_llm"].to_numpy(),
                        "baseline": w["pred_baseline"].to_numpy(), "language": w["language_eval"].to_numpy(),
                        "n_chars": w["n_chars"].to_numpy(), "excerpt": w["text"].str.slice(0, 300).to_numpy(), "cause": ""})

    dest = ROOT / "data/processed"
    pd.DataFrame(out.rows).to_csv(dest / "eval_metrics.csv", index=False)
    pd.DataFrame(out.conf).to_csv(dest / "eval_confusion.csv", index=False)
    err.sort_values(["error_group", "review_id"]).to_csv(dest / "eval_errors.csv", index=False)
    for sc in dict.fromkeys(r["scope"] for r in out.rows if r["scope"] != "extra"):
        show(out, sc, methods)
    print(f"\nĐã ghi eval_metrics.csv ({len(out.rows)} dòng), eval_confusion.csv, eval_errors.csv ({len(err)} review sai).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/cape_town.yaml"))