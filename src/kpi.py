"""KPI từ dữ liệu sau QA (parquet do qa.py tạo).

Chạy: python src/kpi.py configs/cape_town.yaml [--skip-llm]
Ra:   data/processed/kpi_area.csv, kpi_time.csv, kpi_llm.csv, kpi_qa_impact.csv
Dạng dài: kpi, snapshot, level, group, value, n, currency (tỉ lệ lưu dạng 0-1).
KPI 6-7 đọc data/processed/<data_root>/llm_results.csv (cột review_id, sentiment, aspect) do llm.py sinh;
thiếu tệp hoặc có --skip-llm thì bỏ qua, không làm sập pipeline."""
import sys

import pandas as pd
import yaml

from download import ROOT, build_path
from qa import TYPES, cast, load_refs, read_raw

WINDOW, MIN_N = 90, 10  # cửa sổ lấp đầy (ngày); số quan sát tối thiểu để báo cáo một nhóm
LEVELS = ["analysis_area", "neighbourhood_cleansed"]
MONEY = {"median_price", "est_revenue_per_listing"}  # KPI có đơn vị tiền tệ


def tidy(t, kpi, snap, level, cur):
    """t: index = nhóm, hai cột cuối là value và n -> bảng dài."""
    t = t.reset_index()
    group = [" | ".join(map(str, r)) for r in t.iloc[:, :-2].itertuples(index=False)]
    return pd.DataFrame({"kpi": kpi, "snapshot": snap, "level": level, "group": group,
                         "value": t["value"], "n": t["n"], "currency": cur if kpi in MONEY else ""})


def by_level(df, keys, col, fn, kpi, snap, cur, min_n=MIN_N, levels=LEVELS):
    """Gộp `col` bằng fn theo từng cấp khu vực (+ keys); bỏ nhóm có ít hơn min_n quan sát."""
    parts = []
    for lvl in levels:
        t = df.groupby([lvl, *keys])[col].agg(value=fn, n="count")
        parts.append(tidy(t[t["n"] >= min_n], kpi, snap, lvl, cur))
    return parts


def occupancy(cal):
    """Lấp đầy ước tính từng listing trong WINDOW ngày đầu của lịch = 1 - tỉ lệ ngày available = t."""
    w = cal[cal["date"] < cal["date"].min() + pd.Timedelta(days=WINDOW)]
    return (~w["available"]).astype(float).groupby(w["listing_id"]).mean()


def overall(L, C):
    """Ba KPI toàn mẫu, dùng để so trước/sau QA."""
    occ = occupancy(C).rename("occ")
    m = L.join(occ, on="id")
    return {"median_price": m["price"].median(), "occupancy_90d": occ.mean(),
            "est_revenue_per_listing": (m["price"] * m["occ"] * 365).median()}


def main(config):
    cfg = yaml.safe_load(open(config, encoding="utf-8"))
    root, cur, snaps = cfg["data_root"].strip("/ "), cfg["currency"], [str(s) for s in cfg["snapshots"]]
    base = ROOT / "data/processed"
    load = lambda snap, name: pd.read_parquet(base / root / snap / f"{name}.parquet")
    area, time, llm = [], [], []
    for snap in snaps:
        try:
            L, C = load(snap, "listings"), load(snap, "calendar")
        except FileNotFoundError:
            print(f"[skip] {snap}: thiếu dữ liệu sau QA")
            continue
        per = L.set_index("id").join(occupancy(C).rename("occ"))  # mỗi dòng = một listing ở mốc này
        time += by_level(per, [], "occ", "mean", "occupancy_90d", snap, cur)  # KPI 2
        if snap != snaps[-1]:
            continue  # các KPI còn lại chỉ tính ở mốc cuối (price chỉ đáng tin ở mốc này)
        if "price" in per:
            area += by_level(per, ["room_type"], "price", "median", "median_price", snap, cur)  # KPI 1
            per["rev"] = per["price"] * per["occ"] * 365
            area += by_level(per, [], "rev", "median", "est_revenue_per_listing", snap, cur)  # KPI 3
        area += by_level(per, [], "review_scores_rating", "median", "review_score_median", snap, cur)  # KPI 5

        c = C.merge(L[["id", "analysis_area"]], left_on="listing_id", right_on="id")  # KPI 4: biên độ mùa vụ
        c["m"], c["busy"] = c["date"].dt.to_period("M").astype(str), (~c["available"]).astype(float)
        days = c.groupby("m")["date"].nunique()  # chỉ giữ tháng đủ ngày (tháng đầu/cuối của lịch bị cụt)
        mo = c[c["m"].isin(days[days >= 28].index)].groupby(["analysis_area", "m"])["busy"].mean().groupby(level=0)
        season = pd.DataFrame({"value": mo.max() / mo.min(), "n": L.groupby("analysis_area").size()})
        time.append(tidy(season, "seasonality_ratio", snap, "analysis_area", cur))

        R = load(snap, "reviews").merge(per[LEVELS], left_on="listing_id", right_index=True)  # KPI 5: review/tháng
        R["m"] = R["date"].dt.to_period("M").astype(str)
        time += by_level(R, ["m"], "id", "count", "reviews_per_month", snap, cur, min_n=1, levels=LEVELS[:1])

        f = base / root / "llm_results.csv"  # KPI 6-7
        if "--skip-llm" in sys.argv or not f.exists():
            print("[skip] KPI 6-7 (LLM)")
        else:
            z = pd.read_csv(f).merge(R, left_on="review_id", right_on="id")
            s = z.drop_duplicates("review_id").assign(score=lambda d: d["sentiment"].map({"positive": 1, "negative": -1}).fillna(0))
            llm += by_level(s, [], "score", "mean", "net_sentiment", snap, cur)
            neg = z.assign(neg=z["sentiment"].eq("negative")).groupby(["analysis_area", "aspect"])["neg"]
            neg = neg.agg(value="mean", n="count").query("n >= @MIN_N")
            llm.append(tidy(neg.loc[neg.groupby(level=0)["value"].idxmax()], "top_negative_aspect", snap, "analysis_area", cur))

        if "price" in L:  # tác động của QA: cùng KPI tính trên dữ liệu thô (đã lọc ID, đã ép kiểu) và sau QA
            refs = load_refs(cfg)
            raw = {n: read_raw(build_path(root, snap, f"{n}.csv.gz"), n, refs) for n in ("listings", "calendar")}
            typed = lambda n, cols: pd.DataFrame({c: cast(raw[n][c], TYPES[n][c])[0] for c in cols})
            imp = pd.DataFrame({"before_qa": overall(typed("listings", ["id", "price"]),
                                                     typed("calendar", ["listing_id", "date", "available"])),
                                "after_qa": overall(L, C)})
            imp["diff_pct"] = 100 * (imp["after_qa"] - imp["before_qa"]) / imp["before_qa"]
            imp.rename_axis("kpi").reset_index().to_csv(base / "kpi_qa_impact.csv", index=False)

    for name, parts in (("area", area), ("time", time), ("llm", llm)):
        if parts:
            pd.concat(parts).to_csv(base / f"kpi_{name}.csv", index=False)
    print(f"Done: {sum(len(p) for p in area + time + llm)} KPI rows -> data/processed/kpi_*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/cape_town.yaml"))
