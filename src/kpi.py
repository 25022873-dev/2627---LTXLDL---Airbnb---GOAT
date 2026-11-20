"""Tính 5 KPI theo vùng: python src/kpi.py configs/cape_town.yaml  (chạy sau llm.py)
Vào: data/processed/{listings_clean,reviews_clean,llm_results}.csv; tham số ở mục `kpi:` của YAML.
Ra: data/processed/kpi_<tên>.csv. Dạng dài: kpi, snapshot, level, group, room_type, value, n (số mẫu), currency (tỉ lệ lưu dạng 0-1)."""
import sys

import pandas as pd

from common import PROC, run

S, OCC, REV = "snapshot", "estimated_occupancy_l365d", "estimated_revenue_l365d"
LEVELS = ["analysis_area", "neighbourhood_cleansed"]  # cấp vùng: khu vực được giao và phường
COLS = ["kpi", S, "level", "group", "room_type", "value", "n", "currency"]
SCORE = {"positive": 1, "mixed": 0, "negative": -1}
MONEY = {"median_price", "median_revenue"}  # KPI có đơn vị tiền tệ


def agg(d, col, fn, min_n, by=(S,)):
    """fn(col) theo từng cấp vùng (+ by); bỏ dòng thiếu col và nhóm dưới min_n quan sát."""
    d, parts = d.dropna(subset=[col]), []
    for lvl in LEVELS:
        t = d.groupby([*by, lvl])[col].agg(value=fn, n="size").round(4).reset_index()
        parts.append(t.rename(columns={lvl: "group"}).assign(level=lvl))
    out = pd.concat(parts)
    return out[out["n"] >= min_n]


def scored_reviews(df):
    """KPI 5 và biểu đồ cảm xúc: review (status ok, không phải review tự động) -> listing -> vùng ở mốc mới nhất, kèm điểm cảm xúc."""
    info = df.sort_values(S).drop_duplicates("id", keep="last").set_index("id")[LEVELS]  # mới nhất theo ngày, không theo thứ tự trong config
    info.index = info.index.astype(str)
    rv = pd.read_csv(PROC / "reviews_clean.csv", usecols=["id", "listing_id"], dtype=str)
    z = pd.read_csv(PROC / "llm_results.csv", dtype=str).query("status == 'ok' and auto_generated == 'False'").merge(rv, left_on="review_id", right_on="id")
    return z.join(info, on="listing_id").assign(score=lambda d: d["sentiment"].map(SCORE))


def main(cfg):
    cur, K = cfg["currency"], cfg["kpi"]
    n = K["min_n"]
    df = pd.read_csv(PROC / "listings_clean.csv")
    cover = df.groupby(S)["price"].apply(lambda s: s.notna().mean())  # tỉ lệ listing có giá ở từng mốc
    usable = cover[cover >= K["min_price_coverage"]].index
    if usable.empty:
        sys.exit("Không mốc nào có đủ listing có giá: hạ kpi.min_price_coverage hoặc xem qa_report.csv (rule price_empty).")
    for snap in cover.index.difference(usable):
        print(f"Bỏ mốc {snap} khỏi KPI giá/doanh thu: chỉ {cover[snap]:.0%} listing có giá.")
    sel = df[df[S].isin(usable)]
    kpis = {
        "hibernation_rate": agg(df, OCC, lambda s: (s == 0).mean(), n),  # tỉ lệ listing có occupancy = 0
        "median_price": agg(sel, "price", "median", n, (S, "room_type")),
        "median_revenue": agg(sel, REV, "median", n),  # gồm cả listing doanh thu 0
        "median_occupied_nights": agg(df[df[OCC] > 0], OCC, "median", n),
        "net_sentiment": agg(scored_reviews(df), "score", "mean", n, by=()),
    }
    for name, k in kpis.items():
        k = k.assign(kpi=name, currency=cur if name in MONEY else "", **{S: k.get(S, "all")}).reindex(columns=COLS)
        k.to_csv(PROC / f"kpi_{name}.csv", index=False)
        print(f"kpi_{name}.csv: {len(k)} dòng")


if __name__ == "__main__":
    run(main)