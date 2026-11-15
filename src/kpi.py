"""Tính 5 KPI theo vùng: python src/kpi.py configs/cape_town.yaml  (chạy sau llm.py)
Vào: data/processed/{listings_clean,reviews_clean,llm_results}.csv. Ra: data/processed/kpi_<tên>.csv.
Dạng dài: kpi, snapshot, level, group, room_type, value, n (số mẫu), currency (tỉ lệ lưu dạng 0-1)."""
import sys

import pandas as pd
import yaml

from download import ROOT

S, OCC, REV = "snapshot", "estimated_occupancy_l365d", "estimated_revenue_l365d"
LEVELS = ["analysis_area", "neighbourhood_cleansed"]  # cấp vùng: khu vực được giao và phường
COLS = ["kpi", S, "level", "group", "room_type", "value", "n", "currency"]
SCORE = {"positive": 1, "mixed": 0, "negative": -1}
PRICE_MONTHS = ["09", "03", "06"]  # KPI giá và doanh thu chỉ lấy các mốc tháng 9, 3, 6
MIN_N = 10  # nhóm ít hơn MIN_N quan sát thì không báo cáo (trung vị/tỉ lệ quá nhiễu)
MONEY = {"median_price", "median_revenue"}  # KPI có đơn vị tiền tệ


def agg(d, col, fn, by=(S,)):
    """fn(col) theo từng cấp vùng (+ by); bỏ dòng thiếu col và nhóm dưới MIN_N quan sát."""
    d, parts = d.dropna(subset=[col]), []
    for lvl in LEVELS:
        t = d.groupby([*by, lvl])[col].agg(value=fn, n="size").round(4).reset_index()
        parts.append(t.rename(columns={lvl: "group"}).assign(level=lvl))
    out = pd.concat(parts)
    return out[out["n"] >= MIN_N]


def main(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    proc, cur = ROOT / "data" / "processed", cfg["currency"]
    llm_file = proc / "llm_results.csv"
    if not llm_file.exists():
        sys.exit(f"Thiếu {llm_file}: chạy llm.py trước kpi.py.")
    df = pd.read_csv(proc / "listings_clean.csv")
    pick = df[S].astype(str).str[5:7].isin(PRICE_MONTHS)
    sel = df[pick] if pick.any() else df  # mốc lạ (ví dụ giảng viên đổi config): dùng mọi mốc
    # KPI 5: review -> listing -> vùng (lấy ở mốc mới nhất); chỉ review có status ok
    info = df.drop_duplicates("id", keep="last").set_index("id")[LEVELS]
    info.index = info.index.astype(str)
    rv = pd.read_csv(proc / "reviews_clean.csv", usecols=["id", "listing_id"], dtype=str)
    z = pd.read_csv(llm_file, dtype=str).query("status == 'ok'").merge(rv, left_on="review_id", right_on="id")
    z = z.join(info, on="listing_id").assign(score=lambda d: d["sentiment"].map(SCORE))
    kpis = {
        "hibernation_rate": agg(df, OCC, lambda s: (s == 0).mean()),  # tỉ lệ listing có occupancy = 0
        "median_price": agg(sel, "price", "median", (S, "room_type")),
        "median_revenue": agg(sel, REV, "median"),  # gồm cả listing doanh thu 0
        "median_occupied_nights": agg(df[df[OCC] > 0], OCC, "median"),
        "net_sentiment": agg(z, "score", "mean", by=()),
    }
    for name, k in kpis.items():
        k = k.assign(kpi=name, currency=cur if name in MONEY else "", **{S: k.get(S, "all")}).reindex(columns=COLS)
        k.to_csv(proc / f"kpi_{name}.csv", index=False)
        print(f"kpi_{name}.csv: {len(k)} dòng")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/cape_town.yaml")