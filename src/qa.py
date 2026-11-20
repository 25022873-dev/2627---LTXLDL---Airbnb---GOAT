"""QA và làm sạch listings + reviews: python src/qa.py configs/cape_town.yaml
Ra: data/processed/{listings,reviews}_clean.csv, rejected_*.csv, reports/qa_report.csv.
Excel chỉ đọc lần đầu; ID và vùng của nhóm được cache ở data/processed/refs/ (không cần commit Excel)."""

import numpy as np
import pandas as pd

from common import PROC, ROOT, load_refs, run
from download import build_path

ENC = "[ÃÂ][^A-Za-z\\s]|â€|\ufffd"  # dấu hiệu lỗi mã hoá (mojibake)
SENTINELS = {"n/a", "na", "nan", "null", "none", "-", "--", "unknown"}  # chuỗi đánh dấu "thiếu"
LIST_COLS = ["id", "neighbourhood_cleansed", "room_type", "price", "estimated_occupancy_l365d",
             "estimated_revenue_l365d", "number_of_reviews", "latitude", "longitude"]
# tên quy tắc -> (điều kiện, lý do, hành động); bảng/cột được truyền lúc ghi log
RULES = {
    "key_null": ("khoá rỗng hoặc không phải số nguyên", "không định danh được dòng", "không đọc vào"),
    "key_scope": ("khoá không thuộc danh sách ID của nhóm", "chỉ phân tích mẫu được giao", "không đọc vào"),
    "dup_exact": ("trùng mọi cột trong cùng mốc", "dòng lặp không thêm thông tin", "giữ 1 dòng, ghi rejected_*.csv"),
    "key_conflict": ("cùng khoá, nội dung khác nhau trong cùng mốc", "không biết bản nào đúng", "xoá mọi dòng của khoá, ghi rejected_*.csv"),
    "rows_in_out": ("số dòng trước và sau làm sạch", "đo tác động QA", "ghi nhận"),
    "ids_missing": ("ID có trong sheet nhưng không có trong mốc", "mốc thiếu ID của nhóm", "ghi nhận"),
    "dup_snapshot": ("cùng review id ở nhiều mốc", "đánh giá tích luỹ nên lặp giữa các mốc là bình thường", "giữ mốc mới nhất"),
    "text_empty": ("rỗng hoặc là chuỗi sentinel (n/a, null, ...)", "tránh nhóm rỗng giả", "đổi thành NA"),
    "encoding": ("chứa dấu hiệu lỗi mã hoá (Ã, Â, â€, �)", "văn bản có thể đã bị hỏng", "gắn cờ"),
    "no_area": ("id không có analysis_area trong bảng tham chiếu", "không gán được khu vực", "gắn cờ"),
    "price_empty": ("price rỗng", "không có giá", "NA"),
    "price_sentinel": ("price là chuỗi sentinel (n/a, null, ...)", "thiếu nhưng ghi bằng chữ", "NA"),
    "price_parse": ("price không ép được sang số", "định dạng lạ", "NA"),
    "price_min": ("price < 0.01", "giá không hợp lệ", "đổi thành NA"),
    "outlier": ("log1p ngoài [Q1-k*IQR, Q3+k*IQR] theo room_type, chỉ tính giá trị khác 0 (bỏ qua khi IQR = 0)", "giá trị cực đoan có thể là lỗi nhập", "gắn cờ, giữ giá trị"),
    "occ_range": ("ngoài [0, 365]", "số đêm trong 365 ngày", "đổi thành NA, giữ số 0"),
    "rev_negative": ("< 0", "doanh thu không âm", "đổi thành NA, giữ số 0"),
    "zero_kept": ("giá trị đúng bằng 0", "0 có thể là giá trị thật", "giữ nguyên, chỉ đếm"),
    "reviews_neg": ("không phải số nguyên >= 0", "số đánh giá là số đếm", "đổi thành NA"),
    "coord_range": ("|latitude| > 90 hoặc |longitude| > 180", "toạ độ không tồn tại", "đổi thành NA"),
    "coord_box": ("toạ độ nằm ngoài hộp của analysis_area", "sai vị trí hoặc sai khu vực", "gắn cờ"),
    "id_mismatch": ("listing_id khác sheet Review", "dữ liệu thô lệch bảng được giao", "gắn cờ"),
    "comment_empty": ("comments rỗng sau khi cắt khoảng trắng", "không có nội dung để phân tích", "đổi thành NA"),
    "expected": ("số liệu khác kỳ vọng trong qa.expected", "lọc sai ID hoặc dữ liệu đổi", "ghi ok/error, không dừng"),
}


def to_int(s):
    """Chuỗi -> Int64 qua int Python (tránh mất chính xác của float với ID 19 chữ số)."""
    s = s.str.strip().str.replace(r"\.0+$", "", regex=True)
    ok = s.str.fullmatch(r"-?\d+")
    return pd.Series([int(v) if g else pd.NA for v, g in zip(s, ok)], index=s.index, dtype="Int64")


def to_float(s):
    return pd.to_numeric(s, errors="coerce")


def to_money(s):  # bỏ ký hiệu tiền tệ và dấu phẩy ngăn cách nghìn
    return to_float(s.str.replace(r"[^\d.\-]", "", regex=True))


def iqr_flag(x, groups, k):
    z = np.log1p(x)
    q1, q3 = (z.groupby(groups).transform(lambda v: v.quantile(p)) for p in (0.25, 0.75))
    return ((z < q1 - k * (q3 - q1)) | (z > q3 + k * (q3 - q1))) & (q3 > q1)


def _r(v):
    return v if isinstance(v, str) else "" if pd.isna(v) else round(float(v), 2)


class Report:
    """Gom các dòng của reports/qa_report.csv (theo từng mốc) và các dòng bị loại."""

    def __init__(self, exp):
        self.rows, self.rejected, self.snap = [], {}, ""
        self.exp = {k: {str(d): n for d, n in v.items()} if isinstance(v, dict) else v for k, v in exp.items()}

    def log(self, rule, table, col, n, metric="rows", before="", after="", status=""):
        cond, why, act = RULES[rule]
        self.rows.append(dict(snapshot=self.snap, rule=rule, table=table, column=col, condition=cond, reason=why,
                              action=act, rows=int(n), metric=metric, before=_r(before), after=_r(after), status=status))

    def blank(self, x, bad, rule, table, col, keep_zero=False):
        """Đổi giá trị vi phạm thành NA; ghi median trước/sau (keep_zero: đếm thêm số 0 được giữ)."""
        if keep_zero:
            self.log("zero_kept", table, col, (x == 0).sum())
        bad = (bad.fillna(False) & x.notna()).astype(bool)
        y = x.mask(bad)
        self.log(rule, table, col, bad.sum(), "median", x.median(), y.median())
        return y

    def flag(self, df, name, mask, rule, table, col, x=None):
        df[name] = mask.fillna(False).astype(bool)
        kw = {} if x is None else dict(metric="median", before=x.median(), after=x[~df[name]].median())
        self.log(rule, table, col, df[name].sum(), **kw)

    def expect(self, table, col, actual, key=None):
        """So với qa.expected (giá trị, khoảng [min, max], hoặc dict theo mốc); không có kỳ vọng thì bỏ qua."""
        e = self.exp.get(key or col)
        e = e.get(self.snap) if isinstance(e, dict) else e
        if e is not None:
            ok = e[0] <= actual <= e[1] if isinstance(e, list) else actual == e
            self.log("expected", table, col, actual, "expected_vs_actual", "-".join(map(str, e)) if isinstance(e, list) else e,
                     actual, "ok" if ok else "error")


def read(path, ids, table, rep, **kw):
    """Đọc theo khối, chỉ giữ dòng thuộc ID của nhóm (so chuỗi thô); đếm khoá rỗng và dòng ngoài nhóm."""
    kept, total, nulls = [], 0, 0
    for c in pd.read_csv(path, dtype=str, na_filter=False, encoding_errors="replace", chunksize=500_000, **kw):
        k = c["id"].str.strip().str.replace(r"\.0+$", "", regex=True)
        total, nulls = total + len(c), nulls + (~k.str.fullmatch(r"\d+")).sum()
        kept.append(c[k.isin(ids)])
    df = pd.concat(kept, ignore_index=True)
    rep.log("key_null", table, "id", nulls)
    rep.log("key_scope", table, "id", total - len(df) - nulls)
    return df


def prep(df, ids, table, rep):
    """Ép id sang số nguyên; loại dòng trùng hệt và xung đột khoá (lưu vào rejected)."""
    n0, df = len(df), df.assign(id=to_int(df["id"]))
    why = pd.Series("", index=df.index)
    for rule in ("dup_exact", "key_conflict"):
        cand = df[why == ""]
        bad = cand.duplicated() if rule == "dup_exact" else cand.duplicated("id", keep=False)
        why[bad[bad].index] = rule
        rep.log(rule, table, "(all)" if rule == "dup_exact" else "id", bad.sum())
    out = df[why == ""]
    rep.rejected.setdefault(table, []).append(df[why != ""].assign(snapshot=rep.snap, reject_reason=why[why != ""]))
    rep.log("ids_missing", table, "id", len(ids - set(out["id"])))
    rep.log("rows_in_out", table, "(all)", n0 - len(out), "rows", n0, len(out))
    return out.assign(id=out["id"].astype("int64"))


def clean_listings(df, ids, areas, boxes, rep):
    t = "listings"
    df = prep(df, ids, t, rep).reindex(columns=LIST_COLS, fill_value="").assign(snapshot=rep.snap)  # cột vắng mặt -> rỗng, bị đếm ở price_empty
    for c in ("neighbourhood_cleansed", "room_type"):  # (2) text, rỗng/sentinel -> NA
        s = df[c].str.strip()
        miss = (s == "") | s.str.lower().isin(SENTINELS)
        rep.log("text_empty", t, c, miss.sum())
        df[c] = s.mask(miss)
    both = df["neighbourhood_cleansed"].str.contains(ENC, na=False) | df["room_type"].str.contains(ENC, na=False)
    rep.flag(df, "flag_encoding", both, "encoding", t, "neighbourhood_cleansed,room_type")
    df["analysis_area"] = df["id"].map(areas)  # (3) khu vực từ bảng tham chiếu
    rep.flag(df, "flag_no_area", df["analysis_area"].isna(), "no_area", t, "analysis_area")
    raw = df["price"].str.strip()  # (4) giá
    empty, sent, p = raw == "", raw.str.lower().isin(SENTINELS), to_money(raw)
    rep.log("price_empty", t, "price", empty.sum())
    rep.log("price_sentinel", t, "price", sent.sum())
    rep.log("price_parse", t, "price", (p.isna() & ~empty & ~sent).sum())
    df["price"] = rep.blank(p, p < 0.01, "price_min", t, "price")
    occ = to_float(df["estimated_occupancy_l365d"])  # (5)
    df["estimated_occupancy_l365d"] = rep.blank(occ, (occ < 0) | (occ > 365), "occ_range", t, "estimated_occupancy_l365d", True)
    rev = to_money(df["estimated_revenue_l365d"])  # (6)
    df["estimated_revenue_l365d"] = rep.blank(rev, rev < 0, "rev_negative", t, "estimated_revenue_l365d", True)
    n = to_int(df["number_of_reviews"])  # (7)
    df["number_of_reviews"] = rep.blank(n, n < 0, "reviews_neg", t, "number_of_reviews")
    lat, lon = to_float(df["latitude"]), to_float(df["longitude"])  # (8)
    df["latitude"] = lat = rep.blank(lat, lat.abs() > 90, "coord_range", t, "latitude")
    df["longitude"] = lon = rep.blank(lon, lon.abs() > 180, "coord_range", t, "longitude")
    b = boxes.reindex(df["analysis_area"]).set_axis(df.index)
    inbox = (lon >= b.lon_min) & (lon <= b.lon_max) & (lat >= b.lat_min) & (lat <= b.lat_max)
    rep.flag(df, "flag_outside_area", ~inbox & lat.notna() & lon.notna() & b.lon_min.notna(), "coord_box", t, "latitude,longitude")
    rt = df["room_type"].value_counts()  # so với qa.expected
    rep.expect(t, "row_count", len(df), "listings")
    rep.expect(t, "entire_home", rt.get("Entire home/apt", 0))
    rep.expect(t, "private_room", rt.get("Private room", 0))
    rep.expect(t, "zero_reviews", df["number_of_reviews"].eq(0).sum())
    rep.expect(t, "missing_price", df["price"].isna().sum())
    for a, k in df["analysis_area"].value_counts().items():
        rep.expect(t, f"area_size:{a}", k, "area_size")
    return df


def clean_reviews(df, ids, sheet_lid, rep):
    t = "reviews"
    df = prep(df, ids, t, rep)[["id", "listing_id", "date", "comments"]].assign(snapshot=rep.snap)
    df["listing_id"] = to_int(df["listing_id"])  # (9)
    differs = (df["listing_id"] != df["id"].map(sheet_lid)) | df["listing_id"].isna()
    rep.flag(df, "flag_listing_mismatch", differs, "id_mismatch", t, "listing_id")
    c = df["comments"].str.strip()  # (10)
    rep.log("comment_empty", t, "comments", (c == "").sum())
    df["comments"] = c.where(c != "")
    rep.flag(df, "flag_encoding", df["comments"].str.contains(ENC, na=False), "encoding", t, "comments")
    rep.expect(t, "row_count", len(df), "reviews")
    return df


def main(cfg):
    root, qa, refs = cfg["data_root"].strip("/ "), cfg["qa"], load_refs(cfg)
    L, R, V = refs["listing"], refs["review"], refs["area"]
    lid, rid, r_lid = to_int(L["listing_id"]), to_int(R["review_id"]), to_int(R["listing_id"])
    lids, rids = set(lid), set(rid)
    boxes = V.set_index("analysis_area")[["lon_min", "lat_min", "lon_max", "lat_max"]].astype(float)
    rep, lis, rev = Report(qa["expected"]), [], []
    for snap in map(str, cfg["snapshots"]):
        rep.snap = snap
        raw = read(build_path(root, snap, "listings.csv.gz"), set(lid.astype(str)), "listings", rep)
        lis.append(clean_listings(raw, lids, dict(zip(lid, L["analysis_area"])), boxes, rep))
        raw = read(build_path(root, snap, "reviews.csv.gz"), set(rid.astype(str)), "reviews", rep,
                   usecols=["listing_id", "id", "date", "comments"])
        rev.append(clean_reviews(raw, rids, dict(zip(rid, r_lid)), rep))
    rep.snap = "all"  # bước cần nhìn mọi mốc cùng lúc
    listings, reviews = pd.concat(lis, ignore_index=True), pd.concat(rev, ignore_index=True).sort_values("snapshot")
    for col, name in (("price", "flag_price_outlier"), ("estimated_revenue_l365d", "flag_revenue_outlier")):
        rep.flag(listings, name, iqr_flag(listings[col].where(listings[col] != 0), listings["room_type"], qa["iqr_k"]), "outlier", "listings", col, x=listings[col])
    n, reviews = len(reviews), reviews.drop_duplicates("id", keep="last")
    rep.log("dup_snapshot", "reviews", "id", n - len(reviews))
    listings.to_csv(PROC / "listings_clean.csv", index=False)
    reviews.to_csv(PROC / "reviews_clean.csv", index=False)
    for table, parts in rep.rejected.items():
        pd.concat(parts).to_csv(PROC / f"rejected_{table}.csv", index=False)
    (ROOT / "reports").mkdir(exist_ok=True)
    report = pd.DataFrame(rep.rows)
    report.to_csv(ROOT / "reports" / "qa_report.csv", index=False, encoding="utf-8-sig")
    print(f"listings: {len(listings)}, reviews: {len(reviews)}; kỳ vọng sai: {(report.status == 'error').sum()} -> reports/qa_report.csv")


if __name__ == "__main__":
    run(main)