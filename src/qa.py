"""QA: lọc theo ID nhóm -> ép kiểu -> loại dòng lỗi -> gắn cờ -> ghi báo cáo.

Chạy: python src/qa.py configs/cape_town.yaml
Ra:   data/processed/<data_root>/<mốc>/<tệp>.parquet (+ rejected_<tệp>.csv), reports/qa_report.csv
Cấu hình cần: city, group_code, data_root, snapshots, ids_file, qa.expected, qa.iqr_k.
Có cache data/processed/<data_root>/ref_*_<group_code>.csv thì không cần tệp Excel nữa."""
import sys

import numpy as np
import pandas as pd
import yaml

from download import ROOT, build_path

SENTINELS = {"n/a", "na", "nan", "null", "none", "-", "--", "unknown"}
TYPES = {  # cột -> kiểu; chỉ các cột này được đọc
    "listings": {"id": "int", "host_id": "int", "host_since": "date", "host_is_superhost": "bool",
                 "host_response_rate": "pct", "host_acceptance_rate": "pct", "neighbourhood_cleansed": "str",
                 "latitude": "float", "longitude": "float", "property_type": "str", "room_type": "str",
                 "accommodates": "int", "bathrooms": "float", "bedrooms": "int", "beds": "int", "price": "price",
                 "minimum_nights": "int", "maximum_nights": "int", "availability_365": "int",
                 "number_of_reviews": "int", "first_review": "date", "last_review": "date",
                 "review_scores_rating": "float", "instant_bookable": "bool", "reviews_per_month": "float"},
    "calendar": {"listing_id": "int", "date": "date", "available": "bool", "price": "price"},
    "reviews": {"id": "int", "listing_id": "int", "date": "date", "reviewer_id": "int", "comments": "str"},
}
KEYS = {"listings": ["id"], "calendar": ["listing_id", "date"], "reviews": ["id"]}  # KEYS[0] cũng là cột lọc ID
REF = {"listings": "listing", "calendar": "listing", "reviews": "review"}  # lọc theo ID của sheet nào
ID_COL = {"listing": "listing_id", "review": "review_id"}
RANGES = {"price": (0.01, np.inf), "latitude": (-90, 90), "longitude": (-180, 180), "host_response_rate": (0, 1),
          "host_acceptance_rate": (0, 1), "review_scores_rating": (0, 5), "accommodates": (1, np.inf),
          "minimum_nights": (1, np.inf), "availability_365": (0, 365), "number_of_reviews": (0, np.inf)}
STRUCTURAL = {"first_review", "last_review", "reviews_per_month", "review_scores_rating"}  # rỗng hợp lệ nếu 0 review
OUTLIER_COLS = {"listings": ["price", "minimum_nights", "accommodates"], "calendar": ["price"]}
OPTIONAL = {"listings": ["price"], "calendar": ["price"]}  # thiếu cột -> chỉ cảnh báo, KPI tự bỏ qua
REPORT = []

# check -> (điều kiện, lý do, hành động); mọi "row_removed_*" dùng chung mục "row_removed"
RULES = {
    "row_removed": ("dòng rỗng / khoá null / trùng", "không dùng được cho phân tích", "loại, ghi vào rejected_*.csv"),
    "file_missing": ("tệp không có hoặc thiếu cột khoá", "tải lỗi hoặc sai định dạng", "bỏ qua tệp, kiểm tra download"),
    "column_absent": ("cột không có trong tệp", "nguồn không cung cấp cột này", "bỏ KPI cần cột đó"),
    "cell_value": ("ô rỗng / sentinel / không parse được / ngoài khoảng", "giá trị không dùng được",
                   "đặt NA, giữ dòng"),
    "real_zero_kept": ("giá trị đúng bằng 0", "0 có thể là giá trị thật", "giữ nguyên, chỉ thống kê"),
    "encoding_suspect": ("chuỗi chứa ký tự lỗi mã hoá", "đọc sai bảng mã", "gắn cờ"),
    "rows_in_out": ("số dòng trước và sau làm sạch", "đo tác động QA", "ghi nhận"),
    "row_count": ("số dòng khác kỳ vọng của đề", "lọc sai hoặc đọc sai ID", "dừng và kiểm tra"),
    "expected_ids_missing": ("ID trong sheet nhưng không có trong tệp", "mốc thiếu ID của nhóm", "gắn cờ"),
    "area_unmatched": ("listing không có analysis_area", "ID không có trong sheet Listing", "gắn cờ"),
    "room_type_count": ("số loại phòng khác kỳ vọng", "lọc sai hoặc danh mục lệch", "dừng và kiểm tra"),
    "zero_review_listings": ("số listing 0 review khác kỳ vọng", "lọc sai hoặc dữ liệu đổi", "dừng và kiểm tra"),
    "missing_price": ("số listing thiếu price khác kỳ vọng", "lọc sai hoặc price rỗng", "loại khỏi KPI giá"),
    "area_size": ("số listing mỗi vùng ngoài khoảng cho phép", "lọc sai hoặc gán vùng sai", "dừng và kiểm tra"),
    "reviews0_xor_last_review": ("0 review nhưng có last_review (hoặc ngược lại)", "mâu thuẫn nội tại", "gắn cờ"),
    "first_after_last_review": ("first_review sau last_review", "mâu thuẫn ngày", "gắn cờ"),
    "coords_outside_area": ("toạ độ ngoài hình chữ nhật analysis_area", "sai toạ độ hoặc sai vùng", "gắn cờ"),
    "listing_not_365_contiguous_days": ("listing không đủ 365 ngày liên tục", "calendar thiếu ngày", "gắn cờ"),
    "listing_id_differs_from_sheet": ("listing_id của review khác sheet Review", "đọc sai ID", "gắn cờ"),
    "orphan_listing_id": ("review trỏ tới listing ngoài danh sách nhóm", "review mồ côi", "gắn cờ"),
    "iqr_outlier_log": ("ngoài q1-k*IQR..q3+k*IQR trên log1p, theo room_type", "có thể nhập sai", "chỉ gắn cờ"),
}


def num(s):
    """Series nullable -> float64 (NA thành NaN), an toàn cho cả Int64 lẫn Float64."""
    return pd.Series(pd.array(s, dtype="Float64").to_numpy(dtype="float64", na_value=np.nan), index=s.index)


def add(ctx, check, category, n, column="", severity="warn", detail="", pct=None):
    REPORT.append({**ctx, "check": check, "category": category, "column": column, "n": int(n),
                   "pct": pct, "severity": severity, "detail": detail})


def flag(ctx, check, category, n, column="", severity="warn", ids=None):
    """Thêm một dòng vào báo cáo khi n > 0 (ids: vài ID ví dụ)."""
    if n:
        pct = round(100 * n / ctx["n_rows"], 3) if ctx["n_rows"] else None
        add(ctx, check, category, n, column, severity,
            "ids: " + ",".join(map(str, list(ids)[:5])) if ids is not None else "", pct)


def expect(ctx, check, actual, expected, column=""):
    """Ghi kết quả so với số lượng kỳ vọng (luôn ghi, ok/error); expected là list = khoảng [min, max]."""
    ok = expected[0] <= actual <= expected[1] if isinstance(expected, list) else actual == expected
    add(ctx, check, "expected_count", actual, column, "ok" if ok else "error", f"expected {expected}")


def load_refs(cfg):
    """Đọc ID/vùng của nhóm từ Excel một lần, cache CSV nhỏ ở data/processed/<data_root>/."""
    base = ROOT / "data/processed" / cfg["data_root"].strip("/ ")
    base.mkdir(parents=True, exist_ok=True)
    refs = {}
    for sheet, key in (("Listing", "listing"), ("Review", "review"), ("Vùng", "area")):
        cache = base / f"ref_{key}_{cfg['group_code']}.csv"  # gắn mã nhóm: đổi nhóm không dùng nhầm cache
        if not cache.exists():
            df = pd.read_excel(ROOT / cfg["ids_file"], sheet_name=sheet, dtype=str)
            if key == "area":
                df = df[df["Thành phố"].str.lower().str.replace(" ", "_") == cfg["city"]]
            else:
                df = df[df["Mã dữ liệu"] == cfg["group_code"]]
            cache.parent.mkdir(parents=True, exist_ok=True)  # tạo thư mục ngay trước khi ghi
            df.to_csv(cache, index=False)
        refs[key] = pd.read_csv(cache, dtype=str)
        for c in set(ID_COL.values()) & set(refs[key]):
            refs[key][c] = pd.to_numeric(refs[key][c], dtype_backend="numpy_nullable")
    return refs


def read_raw(path, name, refs):
    """Đọc theo khối, giữ nguyên chuỗi thô (để phân biệt rỗng / 'N/A' / 0), chỉ giữ ID của nhóm."""
    ref, key = REF[name], KEYS[name][0]
    ids = set(refs[ref][ID_COL[ref]].astype(str))
    chunks = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=lambda c: c in TYPES[name],
                         chunksize=500_000)
    return pd.concat([c[c[key].isin(ids)] for c in chunks], ignore_index=True)


def cast(s, kind):
    """Ép kiểu một cột chuỗi -> (giá trị, lý do từng ô: '' | MISSING | SENTINEL | PARSE_FAIL)."""
    raw = s.str.strip()
    why = pd.Series("", index=s.index, dtype=object)
    why[raw == ""] = "MISSING"
    why[raw.str.lower().isin(SENTINELS)] = "SENTINEL"
    x = raw.where(why == "")
    if kind == "str":
        val = x
    elif kind == "date":
        val = pd.to_datetime(x, format="%Y-%m-%d", errors="coerce")
    elif kind == "bool":
        val = x.str.lower().map({"t": True, "true": True, "f": False, "false": False}).astype("boolean")
    else:
        if kind == "price":  # "$1,234.00" / "R1,234" -> 1234.0
            x = x.str.replace(r"^[^\d.\-]+", "", regex=True).str.replace(",", "")
        elif kind == "pct":
            x = x.str.rstrip("%")
        val = pd.to_numeric(x, errors="coerce", dtype_backend="numpy_nullable")  # Int64 giữ nguyên ID lớn
        if kind == "int":
            val = val.where((val % 1 == 0).fillna(False)).astype("Int64")
        else:
            val = (val / 100 if kind == "pct" else val).astype("Float64")
    why[val.isna() & (why == "")] = "PARSE_FAIL"
    return val, why


def clean(name, raw, snap, ctx):
    """Ép kiểu + kiểm khoảng giá trị + loại dòng; trả về (dữ liệu sạch, dòng bị loại)."""
    df, whys = pd.DataFrame(index=raw.index), {}
    for col, kind in TYPES[name].items():
        if col not in raw:
            add(ctx, "column_absent", "schema", max(len(raw), 1), col,
                "warn" if col in OPTIONAL.get(name, []) else "error", pct=100)
            continue
        df[col], whys[col] = cast(raw[col], kind)
        if kind in ("int", "float", "price"):  # đếm số 0 thật TRƯỚC khi kiểm khoảng (price=0 sẽ thành NA)
            flag(ctx, "real_zero_kept", "zero_value", (df[col] == 0).sum(), col, "info")
        if kind == "date":  # ngày không được sau mốc chụp (trừ ngày của calendar, vốn nằm sau mốc)
            end = pd.Timestamp.max if (name, col) == ("calendar", "date") else pd.Timestamp(snap)
            bounds = (pd.Timestamp.min, end)
        else:
            bounds = RANGES.get(col)
        if bounds:
            bad = (df[col].notna() & ~df[col].between(*bounds)).astype(bool)
            df.loc[bad, col] = pd.NA
            whys[col][bad] = "OUT_OF_RANGE"
        if kind == "str":
            flag(ctx, "encoding_suspect", "encoding", df[col].str.contains("Ã|â€|\ufffd", na=False).sum(), col)
    if name == "listings" and "number_of_reviews" in df:  # rỗng do 0 review là hợp lệ, không phải lỗi
        zero = df["number_of_reviews"].eq(0).fillna(False)
        for c in STRUCTURAL & whys.keys():
            whys[c] = whys[c].mask(zero & (whys[c] == "MISSING"), "STRUCTURAL_NULL")
    for col, why in whys.items():
        for r, n in why[why != ""].value_counts().items():
            flag(ctx, "cell_value", r.lower(), n, col, "info" if r == "STRUCTURAL_NULL" else "warn")
    keys = [k for k in KEYS[name] if k in df]
    reason = pd.Series("", index=df.index, dtype=object)
    reason[df.drop(columns=keys).isna().all(axis=1)] = "HOLLOW"
    reason[df[keys].isna().any(axis=1)] = "NULL_KEY"
    for why, subset in (("EXACT_DUP", None), ("KEY_CONFLICT", keys)):  # trùng hệt, rồi trùng khoá khác nội dung
        dup = df[reason == ""].duplicated(subset)
        reason[dup[dup].index] = why
    for r, n in reason[reason != ""].value_counts().items():
        flag(ctx, f"row_removed_{r.lower()}", "removed", n, severity="error")
    gone = reason != ""
    return df[~gone].reset_index(drop=True), df[gone].assign(reject_reason=reason[gone])


def checks(name, df, ctx, refs, exp):
    """Đối chiếu số lượng kỳ vọng, khớp ID/merge, tính nhất quán. Trả về df (listings có thêm analysis_area)."""
    key = KEYS[name][0]

    def mark(check, cat, bad, col=""):
        bad = pd.Series(bad, index=df.index).fillna(False).astype(bool)
        flag(ctx, check, cat, bad.sum(), col, ids=df[key][bad])

    expect(ctx, "row_count", len(df), exp[name])
    ref = REF[name]
    flag(ctx, "expected_ids_missing", "merge_alignment", len(set(refs[ref][ID_COL[ref]]) - set(df[key])))
    snap = ctx["snapshot"]
    if name == "listings":
        df = df.merge(refs["listing"][["listing_id", "analysis_area"]], left_on="id", right_on="listing_id",
                      how="left", validate="m:1").drop(columns="listing_id")
        mark("area_unmatched", "merge_alignment", df["analysis_area"].isna())
        rt = df["room_type"].value_counts()
        for room, k in (("Entire home/apt", "entire_home"), ("Private room", "private_room")):
            expect(ctx, "room_type_count", rt.get(room, 0), exp[k], room)
        if snap in exp.get("zero_reviews", {}):  # mốc lạ (không có trong cấu hình) thì bỏ qua
            expect(ctx, "zero_review_listings", df["number_of_reviews"].eq(0).sum(), exp["zero_reviews"][snap])
        if snap in exp.get("missing_price", {}) and "price" in df:
            expect(ctx, "missing_price", df["price"].isna().sum(), exp["missing_price"][snap], "price")
        for area, n in df["analysis_area"].value_counts().items():
            expect(ctx, "area_size", n, exp["area_size"], area)
        mark("reviews0_xor_last_review", "consistency", df["number_of_reviews"].eq(0) ^ df["last_review"].isna())
        mark("first_after_last_review", "consistency", df["first_review"] > df["last_review"])
        box = refs["area"].set_index("analysis_area")[["lon_min", "lat_min", "lon_max", "lat_max"]].astype(float)
        box = box.reindex(df["analysis_area"]).set_axis(df.index)
        lon, lat = num(df["longitude"]), num(df["latitude"])
        mark("coords_outside_area", "consistency",
             (lon < box.lon_min) | (lon > box.lon_max) | (lat < box.lat_min) | (lat > box.lat_max))
    elif name == "calendar":
        g = df.groupby("listing_id")["date"]
        bad = (g.nunique() != 365) | ((g.max() - g.min()).dt.days != 364)
        flag(ctx, "listing_not_365_contiguous_days", "consistency", bad.sum(), ids=bad[bad].index)
    else:
        m = df[["id", "listing_id"]].merge(refs["review"][["review_id", "listing_id"]], left_on="id",
                                           right_on="review_id", how="left", suffixes=("", "_sheet"))
        mark("listing_id_differs_from_sheet", "merge_alignment", m["listing_id"] != m["listing_id_sheet"])
        mark("orphan_listing_id", "merge_alignment", ~df["listing_id"].isin(refs["listing"]["listing_id"]))
    return df


def flag_outliers(name, df, ctx, k):
    """IQR trên log1p, theo room_type nếu có; chỉ gắn cờ, không xoá; bỏ qua khi IQR = 0."""
    grp = df["room_type"] if "room_type" in df else pd.Series(0, index=df.index)
    for col in [c for c in OUTLIER_COLS.get(name, []) if c in df]:
        x = np.log1p(num(df[col]))
        q1, q3 = (x.groupby(grp).transform("quantile", p) for p in (0.25, 0.75))
        out = ((x < q1 - k * (q3 - q1)) | (x > q3 + k * (q3 - q1))) & (q3 > q1)
        flag(ctx, "iqr_outlier_log", "outlier", out.sum(), col, "info", ids=df[KEYS[name][0]][out])


def main(config):
    REPORT.clear()  # để run_pipeline.py gọi nhiều lần không bị lẫn kết quả
    cfg = yaml.safe_load(open(config, encoding="utf-8"))
    root, qa = cfg["data_root"].strip("/ "), cfg["qa"]
    exp = dict(qa["expected"])
    for k in ("zero_reviews", "missing_price"):  # YAML có thể biến khoá ngày thành kiểu date
        exp[k] = {str(d): n for d, n in exp.get(k, {}).items()}
    refs = load_refs(cfg)
    for snap in map(str, cfg["snapshots"]):
        for name in TYPES:
            ctx = {"snapshot": snap, "file": name, "n_rows": 0}
            try:
                raw = read_raw(build_path(root, snap, f"{name}.csv.gz"), name, refs)
            except (OSError, ValueError, EOFError, KeyError) as e:  # tệp thiếu / hỏng / thiếu cột khoá
                add(ctx, "file_missing", "download", 1, severity="error", detail=f"{type(e).__name__}: {e}")
                continue
            ctx["n_rows"] = len(raw)
            df, rejected = clean(name, raw, snap, ctx)
            add(ctx, "rows_in_out", "impact", len(df), severity="info", detail=f"in={len(raw)}, out={len(df)}")
            df = checks(name, df, ctx, refs, exp)
            flag_outliers(name, df, ctx, qa["iqr_k"])
            out = ROOT / "data/processed" / root / snap
            out.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out / f"{name}.parquet", index=False)
            if len(rejected):
                rejected.to_csv(out / f"rejected_{name}.csv", index=False)
    rep = pd.DataFrame(REPORT)
    blank = ("", "", "")
    rep[["condition", "reason", "action"]] = rep["check"].map(
        lambda c: RULES.get("row_removed" if c.startswith("row_removed_") else c, blank)).tolist()
    (ROOT / "reports").mkdir(exist_ok=True)
    rep.to_csv(ROOT / "reports/qa_report.csv", index=False)
    print(f"Done: {len(rep)} findings, {int((rep['severity'] == 'error').sum())} error -> reports/qa_report.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/cape_town.yaml"))