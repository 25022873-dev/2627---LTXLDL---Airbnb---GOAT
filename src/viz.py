"""Sinh hình vào figures/ từ kpi_*.csv, eval_*.csv và listings.parquet (chỉ đọc, không tính lại KPI).

Chạy: python src/viz.py configs/cape_town.yaml
Chỉ dùng matplotlib (backend Agg). Bản đồ: điểm listing + hình chữ nhật analysis_area (không có geojson)."""
import calendar
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Rectangle

from download import ROOT
from qa import load_refs

P, BLUE, CLASSES = ROOT / "data/processed", "#4C78A8", ["positive", "mixed", "negative"]
NAMES = {"floor": "Majority class", "baseline": "VADER baseline", "llm": "LLM"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def kpi(name, k):
    d = pd.read_csv(P / f"kpi_{name}.csv", dtype={"snapshot": str})
    return d[(d["kpi"] == k) & (d["level"] == "analysis_area")].copy()


def barh(ax, d, title, xlabel):
    d = d.sort_values("value")  # cao nhất ở trên; cột bắt đầu từ 0
    ax.barh(d["group"], d["value"], color=BLUE)
    for y, (v, n) in enumerate(zip(d["value"], d["n"])):
        ax.text(v, y, f" n={n}", va="center", fontsize=7)
    ax.margins(x=0.15)
    ax.set_title(title), ax.set_xlabel(xlabel)


def fig1_price_map(cfg):
    last = str(cfg["snapshots"][-1])
    L = pd.read_parquet(P / cfg["data_root"].strip("/ ") / last / "listings.parquet",
                        columns=["latitude", "longitude", "price"]).dropna()
    A = load_refs(cfg)["area"].set_index("analysis_area")[["lon_min", "lat_min", "lon_max", "lat_max"]].astype(float)
    f, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(L["longitude"], L["latitude"], c=L["price"], s=8, alpha=0.8, cmap="viridis",
                    vmax=L["price"].quantile(0.95))  # cắt thang màu ở phân vị 95
    for a, r in A.iterrows():
        ax.add_patch(Rectangle((r.lon_min, r.lat_min), r.lon_max - r.lon_min, r.lat_max - r.lat_min, fill=False, ec="grey"))
        ax.text(r.lon_min, r.lat_max, f" {a}", va="top", fontsize=8, color="dimgrey")
    ax.autoscale()
    ax.set_aspect(1 / np.cos(np.radians(L["latitude"].mean())))
    ax.set_xlabel("Longitude"), ax.set_ylabel("Latitude")
    ax.set_title(f"Listing prices, snapshot {last} (n={len(L)})")
    f.colorbar(sc, ax=ax, label=f"Price ({cfg['currency']}/night)", shrink=0.8)
    return f, "Where are the expensive listings? Colour capped at the 95th percentile. Rectangles are analysis areas, not administrative boundaries."


def fig2_area_bars(cfg):
    p = kpi("area", "median_price")
    p[["group", "room"]] = p["group"].str.split(" | ", regex=False, expand=True)
    room = p.groupby("room")["n"].sum().idxmax()  # loại phòng nhiều listing nhất
    f, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    barh(axs[0], p[p["room"] == room], f"Median price, {room}", f"{cfg['currency']} per night")
    barh(axs[1], kpi("area", "est_revenue_per_listing"), "Median est. revenue per listing", f"{cfg['currency']} per year")
    return f, "Which areas are priciest and earn most? Revenue = price x 90-day occupancy x 365."


def fig3_occupancy(cfg):
    piv = kpi("time", "occupancy_90d").pivot(index="snapshot", columns="group", values="value").sort_index() * 100
    f, ax = plt.subplots(figsize=(7, 4.5))
    for c, col in zip(piv, plt.cm.tab10.colors):
        ax.plot(piv.index, piv[c], marker="o", color=col)
        ax.annotate(c, (piv.index[-1], piv[c].iloc[-1]), xytext=(5, 0), textcoords="offset points",
                    fontsize=8, va="center", color=col)
    ax.margins(x=0.12)
    ax.set_xlabel("Snapshot"), ax.set_ylabel("Estimated occupancy, next 90 days (%)")
    ax.set_title("Estimated occupancy by area")
    return f, "How does estimated occupancy change across snapshots? Blocked days count as booked."


def fig4_seasonality(cfg):
    d = kpi("time", "reviews_per_month")
    tot = d.assign(m=d["group"].str[-7:]).groupby("m")["value"].sum().sort_index().iloc[:-1].iloc[-36:]  # bỏ tháng cuối (cụt)
    by = tot.groupby(tot.index.str[5:].astype(int)).mean()
    f, axs = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    axs[0].bar([calendar.month_abbr[m] for m in by.index], by.values, color=BLUE)
    axs[0].set_ylabel("Reviews per month (all areas)"), axs[0].set_title("Reviews by calendar month")
    barh(axs[1], kpi("time", "seasonality_ratio"), "Seasonality ratio by area", "Max / min monthly occupancy")
    return f, "When is the peak season? Left: mean reviews per month, last 3 full years (proxy for stays)."


def fig5_net_sentiment(cfg):
    f, ax = plt.subplots(figsize=(6, 4.5))
    barh(ax, kpi("llm", "net_sentiment"), "Net sentiment by area (LLM sample)", "Net sentiment (-1 to 1)")
    ax.axvline(0, color="black", lw=0.8)
    return f, "Where are guests happiest? Share positive minus share negative; mixed counts as 0."


def fig6_llm_vs_baseline(cfg):
    raw = pd.read_csv(P / "eval_metrics.csv")
    raw = raw[raw["group"] == "all"]
    E, order = raw.pivot(index="method", columns="metric", values="value"), [m for m in NAMES if m in set(raw["method"])]
    f, ax = plt.subplots(figsize=(6, 4))
    x, w = np.arange(len(order)), 0.38
    for j, k in enumerate(("accuracy", "macro_f1")):
        v = E.loc[order, k]
        ax.bar(x + (j - 0.5) * w, v, w, label=k.replace("_", "-"), capsize=3,
               yerr=[v - E.loc[order, f"{k}_ci_lo"], E.loc[order, f"{k}_ci_hi"] - v])
    ax.set_xticks(x, [NAMES[m] for m in order])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.set_title(f"Sentiment on {int(raw['n'].max())} hand labels (95% bootstrap CI)")
    return f, "Is the LLM better than a simple baseline? Compared with a majority-class floor."


def fig7_confusion(cfg):
    C = pd.read_csv(P / "eval_confusion.csv")
    ms = [m for m in ("llm", "baseline") if m in set(C["method"])]
    f, axs = plt.subplots(1, len(ms), figsize=(4.5 * len(ms), 4), squeeze=False)
    for ax, m in zip(axs[0], ms):
        t = C[C["method"] == m].pivot_table(index="label", columns="pred", values="count", fill_value=0)
        t = t.reindex(index=CLASSES, columns=CLASSES + ["none"], fill_value=0)
        ax.imshow(t, cmap="Blues")
        for (i, j), v in np.ndenumerate(t.values):
            ax.text(j, i, int(v), ha="center", va="center", color="white" if v > t.values.max() / 2 else "black")
        ax.set_xticks(range(4), t.columns), ax.set_yticks(range(3), t.index)
        ax.set_xlabel("Prediction"), ax.set_ylabel("Hand label"), ax.set_title(NAMES[m])
    return f, "Where does each method go wrong? Rows: hand label; none = no valid output."


def main(config):
    cfg = yaml.safe_load(open(config, encoding="utf-8"))
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    snaps = cfg["snapshots"]
    src = f"Source: Inside Airbnb (CC BY 4.0), {cfg['city'].replace('_', ' ').title()}, snapshots {snaps[0]} to {snaps[-1]}."
    for fn in (fig1_price_map, fig2_area_bars, fig3_occupancy, fig4_seasonality, fig5_net_sentiment,
               fig6_llm_vs_baseline, fig7_confusion):
        f, caption = fn(cfg)
        f.text(0.01, 0.01, f"{caption}\n{src}", fontsize=6.5, va="bottom", color="dimgrey")
        f.tight_layout(rect=(0, 0.07, 1, 1))
        f.savefig(out / f"{fn.__name__}.png", dpi=200)
        plt.close(f)
        print(f"[ok]   {fn.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/cape_town.yaml"))