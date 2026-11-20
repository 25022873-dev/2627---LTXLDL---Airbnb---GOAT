"""Vẽ 5 biểu đồ KPI bằng tiếng Việt: python src/viz.py configs/cape_town.yaml
Vào: data/processed/{kpi_*,listings_clean,llm_results,reviews_clean}.csv. Ra: figures/*.png (mỗi hình có chú thích và nguồn dữ liệu)."""
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter

from common import PROC, ROOT, run
from kpi import scored_reviews

S, A, OCC = "snapshot", "analysis_area", "estimated_occupancy_l365d"
vn = lambda x, d=1: f"{x:.{d}f}".replace(".", ",")  # dấu phẩy thập phân
dmy = lambda s: pd.Timestamp(s).strftime("%d/%m/%Y")
thousands = FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", "."))
pivot = lambda k: k.pivot(index="group", columns=S, values="value")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def heatmap(k):
    p = pivot(k) * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(p, cmap="Blues", aspect="auto", vmin=0)  # nhạt = thấp, đậm = cao
    ax.set_xticks(range(p.shape[1]), [dmy(c) for c in p.columns])
    ax.set_yticks(range(len(p)), p.index)
    for (i, j), v in np.ndenumerate(p.to_numpy()):
        if pd.notna(v):
            ax.text(j, i, vn(v) + "%", ha="center", va="center", color="white" if v > .6 * np.nanmax(p) else "black")
    fig.colorbar(im, ax=ax, label="Tỷ lệ listing ngủ đông (%)")
    ax.set(xlabel="Mốc thu thập dữ liệu", ylabel="Khu vực", title="Tỷ lệ listing ngủ đông theo khu vực và mốc thời gian")
    return fig, ("Khu vực nào có nhiều listing không được đặt đêm nào trong 365 ngày, và tỷ lệ này thay đổi ra sao giữa các mốc? "
                 "Ô càng đậm thì tỷ lệ listing ngủ đông càng cao.")


def lines(k, title, ylabel, caption):
    p = pivot(k).T
    p.index = pd.to_datetime(p.index)
    fig, ax = plt.subplots(figsize=(9, 5))
    for c in p:
        ax.plot(p.index, p[c], marker="o", label=c)
    ax.set_xticks(p.index, [d.strftime("%d/%m/%Y") for d in p.index])
    ax.yaxis.set_major_formatter(thousands)
    ax.legend(title="Khu vực", loc="center left", bbox_to_anchor=(1.01, .5), frameon=False)
    ax.set(xlabel="Mốc thu thập dữ liệu", ylabel=ylabel, title=title)
    return fig, caption


def diverging(price, nights, rev, df, room):
    order = rev.groupby("group")["value"].mean().sort_values(ascending=False).index  # doanh thu trung bình cao nhất nằm trên cùng
    dates, y = sorted(rev[S].unique()), np.arange(len(order))
    over = lambda d, col: d.groupby(S)[col].median()  # trung vị toàn mẫu tại từng mốc
    P = (pivot(price) / over(df[df["room_type"] == room], "price") - 1) * 100
    N = (pivot(nights) / over(df[df[OCC] > 0], OCC) - 1) * 100
    fig, axes = plt.subplots(1, len(dates), figsize=(4.2 * len(dates), 5), sharey=True, sharex=True, squeeze=False)
    for ax, d in zip(axes[0], dates):
        ax.axvspan(-10, 10, color="grey", alpha=.25)
        ax.barh(y - .2, P[d].reindex(order).to_numpy(), .4, color="tab:orange")
        ax.barh(y + .2, N[d].reindex(order).to_numpy(), .4, color="tab:blue")
        ax.axvline(0, color="black", lw=.8)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.set(title=dmy(d), xlabel="Chênh lệch so với trung vị toàn mẫu (%)")
    axes[0][0].set_yticks(y, order)
    axes[0][0].invert_yaxis()
    axes[0][0].set_ylabel("Khu vực")
    fig.legend(handles=[Patch(color="tab:orange", label="Giá trung vị (nhà/căn hộ nguyên căn)"),
                        Patch(color="tab:blue", label="Số đêm đặt trung vị (listing có đêm đặt)"),
                        Patch(color="grey", alpha=.25, label="Vùng ±10% quanh trung vị toàn mẫu (xấp xỉ mức chung)")],
               loc="lower center", ncol=3, bbox_to_anchor=(.5, -.06), frameon=False)
    fig.suptitle("Chênh lệch giá và số đêm đặt trung vị của từng khu vực so với toàn mẫu\n"
                 "(khu vực xếp theo doanh thu trung vị trung bình, cao nhất ở trên cùng)", y=1.09)
    return fig, ("Khu vực nào đắt hơn hoặc được đặt nhiều đêm hơn mức chung của toàn mẫu? "
                 "Thanh nằm trong vùng xám nghĩa là chênh lệch dưới 10%, xấp xỉ mức chung.")


def sentiment(z, min_n):
    tot = z["score"].mean()
    z = z[z.groupby(A)["score"].transform("size") >= min_n]  # cùng ngưỡng với kpi_net_sentiment.csv
    g = z.groupby(A)["score"].agg(["mean", "std", "size"]).sort_values("mean")  # cao nhất nằm trên cùng
    g["ci"] = 1.96 * g["std"] / np.sqrt(g["size"])
    pct = pd.crosstab(z[A], z["sentiment"], normalize="index").reindex(g.index)[["negative", "mixed", "positive"]] * 100
    y = np.arange(len(g))
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    a.hlines(y, g["mean"] - g["ci"], g["mean"] + g["ci"], color="tab:blue", alpha=.35, lw=7, label="Khoảng tin cậy 95%")
    a.plot(g["mean"], y, "o", color="tab:blue", label="Điểm cảm xúc ròng ước tính")
    a.axvline(tot, ls="--", color="black", label=f"Toàn mẫu ({vn(tot, 2)})")
    for yi, (m, c, n) in enumerate(zip(g["mean"], g["ci"], g["size"])):
        a.annotate(f"{vn(m, 2)} (n={n})", (m + c, yi), xytext=(5, 0), textcoords="offset points", va="center",
                   bbox=dict(fc="white", ec="none", pad=1.5))
    a.set_xlim(right=(g["mean"] + g["ci"]).max() + .35)
    a.set_yticks(y, g.index)
    a.legend(loc="upper center", bbox_to_anchor=(.5, -.14), ncol=3, frameon=False)
    a.set(xlabel="Điểm cảm xúc ròng (từ −1 đến +1)", ylabel="Khu vực", title="Điểm cảm xúc ròng và khoảng tin cậy 95%")
    left = np.zeros(len(g))
    for col, name, color in zip(pct.columns, ["Tiêu cực", "Hỗn hợp", "Tích cực"], ["tab:red", "lightgrey", "tab:blue"]):
        b.barh(y, pct[col], left=left, color=color, label=name)
        left += pct[col].to_numpy()
    b.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    b.legend(title="Cảm xúc", loc="center left", bbox_to_anchor=(1.01, .5), frameon=False)
    b.set(xlim=(0, 100), xlabel="Tỷ lệ review (%)", title="Cơ cấu cảm xúc của review (mỗi thanh = 100%)")
    return fig, ("Khu vực nào khách hài lòng nhất? Điểm cảm xúc ròng = tỷ lệ tích cực trừ tỷ lệ tiêu cực (hỗn hợp tính 0), "
                 "trên mẫu review do LLM gán nhãn (không tính review tự động); khoảng tin cậy tính theo xấp xỉ chuẩn.")


def main(cfg):
    cur, room = cfg["currency"], cfg["kpi"]["room_type"]
    K = lambda n: (lambda k: k[k["level"] == A])(pd.read_csv(PROC / f"kpi_{n}.csv"))  # chỉ cấp khu vực
    df, price, rev = pd.read_csv(PROC / "listings_clean.csv"), K("median_price"), K("median_revenue")
    price = price[price["room_type"] == room]
    figs = {
        "1_ty_le_ngu_dong": heatmap(K("hibernation_rate")),
        "2_gia_trung_vi": lines(price, "Giá trung vị theo khu vực (nhà/căn hộ nguyên căn)", f"Giá trung vị ({cur}/đêm)",
                                "Giá trung vị của từng khu vực thay đổi ra sao giữa các mốc? Chỉ tính nhà/căn hộ nguyên căn; mỗi chấm là một mốc thu thập."),
        "3_doanh_thu_trung_vi": lines(rev, "Doanh thu trung vị mỗi listing theo khu vực", f"Doanh thu ước tính 365 ngày ({cur})",
                                      "Khu vực nào có doanh thu ước tính mỗi listing cao nhất? Doanh thu là ước tính 365 ngày của Inside Airbnb, tính cả listing doanh thu 0."),
        "4_so_dem_dat_trung_vi": diverging(price, K("median_occupied_nights"), rev, df, room),
        "5_cam_xuc_rong": sentiment(scored_reviews(df), cfg["kpi"]["min_n"]),
    }
    snaps = cfg["snapshots"]
    src = f"Nguồn: Inside Airbnb (CC BY 4.0), {cfg['city'].replace('_', ' ').title()}, các mốc từ {dmy(snaps[0])} đến {dmy(snaps[-1])}."
    (ROOT / "figures").mkdir(exist_ok=True)
    for name, (f, caption) in figs.items():
        bb = f.get_tightbbox()  # đặt chú thích ngay dưới phần thấp nhất của hình (kể cả chú giải nằm ngoài)
        f.text(bb.x0 / f.get_figwidth(), bb.y0 / f.get_figheight() - .02,
               textwrap.fill(caption, int(f.get_figwidth() * 18)) + "\n" + src, fontsize=7, va="top", color="dimgrey")
        f.savefig(ROOT / "figures" / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(f)
        print(f"[ok]   {name}")


if __name__ == "__main__":
    run(main)