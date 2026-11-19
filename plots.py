"""
plots.py - all charts for the app (matplotlib). Each function returns a figure.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

NAVY, FAST, SLOW, GRAY = "#1F3A5F", "#2A7F9E", "#C8553D", "#9CA3AF"
FAST_LIGHT, SLOW_LIGHT = "#BBDCE7", "#F0C4BA"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})


def _clean(ax):
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#BBBBBB")


def class_bar(counts):
    """Overview: how many Fast and Slow sales."""
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(["Fast Sale\n(<= 30 days)", "Slow Sale\n(> 30 days)"], [counts["Fast"], counts["Slow"]],
                  color=[FAST, SLOW], width=0.55)
    for b, k in zip(bars, ["Fast", "Slow"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                f"{counts[k]} cars\n({counts[k] / counts['Total'] * 100:.1f}%)", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(counts["Fast"], counts["Slow"]) * 1.3)
    ax.set_ylabel("Number of sold cars"); _clean(ax)
    ax.set_title(f"The {counts['Total']} sold cars", fontsize=10, fontweight="bold", color=NAVY)
    fig.tight_layout(); return fig


def waffle(sold, feature, value, label):
    """One square per sold car. Dark squares = cars that have the chosen feature value."""
    cars = sold.sort_values(["Class"]).copy()
    groups = [("Fast", True, FAST, f"Fast and {value}"), ("Fast", False, FAST_LIGHT, f"Fast, not {value}"),
              ("Slow", True, SLOW, f"Slow and {value}"), ("Slow", False, SLOW_LIGHT, f"Slow, not {value}")]
    squares, handles = [], []
    for cls, has, color, name in groups:
        n = int(((cars["Class"] == cls) & ((cars[feature] == value) == has)).sum())
        squares += [color] * n
        handles.append(Patch(color=color, label=f"{name}: {n}"))
    total = len(squares)
    cols = 9
    rows = -(-total // cols)
    fig, ax = plt.subplots(figsize=(6.2, 0.55 * rows + 1.4))
    for i, color in enumerate(squares):
        r, c = divmod(i, cols)
        ax.add_patch(Rectangle((c, rows - 1 - r), 0.9, 0.9, color=color))
    ax.set_xlim(0, cols); ax.set_ylim(0, rows); ax.axis("off")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=8.5)
    ax.set_title(f"Each square is one sold car ({total} cars). {label}", fontsize=9.5, color=NAVY, fontweight="bold")
    fig.tight_layout(); return fig


def prior_vs_posterior(prior, posterior, cls, value):
    """Shows how the chance of Fast (or Slow) changes after we see the feature."""
    color = FAST if cls == "Fast" else SLOW
    fig, ax = plt.subplots(figsize=(5.2, 3))
    bars = ax.bar([f"Prior\nP({cls})", f"Posterior\nP({cls} | {value})"], [prior * 100, (posterior or 0) * 100],
                  color=[GRAY, color], width=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.1f}%", ha="center",
                fontweight="bold", fontsize=10)
    ax.set_ylim(0, 105); ax.set_ylabel("Chance (%)"); _clean(ax)
    ax.set_title("Prior knowledge + new evidence = posterior", fontsize=10, fontweight="bold", color=NAVY)
    fig.tight_layout(); return fig


def evidence_two_ways(count_way, sum_way, parts, value):
    """Stacked bar: P(value) built from the Fast part and the Slow part."""
    fig, ax = plt.subplots(figsize=(6, 1.7))
    fast_part, slow_part = parts
    ax.barh(0, fast_part * 100, color=FAST, height=0.5)
    ax.barh(0, slow_part * 100, left=fast_part * 100, color=SLOW, height=0.5)
    ax.text(fast_part * 50, 0, f"Fast part\n{fast_part:.3f}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.text(fast_part * 100 + slow_part * 50, 0, f"Slow part\n{slow_part:.3f}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.set_xlim(0, max(60, (fast_part + slow_part) * 100 + 10)); ax.set_yticks([]); ax.set_xlabel("Percent of all sold cars")
    ax.set_title(f"Evidence P({value}) = {sum_way:.3f} (same as counting: {count_way:.3f})", fontsize=9, fontweight="bold", color=NAVY)
    for s in ("left",): ax.spines[s].set_visible(False)
    _clean(ax); fig.tight_layout(); return fig


def factor_chart(res):
    """The four numbers that are multiplied together, for Fast and Slow."""
    labels = ["Prior"] + [res["values"][k] for k in ("Type", "Mileage_Band", "Condition")]
    f = [x["value"] for x in res["Fast"]["factors"]]
    s = [x["value"] for x in res["Slow"]["factors"]]
    import numpy as np
    x = np.arange(4); w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    b1 = ax.bar(x - w / 2, f, w, color=FAST, label="Fast Sale")
    b2 = ax.bar(x + w / 2, s, w, color=SLOW, label="Slow Sale")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Value"); ax.legend(frameon=False, fontsize=8.5); _clean(ax)
    ax.set_title("The four numbers we multiply", fontsize=10, fontweight="bold", color=NAVY)
    fig.tight_layout(); return fig


def running_product(res):
    """Shows the product getting smaller as each factor is multiplied in."""
    steps = ["Prior", "x Type", "x Mileage", "x Condition"]
    fig, ax = plt.subplots(figsize=(6.4, 3))
    ax.plot(steps, res["Fast"]["running"], marker="o", color=FAST, lw=2, label="Fast Sale")
    ax.plot(steps, res["Slow"]["running"], marker="o", color=SLOW, lw=2, label="Slow Sale")
    ax.annotate(f"{res['Fast']['score']:.4f}", (3, res["Fast"]["running"][-1]), textcoords="offset points", xytext=(8, 6), color=FAST, fontweight="bold", fontsize=9)
    ax.annotate(f"{res['Slow']['score']:.4f}", (3, res["Slow"]["running"][-1]), textcoords="offset points", xytext=(8, -14), color=SLOW, fontweight="bold", fontsize=9)
    ax.set_ylabel("Running product"); ax.legend(frameon=False, fontsize=8.5); _clean(ax)
    ax.set_xlim(-0.2, 3.6)
    ax.set_title("Running product after each multiplication", fontsize=10, fontweight="bold", color=NAVY)
    fig.tight_layout(); return fig


def percent_bar(res):
    """Fast % vs Slow % for one car."""
    fp, sp = res["Fast"]["percent"], res["Slow"]["percent"]
    fig, ax = plt.subplots(figsize=(6.4, 1.5))
    ax.barh(0, fp, color=FAST, height=0.55); ax.barh(0, sp, left=fp, color=SLOW, height=0.55)
    ax.text(fp / 2, 0, f"Fast {fp:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax.text(fp + sp / 2, 0, f"Slow {sp:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax.set_xlim(0, 100); ax.axis("off"); fig.tight_layout(); return fig


def joint_vs_product(actual, product, text_actual, text_product):
    """Independence check: real joint probability vs the 'naive' product."""
    fig, ax = plt.subplots(figsize=(5.4, 3))
    bars = ax.bar(["Real data\nP(A and B | C)", "Naive assumption\nP(A | C) x P(B | C)"], [actual, product],
                  color=[NAVY, GRAY], width=0.5)
    for b, t in zip(bars, [text_actual, text_product]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, t, ha="center", fontweight="bold", fontsize=9)
    ax.set_ylim(0, max(actual, product, 0.05) * 1.3); ax.set_ylabel("Probability"); _clean(ax)
    ax.set_title("Are the two features independent?", fontsize=10, fontweight="bold", color=NAVY)
    fig.tight_layout(); return fig


def results_split(hist, pred):
    """Past sales vs predictions for the cars on the lot (Figure 2 in the report).
    The cars that cannot be classified are not drawn; a note under the chart explains them."""
    rows = [("Current inventory\n(predicted)", [("Fast", pred["Fast"], FAST), ("Slow", pred["Slow"], SLOW)], sum(pred.values())),
            ("Historical sales\n(actual)", [("Fast", hist["Fast"], FAST), ("Slow", hist["Slow"], SLOW)], hist["Fast"] + hist["Slow"])]
    fig, ax = plt.subplots(figsize=(7, 2.3))
    for i, (_, segs, tot) in enumerate(rows):
        left = 0
        for name, n, col in segs:
            if n == 0:
                continue
            wd = n / tot * 100  # share of ALL cars in the row, so it matches the table
            ax.barh(i, wd, left=left, color=col, height=0.55)
            ax.text(left + wd / 2, i, f"{name}: {n} ({wd:.1f}%)", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
            left += wd
    ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0, 100); ax.set_xticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    n_na = pred.get("Cannot classify", 0)
    if n_na:
        ax.text(100, -0.62, f"Not shown: {n_na} cars cannot be classified ({n_na / sum(pred.values()) * 100:.1f}%)",
                ha="right", va="top", fontsize=8, color=GRAY)
    fig.tight_layout(); return fig


def rates_chart(rates, baseline):
    """Fast-sale rate for every feature value (Figure 3 in the report)."""
    feats = list(dict.fromkeys(rates["Feature"]))
    widths = [len(rates[rates["Feature"] == f]) for f in feats]
    fig, axs = plt.subplots(1, len(feats), figsize=(8, 3.2), sharey=True, gridspec_kw={"width_ratios": widths})
    for ax, f in zip(axs, feats):
        sub = rates[rates["Feature"] == f]
        bars = ax.bar(sub["Value"], sub["Fast-sale rate (%)"], color=FAST, width=0.62)
        ax.axhline(baseline, color=SLOW, ls="--", lw=1.2)
        for b, (_, r) in zip(bars, sub.iterrows()):
            ax.text(b.get_x() + b.get_width() / 2, r["Fast-sale rate (%)"] + 3, f"{r['Fast-sale rate (%)']:.0f}%", ha="center", fontsize=8, fontweight="bold",
                    bbox=dict(fc="white", ec="none", pad=1))
            ax.text(b.get_x() + b.get_width() / 2, -14, f"n={r['Sold cars']}", ha="center", fontsize=7, color="#6B7280")
        ax.set_title(f, fontsize=9.5, fontweight="bold", color=NAVY); ax.set_ylim(0, 100)
        ax.tick_params(axis="x", labelsize=8, length=0); _clean(ax)
    axs[0].set_ylabel("Sold within 30 days (%)")
    fig.legend(handles=[plt.Line2D([0], [0], color=SLOW, ls="--", label=f"Overall fast-sale rate ({baseline:.1f}%)")],
               loc="lower center", frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); return fig


def mileage_price_scatter(df):
    """Mileage vs price for all cars. A falling line means: more miles, lower price."""
    import numpy as np
    d = df.dropna(subset=["Mileage_Num", "Price_Num"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for cls, col, lab in [(None, GRAY, "Unsold"), ("Slow", SLOW, "Slow Sale"), ("Fast", FAST, "Fast Sale")]:
        sub = d[d["Class"].isna()] if cls is None else d[d["Class"] == cls]
        ax.scatter(sub["Mileage_Num"] / 1000, sub["Price_Num"] / 1000, s=34, color=col, alpha=0.85, label=lab, edgecolor="white", linewidth=0.6)
    m, b = np.polyfit(d["Mileage_Num"] / 1000, d["Price_Num"] / 1000, 1)
    xs = np.array([d["Mileage_Num"].min() / 1000, d["Mileage_Num"].max() / 1000])
    ax.plot(xs, m * xs + b, color=NAVY, ls="--", lw=1.8, label="Trend line")
    ax.set_xlabel("Mileage (thousand miles)"); ax.set_ylabel("Listed price ($ thousand)")
    ax.legend(frameon=False, fontsize=8.5); _clean(ax)
    ax.set_title("More miles, lower price (all 100 cars)", fontsize=10, fontweight="bold", color=NAVY)
    fig.tight_layout(); return fig


def price_by_band(bands):
    """Average listed price for each mileage band."""
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    bars = ax.bar(list(bands.index), bands["mean"] / 1000, color=["#2A7F9E", "#4C97B0", "#7DB1C4", "#A9CBD7"][:len(bands)], width=0.6)
    for b, (_, r) in zip(bars, bands.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"${r['mean']:,.0f}", ha="center", fontweight="bold", fontsize=9)
        ax.text(b.get_x() + b.get_width() / 2, -2.6, f"n={int(r['count'])}", ha="center", fontsize=8, color=GRAY)
    ax.set_ylim(0, max(bands["mean"] / 1000) * 1.2); ax.set_ylabel("Average listed price ($ thousand)"); _clean(ax)
    ax.set_title("Average price by mileage band", fontsize=10, fontweight="bold", color=NAVY)
    fig.tight_layout(); return fig