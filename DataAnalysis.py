import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from scipy import stats
from collections import Counter

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#0f1117",
    "axes.edgecolor":   "#2a2d3a",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "monospace",
})

C_RATING  = "#58a6ff"   # blue  — ratings
C_WISH    = "#f78166"   # coral — wishlist
C_OVERLAP = "#3fb950"   # green — overlap
C_NEUTRAL = "#8b949e"   # grey  — neutral


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ax_hist(ax, data, bins, color, label, log_y=True):
    ax.hist(data, bins=bins, color=color, alpha=0.85, edgecolor="none", label=label)
    ax.set_yscale("log" if log_y else "linear")
    ax.grid(True)
    ax.legend(fontsize=8)


def _powerlaw_fit(degrees):
    """Fit and return (alpha, xmin, r²) for a power-law via log-log OLS."""
    counts = np.array(sorted(Counter(degrees).values(), reverse=True))
    ranks  = np.arange(1, len(counts) + 1)
    log_r  = np.log(ranks)
    log_c  = np.log(counts + 1e-9)
    slope, intercept, r, *_ = stats.linregress(log_r, log_c)
    return slope, intercept, r**2


def _summary_block(ax, lines):
    ax.axis("off")
    ax.text(0.05, 0.95, "\n".join(lines),
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            color="#c9d1d9",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22", edgecolor="#30363d"))


# ── Main Analysis ─────────────────────────────────────────────────────────────

def analyze(loader, save_path="analysis.png"):
    """
    Full analysis of the loaded data. Call after loader.load_and_split().

    Parameters
    ----------
    loader    : DataLoader instance (already called load_and_split())
    save_path : where to save the figure
    """
    train   = loader.train_df
    valid   = loader.valid_df
    wish    = loader.wishlist_df
    all_rat = pd.concat([train, valid], ignore_index=True)

    # ── Derived structures ────────────────────────────────────────────────────
    rat_user_deg  = all_rat.groupby("sid")["pid"].count()
    rat_item_deg  = all_rat.groupby("pid")["sid"].count()
    wish_user_deg = wish.groupby("sid")["pid"].count() if not wish.empty else pd.Series(dtype=int)
    wish_item_deg = wish.groupby("pid")["sid"].count() if not wish.empty else pd.Series(dtype=int)

    # Users / items present in both datasets
    rated_users  = set(all_rat["sid"].unique())
    wished_users = set(wish["sid"].unique()) if not wish.empty else set()
    rated_items  = set(all_rat["pid"].unique())
    wished_items = set(wish["pid"].unique()) if not wish.empty else set()

    overlap_users = rated_users & wished_users
    overlap_items = rated_items & wished_items

    # Per-user stats (only users in both)
    if overlap_users:
        common_users_df = pd.DataFrame({
            "rat_deg":  rat_user_deg.reindex(list(overlap_users)).fillna(0),
            "wish_deg": wish_user_deg.reindex(list(overlap_users)).fillna(0),
        })
        rat_deg_arr  = common_users_df["rat_deg"].values
        wish_deg_arr = common_users_df["wish_deg"].values
        corr_users   = np.corrcoef(rat_deg_arr, wish_deg_arr)[0, 1]
    else:
        common_users_df = pd.DataFrame()
        corr_users = float("nan")

    # Rating distribution
    rating_counts = all_rat["rating"].value_counts().sort_index()

    # Wishlist / rating overlap per user (aspirational gap)
    if overlap_users:
        rated_per_user  = all_rat.groupby("sid")["pid"].apply(set)
        wished_per_user = wish.groupby("sid")["pid"].apply(set)
        common_u_list   = list(overlap_users)
        wishlist_then_rated = pd.Series({
            u: len(rated_per_user.get(u, set()) & wished_per_user.get(u, set()))
            for u in common_u_list
        })
        total_wished = pd.Series({
            u: len(wished_per_user.get(u, set())) for u in common_u_list
        })
        completion_rate = (wishlist_then_rated / total_wished.replace(0, np.nan)).dropna()
    else:
        completion_rate = pd.Series(dtype=float)

    # Item popularity vs mean rating
    item_stats = all_rat.groupby("pid").agg(
        mean_rating=("rating", "mean"),
        n_ratings=("rating", "count")
    ).reset_index()

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 22), facecolor="#0f1117")
    gs  = gridspec.GridSpec(
        4, 4, figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.97, top=0.94, bottom=0.04
    )

    fig.suptitle("Dataset Analysis", fontsize=16, color="#e6edf3",
                 fontfamily="monospace", fontweight="bold", y=0.97)

    # ── Row 0: Degree distributions (4 panels) ────────────────────────────────
    ax_ru = fig.add_subplot(gs[0, 0])
    ax_ri = fig.add_subplot(gs[0, 1])
    ax_wu = fig.add_subplot(gs[0, 2])
    ax_wi = fig.add_subplot(gs[0, 3])

    bins_u = np.logspace(0, np.log10(rat_user_deg.max() + 1), 50)
    bins_i = np.logspace(0, np.log10(rat_item_deg.max() + 1), 50)

    _ax_hist(ax_ru, rat_user_deg,  bins_u, C_RATING, "user degree (ratings)")
    _ax_hist(ax_ri, rat_item_deg,  bins_i, C_RATING, "item degree (ratings)")

    if not wish_user_deg.empty:
        bins_wu = np.logspace(0, np.log10(wish_user_deg.max() + 1), 50)
        bins_wi = np.logspace(0, np.log10(wish_item_deg.max() + 1), 50)
        _ax_hist(ax_wu, wish_user_deg, bins_wu, C_WISH, "user degree (wishlist)")
        _ax_hist(ax_wi, wish_item_deg, bins_wi, C_WISH, "item degree (wishlist)")

    for ax in [ax_ru, ax_ri, ax_wu, ax_wi]:
        ax.set_xscale("log")
        ax.set_xlabel("degree (log)", fontsize=8)
        ax.set_ylabel("count (log)", fontsize=8)

    # Annotate power-law slope
    for ax, deg in [(ax_ru, rat_user_deg), (ax_ri, rat_item_deg),
                    (ax_wu, wish_user_deg), (ax_wi, wish_item_deg)]:
        if len(deg) > 10:
            slope, _, r2 = _powerlaw_fit(deg.values)
            ax.set_title(f"α={slope:.2f}  R²={r2:.2f}", fontsize=8, color=C_NEUTRAL)

    # ── Row 1: Rating dist | Sparsity | User overlap | Item overlap ───────────
    ax_rdist = fig.add_subplot(gs[1, 0])
    ax_spar  = fig.add_subplot(gs[1, 1])
    ax_uovlp = fig.add_subplot(gs[1, 2])
    ax_iovlp = fig.add_subplot(gs[1, 3])

    # Rating distribution
    bars = ax_rdist.bar(rating_counts.index, rating_counts.values,
                        color=C_RATING, alpha=0.85, edgecolor="none")
    for bar, v in zip(bars, rating_counts.values):
        ax_rdist.text(bar.get_x() + bar.get_width()/2, v + rating_counts.max()*0.01,
                      f"{v:,}", ha="center", va="bottom", fontsize=7, color=C_NEUTRAL)
    ax_rdist.set_xlabel("rating value", fontsize=8)
    ax_rdist.set_ylabel("count", fontsize=8)
    ax_rdist.set_title("Rating Distribution", fontsize=9)
    ax_rdist.grid(True, axis="y")

    # Sparsity summary
    n_users   = loader.num_users
    n_items   = loader.num_items
    n_rat     = len(all_rat)
    n_wish    = len(wish)
    sparsity_rat  = 1 - n_rat  / (n_users * n_items)
    sparsity_wish = 1 - n_wish / (n_users * n_items)
    _summary_block(ax_spar, [
        "── Sparsity ──────────────────",
        f"  Users (max id+1) : {n_users:>10,}",
        f"  Items (max id+1) : {n_items:>10,}",
        f"  Matrix size      : {n_users*n_items:>10,}",
        "",
        f"  Rating edges     : {n_rat:>10,}",
        f"  Rating sparsity  : {sparsity_rat:>10.6f}",
        "",
        f"  Wishlist edges   : {n_wish:>10,}",
        f"  Wishlist sparsity: {sparsity_wish:>10.6f}",
        "",
        f"  Train / Valid    : {len(train):,} / {len(valid):,}",
    ])

    # User overlap (Venn-style bar)
    u_only_rat  = len(rated_users - wished_users)
    u_only_wish = len(wished_users - rated_users)
    u_both      = len(overlap_users)
    ax_uovlp.barh(["only rated", "both", "only wished"],
                  [u_only_rat, u_both, u_only_wish],
                  color=[C_RATING, C_OVERLAP, C_WISH], alpha=0.85)
    ax_uovlp.set_xlabel("user count", fontsize=8)
    ax_uovlp.set_title("User Overlap", fontsize=9)
    ax_uovlp.grid(True, axis="x")

    # Item overlap
    i_only_rat  = len(rated_items - wished_items)
    i_only_wish = len(wished_items - rated_items)
    i_both      = len(overlap_items)
    ax_iovlp.barh(["only rated", "both", "only wished"],
                  [i_only_rat, i_both, i_only_wish],
                  color=[C_RATING, C_OVERLAP, C_WISH], alpha=0.85)
    ax_iovlp.set_xlabel("item count", fontsize=8)
    ax_iovlp.set_title("Item Overlap", fontsize=9)
    ax_iovlp.grid(True, axis="x")

    # ── Row 2: User rating vs wish degree | Completion rate | Popularity-rating ─
    ax_corr  = fig.add_subplot(gs[2, 0:2])
    ax_comp  = fig.add_subplot(gs[2, 2])
    ax_popr  = fig.add_subplot(gs[2, 3])

    # Scatter: user rating degree vs wishlist degree
    if not common_users_df.empty:
        ax_corr.scatter(common_users_df["rat_deg"], common_users_df["wish_deg"],
                        alpha=0.3, s=8, c=C_OVERLAP, linewidths=0)
        # Add regression line
        x = common_users_df["rat_deg"].values
        y = common_users_df["wish_deg"].values
        m, b = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 100)
        ax_corr.plot(xr, m*xr + b, color=C_WISH, lw=1.5, label=f"r={corr_users:.2f}")
        ax_corr.set_xlabel("# items rated", fontsize=8)
        ax_corr.set_ylabel("# items wishlisted", fontsize=8)
        ax_corr.set_title("User Activity: Ratings vs Wishlist Degree", fontsize=9)
        ax_corr.legend(fontsize=8)
        ax_corr.grid(True)

    # Wishlist completion rate histogram
    if not completion_rate.empty:
        ax_comp.hist(completion_rate, bins=30, color=C_OVERLAP, alpha=0.85, edgecolor="none")
        ax_comp.axvline(completion_rate.mean(), color=C_WISH, lw=1.5,
                        label=f"mean={completion_rate.mean():.2f}")
        ax_comp.set_xlabel("fraction of wishlist later rated", fontsize=8)
        ax_comp.set_ylabel("user count", fontsize=8)
        ax_comp.set_title("Wishlist Completion Rate", fontsize=9)
        ax_comp.legend(fontsize=8)
        ax_comp.grid(True)

    # Popularity (log n_ratings) vs mean rating — hexbin
    hb = ax_popr.hexbin(
        np.log1p(item_stats["n_ratings"]),
        item_stats["mean_rating"],
        gridsize=35, cmap="Blues", norm=LogNorm(), linewidths=0.2
    )
    ax_popr.set_xlabel("log(1 + n_ratings)", fontsize=8)
    ax_popr.set_ylabel("mean rating", fontsize=8)
    ax_popr.set_title("Item Popularity vs Mean Rating", fontsize=9)
    plt.colorbar(hb, ax=ax_popr, label="item count")

    # ── Row 3: User bias | Item bias | Rating entropy | Summary ──────────────
    ax_ubias = fig.add_subplot(gs[3, 0])
    ax_ibias = fig.add_subplot(gs[3, 1])
    ax_entr  = fig.add_subplot(gs[3, 2])
    ax_sum   = fig.add_subplot(gs[3, 3])

    # User rating bias (mean rating per user - global mean)
    global_mean = all_rat["rating"].mean()
    user_mean   = all_rat.groupby("sid")["rating"].mean()
    user_bias   = user_mean - global_mean
    ax_ubias.hist(user_bias, bins=50, color=C_RATING, alpha=0.85, edgecolor="none")
    ax_ubias.axvline(0, color=C_WISH, lw=1.5, linestyle="--")
    ax_ubias.set_xlabel("user mean − global mean", fontsize=8)
    ax_ubias.set_ylabel("user count", fontsize=8)
    ax_ubias.set_title("User Rating Bias", fontsize=9)
    ax_ubias.grid(True)

    # Item rating bias
    item_mean = all_rat.groupby("pid")["rating"].mean()
    item_bias = item_mean - global_mean
    ax_ibias.hist(item_bias, bins=50, color=C_RATING, alpha=0.85, edgecolor="none")
    ax_ibias.axvline(0, color=C_WISH, lw=1.5, linestyle="--")
    ax_ibias.set_xlabel("item mean − global mean", fontsize=8)
    ax_ibias.set_ylabel("item count", fontsize=8)
    ax_ibias.set_title("Item Rating Bias", fontsize=9)
    ax_ibias.grid(True)

    # Per-user rating entropy (how spread is a user's rating distribution)
    def user_entropy(group):
        counts = group["rating"].value_counts(normalize=True)
        return stats.entropy(counts)

    entropies = all_rat.groupby("sid").filter(lambda g: len(g) >= 3) \
                       .groupby("sid").apply(user_entropy)
    ax_entr.hist(entropies, bins=40, color=C_WISH, alpha=0.85, edgecolor="none")
    ax_entr.axvline(entropies.mean(), color=C_RATING, lw=1.5,
                    label=f"mean={entropies.mean():.2f}")
    ax_entr.set_xlabel("entropy (nats)", fontsize=8)
    ax_entr.set_ylabel("user count", fontsize=8)
    ax_entr.set_title("Per-User Rating Entropy", fontsize=9)
    ax_entr.legend(fontsize=8)
    ax_entr.grid(True)

    # Summary stats block
    _summary_block(ax_sum, [
        "── Key Stats ─────────────────",
        f"  Global mean rating  : {global_mean:.4f}",
        f"  Global std  rating  : {all_rat['rating'].std():.4f}",
        "",
        f"  Median user degree  : {rat_user_deg.median():.0f}",
        f"  Median item degree  : {rat_item_deg.median():.0f}",
        f"  Max user degree     : {rat_user_deg.max()}",
        f"  Max item degree     : {rat_item_deg.max()}",
        "",
        f"  User bias std       : {user_bias.std():.4f}",
        f"  Item bias std       : {item_bias.std():.4f}",
        "",
        f"  Wish completion mean: {completion_rate.mean():.4f}" if not completion_rate.empty else "  Wish completion     : n/a",
        f"  User activity corr  : {corr_users:.4f}",
    ])

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"Saved → {save_path}")


# ── Usage ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from helper.DataLoader import DataLoader          # adjust import to your project

    loader = DataLoader(data_dir="data/")
    loader.load_and_split()
    analyze(loader, save_path="analysis.png")