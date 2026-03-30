# =========================================================
# 1. Causal Regression Modeling + Data capture + Figures
# =========================================================


import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


try:
    from scipy.stats import gaussian_kde
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -----------------------------
# Paths
# -----------------------------
PHASE1_PATH = "phase1.csv"
PHASE2_PATH = "phase2.csv"


# Outputs
OUT_EFFECTS_CSV = "causal_effects_tidy.csv"
OUT_BASELINE_ADJ_CSV = "causal_effects_baseline_adjusted_tidy.csv"
OUT_DIR = "C1_outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
DATA_DIR = os.path.join(OUT_DIR, "derived_data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# -----------------------------
# Global style
# -----------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.3,
})


SIG_NAVY = "#000080"
GRID_ALPHA = 0.18


# -----------------------------
# Parameters
# -----------------------------
RNG_SEED = 42
BOOT = 5000
PERM = 12000


TREATMENTS = ["SAL", "CST", "PST", "BOTH"]
CONTRASTS = [("CST","SAL"), ("PST","SAL"), ("BOTH","SAL"), ("CST","PST"), ("BOTH","CST")]
PRIMARY_CONTRASTS = [("CST","SAL"), ("PST","SAL"), ("BOTH","SAL")]


# =========================================================
# Helpers
# =========================================================
def _u(x):
    return str(x).strip().upper()


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.2, length=4)


def _panel_letter(ax, letter):
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="bottom", ha="left")


def _safe_col(df, col, default=np.nan):
    if col not in df.columns:
        df[col] = default
    return df


def add_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()


    # SAT endpoint
    if "SAT_10" in df.columns:
        df["SAT_END"] = pd.to_numeric(df["SAT_10"], errors="coerce")
    else:
        df["SAT_END"] = np.nan


    # MEM endpoint (Novel Arm)
    if "MEM_10" in df.columns:
        df["MEM_END"] = pd.to_numeric(df["MEM_10"], errors="coerce")
    else:
        df["MEM_END"] = np.nan


    return df


def cohens_d(x, y):
    """Standardized effect size: (mean(x)-mean(y))/pooled_sd on endpoint values."""
    x = np.asarray(x); y = np.asarray(y)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    s1 = np.var(x, ddof=1); s2 = np.var(y, ddof=1)
    sp = np.sqrt(((len(x)-1)*s1 + (len(y)-1)*s2) / (len(x)+len(y)-2))
    if sp == 0 or np.isnan(sp):
        return np.nan
    return (np.mean(x) - np.mean(y)) / sp


def bootstrap_ci_diff(x, y, n_boot=BOOT, seed=RNG_SEED, return_draws=False):
    """Bootstrap CI for mean difference using endpoint values only."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        if return_draws:
            return (np.nan, np.nan, np.nan, np.array([], dtype=float))
        return (np.nan, np.nan, np.nan)


    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs[i] = np.mean(xb) - np.mean(yb)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    ate = float(np.mean(x) - np.mean(y))
    if return_draws:
        return ate, float(lo), float(hi), diffs
    return ate, float(lo), float(hi)


def perm_pvalue_diff(x, y, n_perm=PERM, seed=RNG_SEED, return_null=False):
    """Permutation test for mean difference (endpoint only)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        if return_null:
            return np.nan, np.array([], dtype=float)
        return np.nan


    obs = np.mean(x) - np.mean(y)
    pooled = np.concatenate([x, y]).copy()
    n_x = len(x)
    null = np.empty(n_perm, dtype=float)


    count = 0
    for i in range(n_perm):
        rng.shuffle(pooled)
        d = np.mean(pooled[:n_x]) - np.mean(pooled[n_x:])
        null[i] = d
        if abs(d) >= abs(obs):
            count += 1


    p = (count + 1) / (n_perm + 1)
    if return_null:
        return p, null
    return p


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    """BH-FDR adjusted p-values."""
    p = pvals.values.astype(float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n-1, -1, -1):
        rank = i + 1
        val = (ranked[i] * n) / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0, 1)
    return pd.Series(out, index=pvals.index)


def baseline_adjusted_effect_phase1(df, outcome_end, baseline_col, treat, control,
                                    n_boot=BOOT, seed=RNG_SEED, return_draws=False):
    d = df[df["TREATMENT"].isin([treat, control])].copy()
    cols_needed = [outcome_end, baseline_col, "SEX", "TREATMENT"]
    d = d.dropna(subset=[c for c in cols_needed if c in d.columns])
    if len(d) < 10:
        if return_draws:
            return (np.nan, np.nan, np.nan, np.array([], dtype=float))
        return (np.nan, np.nan, np.nan)


    y = pd.to_numeric(d[outcome_end], errors="coerce").values.astype(float)
    baseline = pd.to_numeric(d[baseline_col], errors="coerce").values.astype(float)
    sexM = (d["SEX"].astype(str).str.upper() == "M").astype(int).values
    trt = (d["TREATMENT"] == treat).astype(int).values


    ok = ~np.isnan(y) & ~np.isnan(baseline)
    y = y[ok]; baseline = baseline[ok]; sexM = sexM[ok]; trt = trt[ok]


    X = np.column_stack([np.ones(len(y)), baseline, sexM, trt])


    def ols_beta(Xm, ym):
        XtX = Xm.T @ Xm
        if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
            return None
        return np.linalg.inv(XtX) @ (Xm.T @ ym)


    beta = ols_beta(X, y)
    if beta is None:
        if return_draws:
            return (np.nan, np.nan, np.nan, np.array([], dtype=float))
        return (np.nan, np.nan, np.nan)


    rng = np.random.default_rng(seed)
    b = np.empty(n_boot, dtype=float)
    idx = np.arange(len(y))
    for i in range(n_boot):
        ii = rng.choice(idx, size=len(y), replace=True)
        bi = ols_beta(X[ii], y[ii])
        b[i] = np.nan if bi is None else bi[3]
    b = b[~np.isnan(b)]
    if len(b) < 50:
        if return_draws:
            return (float(beta[3]), np.nan, np.nan, b)
        return (float(beta[3]), np.nan, np.nan)


    lo, hi = np.quantile(b, [0.025, 0.975])
    if return_draws:
        return (float(beta[3]), float(lo), float(hi), b)
    return (float(beta[3]), float(lo), float(hi))


# =========================================================
# Forest Plot
# =========================================================
def _ci_crosses_zero(lo, hi):
    return (lo <= 0) & (hi >= 0)


def _forest_f(ax, d, title, effect_col, lo_col, hi_col, xlabel):
    d = d.copy()
    for c in [effect_col, lo_col, hi_col, "Stratum", "Contrast"]:
        d = _safe_col(d, c)


    # numeric coercion
    d[effect_col] = pd.to_numeric(d[effect_col], errors="coerce")
    d[lo_col] = pd.to_numeric(d[lo_col], errors="coerce")
    d[hi_col] = pd.to_numeric(d[hi_col], errors="coerce")


    d = d.dropna(subset=[effect_col, lo_col, hi_col]).copy()
    if d.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No matching rows", ha="center", va="center", fontsize=12)
        return


    d["Stratum"] = d["Stratum"].astype(str)
    d["Contrast"] = d["Contrast"].astype(str)


    d = d.sort_values(["Stratum", "Contrast"]).reset_index(drop=True)


    eff = d[effect_col].values.astype(float)
    lo  = d[lo_col].values.astype(float)
    hi  = d[hi_col].values.astype(float)


    sig = ~_ci_crosses_zero(lo, hi)


    y = np.arange(len(d))[::-1]


    for i in range(len(d)):
        is_sig = sig[i]
        color = SIG_NAVY if is_sig else "black"
        lw = 2.8 if is_sig else 1.8
        alpha = 0.95 if is_sig else 0.55
        ax.hlines(y[i], lo[i], hi[i], lw=lw, alpha=alpha, color=color)
        ax.vlines([lo[i], hi[i]], y[i]-0.10, y[i]+0.10, lw=lw, alpha=alpha, color=color)


    ax.scatter(eff[sig], y[sig], s=55, color=SIG_NAVY, zorder=3)
    ax.scatter(eff[~sig], y[~sig], s=55, facecolors="white", edgecolors="black", linewidths=1.4, zorder=3)


    ax.axvline(0, ls="--", lw=1.6, color="black", alpha=0.7)


    labels = []
    for i in range(len(d)):
        base = f"{d.loc[i,'Stratum']} | {d.loc[i,'Contrast']}"
        labels.append(base + (" ★" if sig[i] else ""))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)


    for tick, is_sig in zip(ax.get_yticklabels(), sig[::-1]):
        if is_sig:
            tick.set_fontweight("bold")
            tick.set_color(SIG_NAVY)


    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=10)
    _clean_axes(ax)


    xmin, xmax = np.nanmin(lo), np.nanmax(hi)
    pad = 0.10 * (xmax - xmin + 1e-9)
    ax.set_xlim(xmin - pad, xmax + pad)


    ax.grid(axis="x", alpha=GRID_ALPHA)


def forest_2x2(unadj_df, filename, suptitle, effect_col, lo_col, hi_col, xlabel):
    fig = plt.figure(figsize=(13.6, 7.9))
    gs = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.42)


    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])


    def _subset(phase, outcome, geno_contains):
        d = unadj_df.copy()
        for c in ["PHASE","GENOTYPE","OUTCOME","Stratum","Contrast"]:
            d = _safe_col(d, c)
        d["PHASE"] = d["PHASE"].astype(str)
        d["GENOTYPE"] = d["GENOTYPE"].astype(str)
        d["OUTCOME"] = d["OUTCOME"].astype(str)
        mask = (
            d["PHASE"].str.upper().str.contains(str(phase).upper(), na=False) &
            d["GENOTYPE"].str.upper().str.contains(str(geno_contains).upper(), na=False) &
            (d["OUTCOME"].str.upper() == str(outcome).upper())
        )
        return d[mask].copy()


    d_p1_sat = _subset("Phase 1", "SAT_END", "CGA")
    d_p1_mem = _subset("Phase 1", "MEM_END", "CGA")
    d_p2_sat = _subset("Phase 2", "SAT_END", "PS19")
    d_p2_mem = _subset("Phase 2", "MEM_END", "PS19")


    _forest_f(axA, d_p1_sat, "Phase 1 (CgA-KO) — SAT_END", effect_col, lo_col, hi_col, xlabel)
    _forest_f(axB, d_p1_mem, "Phase 1 (CgA-KO) — MEM_END", effect_col, lo_col, hi_col, xlabel)
    _forest_f(axC, d_p2_sat, "Phase 2 (PS19) — SAT_END", effect_col, lo_col, hi_col, xlabel)
    _forest_f(axD, d_p2_mem, "Phase 2 (PS19) — MEM_END", effect_col, lo_col, hi_col, xlabel)


    _panel_letter(axA, "A"); _panel_letter(axB, "B"); _panel_letter(axC, "C"); _panel_letter(axD, "D")
    fig.suptitle(suptitle, fontsize=15, y=0.995)


    outpath = os.path.join(FIG_DIR, filename)
    fig.savefig(outpath, dpi=350, bbox_inches="tight")
    plt.close(fig)
    print("[SAVED]", outpath)


# =========================================================
# Tables + other plots
# =========================================================
def save_table_png(df, filename, title=None, floatfmt="{:.4g}"):
    fig, ax = plt.subplots(figsize=(12, max(2.5, 0.35 * len(df) + 1)))
    ax.axis("off")


    show = df.copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].map(lambda x: "" if pd.isna(x) else floatfmt.format(x))


    tbl = ax.table(cellText=show.values, colLabels=show.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)


    if title:
        ax.set_title(title, fontsize=13, pad=12)


    outpath = os.path.join(FIG_DIR, filename)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[SAVED]", outpath)


def volcano_plot(df, filename, q_col="p_BH", eff_col="Effect (mean diff)"):
    d = df.copy()
    d[q_col] = pd.to_numeric(d[q_col], errors="coerce")
    d[eff_col] = pd.to_numeric(d[eff_col], errors="coerce")
    d = d.dropna(subset=[q_col, eff_col]).copy()
    d["neglog10q"] = -np.log10(np.clip(d[q_col].values.astype(float), 1e-12, 1))


    sig = d[q_col].values < 0.05


    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d.loc[~sig, eff_col], d.loc[~sig, "neglog10q"],
               s=35, facecolors="white", edgecolors="black", linewidths=1.2)
    ax.scatter(d.loc[sig, eff_col], d.loc[sig, "neglog10q"],
               s=35, color=SIG_NAVY)


    ax.axvline(0, ls="--", lw=1.2, color="black", alpha=0.7)
    ax.axhline(-math.log10(0.05), ls="--", lw=1.2, color="black", alpha=0.6)


    ax.set_xlabel("Effect (mean difference)")
    ax.set_ylabel("–log10(q)")
    ax.set_title("Volcano Plot (Effect vs –log10(q))", pad=10)
    ax.grid(alpha=GRID_ALPHA)
    _clean_axes(ax)


    outpath = os.path.join(FIG_DIR, filename)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[SAVED]", outpath)


def adjusted_vs_unadjusted_scatter(unadj_df, adj_df, filename):
    u = unadj_df.copy()
    u = u[(u["PHASE"] == "Phase 1") & (u["Stratum"] == "ALL")].copy()
    u["Effect_unadj"] = pd.to_numeric(u["Effect (mean diff)"], errors="coerce")


    a = adj_df.copy()
    a = a[a["PHASE"] == "Phase 1"].copy()
    a["Effect_adj"] = pd.to_numeric(a["Effect"], errors="coerce")


    key = ["PHASE","OUTCOME","TREAT","CONTROL","Contrast"]
    m = pd.merge(
        u[key + ["Effect_unadj"]],
        a[key + ["Effect_adj"]],
        on=key, how="inner"
    ).dropna(subset=["Effect_unadj","Effect_adj"])


    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(m["Effect_unadj"], m["Effect_adj"], s=45,
               facecolors="white", edgecolors="black", linewidths=1.2)


    lo = float(min(m["Effect_unadj"].min(), m["Effect_adj"].min()))
    hi = float(max(m["Effect_unadj"].max(), m["Effect_adj"].max()))
    pad = 0.05 * (hi - lo + 1e-9)
    lo -= pad; hi += pad
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.6, color="black", alpha=0.7)


    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Unadjusted effect")
    ax.set_ylabel("Baseline-adjusted effect")
    ax.set_title("Adjusted vs Unadjusted Effects (Phase 1)", pad=10)
    ax.grid(alpha=GRID_ALPHA)
    _clean_axes(ax)


    outpath = os.path.join(FIG_DIR, filename)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[SAVED]", outpath)


def _kde_or_hist_line(ax, samples, label=None):
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) < 50:
        return
    if _HAS_SCIPY:
        kde = gaussian_kde(samples)
        xs = np.linspace(np.quantile(samples, 0.01), np.quantile(samples, 0.99), 250)
        ax.plot(xs, kde(xs), lw=2.3, label=label)
    else:
        hist, edges = np.histogram(samples, bins=40, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, hist, lw=2.3, label=label)


def bootstrap_kde_grid(draws_map, which_kind, filename, title):
    fig = plt.figure(figsize=(13.6, 7.9))
    gs = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.42)


    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])


    panels = [
        ("Phase 1", "SAT_END", axA, "A"),
        ("Phase 1", "MEM_END", axB, "B"),
        ("Phase 2", "SAT_END", axC, "C"),
        ("Phase 2", "MEM_END", axD, "D"),
    ]


    for phase, outcome, ax, letter in panels:
        ax.axvline(0, ls="--", lw=1.6, color="black", alpha=0.7)
        for treat, control in PRIMARY_CONTRASTS:
            key = (phase, outcome, "ALL", treat, control, which_kind)
            if key in draws_map and len(draws_map[key]) >= 50:
                _kde_or_hist_line(ax, draws_map[key], label=f"{treat}-{control}")


        ax.set_title(f"{phase} — {outcome}", pad=10)
        ax.set_xlabel("Effect")
        ax.set_ylabel("Density")
        ax.grid(alpha=GRID_ALPHA)
        _clean_axes(ax)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.legend(frameon=False, fontsize=9, loc="best")
        _panel_letter(ax, letter)


    fig.suptitle(title, fontsize=15, y=0.995)
    outpath = os.path.join(FIG_DIR, filename)
    fig.savefig(outpath, dpi=350, bbox_inches="tight")
    plt.close(fig)
    print("[SAVED]", outpath)


def permutation_nulls_grid(nulls_map, obs_df, filename, title):
    fig = plt.figure(figsize=(13.6, 7.9))
    gs = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.42)


    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])


    panels = [
        ("Phase 1", "SAT_END", axA, "A"),
        ("Phase 1", "MEM_END", axB, "B"),
        ("Phase 2", "SAT_END", axC, "C"),
        ("Phase 2", "MEM_END", axD, "D"),
    ]


    obs_df = obs_df.copy()
    obs_df["Effect (mean diff)"] = pd.to_numeric(obs_df["Effect (mean diff)"], errors="coerce")


    for phase, outcome, ax, letter in panels:
        ax.axvline(0, ls="--", lw=1.6, color="black", alpha=0.5)


        for treat, control in PRIMARY_CONTRASTS:
            key = (phase, outcome, "ALL", treat, control)
            null = nulls_map.get(key, None)
            if null is None or len(null) < 50:
                continue
            null = np.asarray(null, dtype=float)
            null = null[np.isfinite(null)]
            ax.hist(null, bins=45, density=True, histtype="step", lw=2.2,
                    label=f"Null {treat}-{control}")


            obs = obs_df[
                (obs_df["PHASE"] == phase) &
                (obs_df["OUTCOME"] == outcome) &
                (obs_df["Stratum"] == "ALL") &
                (obs_df["TREAT"] == treat) &
                (obs_df["CONTROL"] == control)
            ]["Effect (mean diff)"]
            if len(obs) and np.isfinite(obs.iloc[0]):
                ax.axvline(float(obs.iloc[0]), lw=2.6, color=SIG_NAVY)


        ax.set_title(f"{phase} — {outcome}", pad=10)
        ax.set_xlabel("Permuted effect")
        ax.set_ylabel("Density")
        ax.grid(alpha=GRID_ALPHA)
        _clean_axes(ax)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.legend(frameon=False, fontsize=9, loc="best")
        _panel_letter(ax, letter)


    fig.suptitle(title, fontsize=15, y=0.995)
    outpath = os.path.join(FIG_DIR, filename)
    fig.savefig(outpath, dpi=350, bbox_inches="tight")
    plt.close(fig)
    print("[SAVED]", outpath)


# =========================================================
# Runner
# =========================================================
def run_effects(df, phase_name, genotype_value, outcome_end,
                boot_store=None, perm_store=None):
    dphase = df[(df["GENOTYPE"] == genotype_value) & (df["TREATMENT"].isin(TREATMENTS))].copy()


    print("\n====================")
    print(f"{phase_name} | GENOTYPE={genotype_value} | Outcome={outcome_end} | N={len(dphase)}")
    print("====================")


    results = []
    anc_rows = []


    for strata_name, dsub in [("ALL", dphase),
                              ("F", dphase[dphase["SEX"] == "F"]),
                              ("M", dphase[dphase["SEX"] == "M"])]:


        for treat, control in CONTRASTS:
            a = pd.to_numeric(dsub.loc[dsub["TREATMENT"] == treat, outcome_end], errors="coerce").values.astype(float)
            b = pd.to_numeric(dsub.loc[dsub["TREATMENT"] == control, outcome_end], errors="coerce").values.astype(float)


            ate, lo, hi, bdraws = bootstrap_ci_diff(a, b, return_draws=True)
            p_perm, null = perm_pvalue_diff(a, b, return_null=True)
            d_eff = cohens_d(a, b)


            if boot_store is not None:
                boot_store[(phase_name, outcome_end, strata_name, treat, control, "unadjusted")] = bdraws
            if perm_store is not None:
                perm_store[(phase_name, outcome_end, strata_name, treat, control)] = null


            results.append({
                "Stratum": strata_name,
                "Contrast": f"{treat} - {control}",
                "Effect (mean diff)": ate,
                "95% CI low": lo,
                "95% CI high": hi,
                "Cohen_d": d_eff,
                "p_perm": p_perm,
                "n_treat": int(np.sum(~np.isnan(a))),
                "n_control": int(np.sum(~np.isnan(b))),
                "PHASE": phase_name,
                "GENOTYPE": genotype_value,
                "OUTCOME": outcome_end,
                "TREAT": treat,
                "CONTROL": control,
                "EFFECT_TYPE": "diff_in_means",
            })


    res = pd.DataFrame(results)


    res["p_BH"] = benjamini_hochberg(res["p_perm"].fillna(1.0))


    if genotype_value == "CGAKO" and phase_name == "Phase 1":
        baseline_col = "SAT_0" if outcome_end == "SAT_END" else "MEM_0"
        for treat, control in CONTRASTS:
            est, loA, hiA, adraws = baseline_adjusted_effect_phase1(
                dphase, outcome_end, baseline_col, treat, control, return_draws=True
            )
            if boot_store is not None:
                boot_store[(phase_name, outcome_end, "ALL", treat, control, "baseline_adjusted")] = adraws


            anc_rows.append({
                "PHASE": phase_name,
                "GENOTYPE": genotype_value,
                "OUTCOME": outcome_end,
                "Stratum": "ALL",
                "Contrast": f"{treat} - {control}",
                "TREAT": treat,
                "CONTROL": control,
                "EFFECT_TYPE": f"baseline_adjusted({baseline_col}+SEX)",
                "Effect": est,
                "95% CI low": loA,
                "95% CI high": hiA,
                "BaselineCov": baseline_col
            })


    return res, (pd.DataFrame(anc_rows) if len(anc_rows) else pd.DataFrame())


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":


    p1 = pd.read_csv(PHASE1_PATH)
    p2 = pd.read_csv(PHASE2_PATH)


    for df0 in (p1, p2):
        for c in ["GENOTYPE", "TREATMENT", "SEX"]:
            if c in df0.columns:
                df0[c] = df0[c].astype(str).map(_u)


    p1["PHASE"] = "PHASE1"
    p2["PHASE"] = "PHASE2"


    p1 = add_endpoints(p1)
    p2 = add_endpoints(p2)


    df = pd.concat([p1, p2], ignore_index=True)


    boot_draws = {}
    perm_nulls = {}


    all_res = []
    all_anc = []


    for phase_name, genotype, outcome in [
        ("Phase 1", "CGAKO", "SAT_END"),
        ("Phase 1", "CGAKO", "MEM_END"),
        ("Phase 2", "PS19", "SAT_END"),
        ("Phase 2", "PS19", "MEM_END"),
    ]:
        r, a = run_effects(df, phase_name, genotype, outcome,
                           boot_store=boot_draws, perm_store=perm_nulls)
        all_res.append(r)
        if len(a):
            all_anc.append(a)


    effects_tidy = pd.concat(all_res, ignore_index=True)
    effects_tidy.to_csv(OUT_EFFECTS_CSV, index=False)
    print(f"\n[SAVED] {OUT_EFFECTS_CSV}  ({len(effects_tidy)} rows)")


    anc_tidy = None
    if len(all_anc):
        anc_tidy = pd.concat(all_anc, ignore_index=True)
        anc_tidy.to_csv(OUT_BASELINE_ADJ_CSV, index=False)
        print(f"[SAVED] {OUT_BASELINE_ADJ_CSV}  ({len(anc_tidy)} rows)")


    # -----------------------------
    # Save derived data
    # -----------------------------
    effects_tidy.to_csv(os.path.join(DATA_DIR, "effects_unadjusted_full.csv"), index=False)


    perm_table = effects_tidy[[
        "PHASE","GENOTYPE","OUTCOME","Stratum","Contrast","TREAT","CONTROL",
        "Effect (mean diff)","95% CI low","95% CI high","p_perm","n_treat","n_control"
    ]].copy()
    perm_table.to_csv(os.path.join(DATA_DIR, "permutation_pvalues.csv"), index=False)


    bh_table = effects_tidy[[
        "PHASE","GENOTYPE","OUTCOME","Stratum","Contrast","TREAT","CONTROL",
        "Effect (mean diff)","p_perm","p_BH"
    ]].copy()
    bh_table.to_csv(os.path.join(DATA_DIR, "bh_qvalues.csv"), index=False)


    # Save draws (NPZ)
    np.savez_compressed(
        os.path.join(DATA_DIR, "bootstrap_draws.npz"),
        **{f"{k[0]}|{k[1]}|{k[2]}|{k[3]}|{k[4]}|{k[5]}": v for k, v in boot_draws.items()}
    )
    print("[SAVED]", os.path.join(DATA_DIR, "bootstrap_draws.npz"))


    np.savez_compressed(
        os.path.join(DATA_DIR, "permutation_nulls.npz"),
        **{f"{k[0]}|{k[1]}|{k[2]}|{k[3]}|{k[4]}": v for k, v in perm_nulls.items()}
    )
    print("[SAVED]", os.path.join(DATA_DIR, "permutation_nulls.npz"))


    # =========================================================
    # Figures
    # =========================================================


    # Forest Plot
    forest_2x2(
        effects_tidy,
        filename="C1_Forest_Unadjusted.png",
        suptitle="Causal Treatment Effects (Unadjusted)",
        effect_col="Effect (mean diff)",
        lo_col="95% CI low",
        hi_col="95% CI high",
        xlabel="Effect (mean difference) with 95% CI"
    )


    forest_2x2(
        dplot,
        filename="C1_Forest_CohensD.png",
        suptitle="Standardized Effect Size (Cohen’s d)",
        effect_col="Effect_d",
        lo_col="d_lo",
        hi_col="d_hi",
        xlabel="Cohen’s d (standardized units)"
    )


    # Permutation p-value Table
    save_table_png(
        perm_table.sort_values(["PHASE","OUTCOME","Stratum","Contrast"]).reset_index(drop=True),
        filename="C1_Table_PermutationP.png",
        title="Permutation p-values (12,000 permutations)"
    )


    # BH-Adjusted q-value Table
    save_table_png(
        bh_table.sort_values(["PHASE","OUTCOME","Stratum","Contrast"]).reset_index(drop=True),
        filename="C1_Table_BHq.png",
        title="BH-Adjusted q-values (FDR)"
    )


    # Volcano Plot (Effect vs –log10(q))
    volcano_plot(effects_tidy, filename="C1_Volcano_BHq.png")


    # Adjusted vs Unadjusted Scatter
    if anc_tidy is not None and len(anc_tidy):
        adjusted_vs_unadjusted_scatter(
            unadj_df=effects_tidy,
            adj_df=anc_tidy,
            filename="C1_Adjusted_vs_Unadjusted.png"
        )


    # Bootstrap KDE
    bootstrap_kde_grid(
        draws_map=boot_draws,
        which_kind="unadjusted",
        filename="C1_Bootstrap_KDE.png",
        title="Bootstrap Sampling Distributions"
    )


    # Permutation Null Histograms
    permutation_nulls_grid(
        nulls_map=perm_nulls,
        obs_df=effects_tidy,
        filename="C1_Permutation_Nulls.png",
        title="Permutation Null Distributions"
    )


    print("\nDone.")
    print("Figures:", FIG_DIR)
    print("Derived data:", DATA_DIR)
