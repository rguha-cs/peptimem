# ============================================================
# 2. Residual ML Framework (Two-stage) + Data capture + Figures
# ============================================================


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.calibration import calibration_curve


# -----------------------------
# Paths to Data
# -----------------------------
PHASE1_PATH = "phase1.csv"
PHASE2_PATH = "phase2.csv"


# Output Files
OUT_METRICS = "two_stage_metrics_tidy.csv"
OUT_PREDS   = "two_stage_predictions_tidy.csv"
OUT_COEFS   = "two_stage_stage2_coefficients_tidy.csv"
OUT_AUCBOOT = "two_stage_auc_bootstrap_tidy.csv"


FIG_DIR = "figures_component2"
os.makedirs(FIG_DIR, exist_ok=True)


RNG_SEED = 42
BOOT_AUC = 3000


# -----------------------------
# Load + harmonize Phase 2
# -----------------------------
p1 = pd.read_csv(PHASE1_PATH)
p2 = pd.read_csv(PHASE2_PATH)


def _u(x): return str(x).strip().upper()
def to_num(x): return pd.to_numeric(x, errors="coerce")


for df in (p1, p2):
    for c in ["GENOTYPE","TREATMENT","SEX"]:
        if c in df.columns:
            df[c] = df[c].astype(str).map(_u)
    for c in ["MEM_0","MEM_5","MEM_10","SAT_0","SAT_5","SAT_10"]:
        if c in df.columns:
            df[c] = to_num(df[c])


def add_early_features(df):
    df = df.copy()
    if "SAT_0" in df.columns and "SAT_5" in df.columns:
        df["dSAT_0_5"] = df["SAT_5"] - df["SAT_0"]
        df["SAT_slope_0_5"] = df["dSAT_0_5"] / 5.0
    if "MEM_0" in df.columns and "MEM_5" in df.columns:
        df["dMEM_0_5"] = df["MEM_5"] - df["MEM_0"]
        df["MEM_slope_0_5"] = df["dMEM_0_5"] / 5.0
    return df


p1 = add_early_features(p1)
p2 = add_early_features(p2)


train_all = p1[p1["GENOTYPE"]=="CGAKO"].copy()
test_all  = p2[p2["GENOTYPE"]=="PS19"].copy()


TARGET_W10 = "MEM_10"
CONTROL_TREAT = "SAL"
SEX_COL = "SEX"
TRT_COL = "TREATMENT"


BASELINE_PREDICTORS = ["MEM_0","SAT_0","SEX"]


STAGE2_NUMERIC_CANDIDATES = [
    "MEM_0","SAT_0","SAT_5","dSAT_0_5","SAT_slope_0_5",
    "MEM_5","dMEM_0_5","MEM_slope_0_5"
]


# -----------------------------
# Preprocessing
# -----------------------------
def make_preprocessor(feature_list, categorical_cols):
    cat_cols = [c for c in categorical_cols if c in feature_list]
    num_cols = [c for c in feature_list if c not in cat_cols]


    return ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"))
            ]), cat_cols),
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
        ],
        remainder="drop"
    )


# -----------------------------
# Metrics helpers
# -----------------------------
def balanced_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = np.sum((y_true==1) & (y_pred==1))
    fn = np.sum((y_true==1) & (y_pred==0))
    tn = np.sum((y_true==0) & (y_pred==0))
    fp = np.sum((y_true==0) & (y_pred==1))
    tpr = tp / (tp + fn) if (tp+fn)>0 else np.nan
    tnr = tn / (tn + fp) if (tn+fp)>0 else np.nan
    return np.nanmean([tpr, tnr])


def auc_bootstrap_dist(y_true, y_prob, n_boot=BOOT_AUC, seed=RNG_SEED):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return np.array([])
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    aucs = []
    for _ in range(n_boot):
        ii = rng.choice(idx, size=len(idx), replace=True)
        yt, yp = y_true[ii], y_prob[ii]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    return np.asarray(aucs, dtype=float)


def auc_bootstrap_ci(y_true, y_prob, n_boot=BOOT_AUC, seed=RNG_SEED):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return (np.nan, np.nan, np.nan)
    aucs = auc_bootstrap_dist(y_true, y_prob, n_boot=n_boot, seed=seed)
    if len(aucs) < 50:
        a = roc_auc_score(y_true, y_prob)
        return (a, np.nan, np.nan)
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    return (roc_auc_score(y_true, y_prob), float(lo), float(hi))


def bootstrap_roc_ci(y_true, y_prob, n_boot=2000, seed=RNG_SEED, grid=None):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    if grid is None:
        grid = np.linspace(0, 1, 101)
    if len(np.unique(y_true)) < 2 or len(y_true) < 2:
        return grid, np.full_like(grid, np.nan), np.full_like(grid, np.nan), np.full_like(grid, np.nan)


    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    tprs = []


    for _ in range(n_boot):
        ii = rng.choice(idx, size=len(idx), replace=True)
        yt, yp = y_true[ii], y_prob[ii]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        tpr_i = np.interp(grid, fpr, tpr, left=0.0, right=1.0)
        tprs.append(tpr_i)


    if len(tprs) < 50:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        mean_tpr = np.interp(grid, fpr, tpr, left=0.0, right=1.0)
        return grid, mean_tpr, np.full_like(grid, np.nan), np.full_like(grid, np.nan)


    tprs = np.vstack(tprs)
    mean_tpr = np.nanmean(tprs, axis=0)
    lo = np.nanquantile(tprs, 0.025, axis=0)
    hi = np.nanquantile(tprs, 0.975, axis=0)
    return grid, mean_tpr, lo, hi


def sens_spec(cm):
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return float(sens), float(spec)


# -----------------------------
# Stage 1 baseline model
# -----------------------------
def fit_saline_baseline_model(df_train, pooled=True):
    d = df_train[df_train[TRT_COL]==CONTROL_TREAT].copy()
    needed = [TARGET_W10] + [c for c in BASELINE_PREDICTORS if c in d.columns]
    d = d.dropna(subset=needed)


    baseline_predictors = BASELINE_PREDICTORS.copy()
    if not pooled and "SEX" in baseline_predictors:
        baseline_predictors.remove("SEX")


    cat_cols = ["SEX"] if pooled and "SEX" in baseline_predictors else []
    feature_list = [c for c in baseline_predictors if c in d.columns]


    if len(d) < 6:
        return None, feature_list


    X = d[feature_list]
    y = d[TARGET_W10].astype(float)


    pre = make_preprocessor(feature_list, cat_cols)
    ridge = Ridge(alpha=1.0, random_state=RNG_SEED)
    pipe = Pipeline([("prep", pre), ("model", ridge)])
    pipe.fit(X, y)
    return pipe, feature_list


def predict_expected(pipe, feats, df):
    return pipe.predict(df[feats])


# -----------------------------
# Stage 2 classifier
# -----------------------------
def make_stage2_classifier():
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=1.0,
        max_iter=20000,
        class_weight="balanced",
        random_state=RNG_SEED
    )


# -----------------------------
# Runner helpers
# -----------------------------
def ensure_columns(df, cols):
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


METRICS_ROWS = []
PRED_ROWS = []
COEF_ROWS = []
AUC_BOOT_ROWS = []


def _safe_get(row, key):
    return row[key] if key in row.index else np.nan


# -----------------------------
# Runner
# -----------------------------
def run_two_stage(pooled=True, sex_value=None, include_treatment=True):
    df_tr = train_all.copy()
    df_te = test_all.copy()


    tag = "POOLED"
    if not pooled:
        tag = f"SEX={sex_value}"
        df_tr = df_tr[df_tr[SEX_COL]==sex_value].copy()
        df_te = df_te[df_te[SEX_COL]==sex_value].copy()


    # Stage2 feature list
    stage2_cat = ["SEX"] if pooled else []
    if include_treatment:
        stage2_cat = ["TREATMENT"] + stage2_cat


    stage2_features = stage2_cat + [c for c in STAGE2_NUMERIC_CANDIDATES if c in df_tr.columns]
    stage2_features_ext = [c for c in stage2_features if c in df_te.columns]


    print("\n" + "="*75)
    print(f"TWO-STAGE RESIDUAL ML | {tag} | include_treatment={include_treatment} | Target={TARGET_W10}")
    print("="*75)
    print("Stage2 Features (CV):", stage2_features)
    print("Stage2 Features (External):", stage2_features_ext)


    df_tr = df_tr.dropna(subset=[TARGET_W10]).reset_index(drop=True)
    df_te = df_te.dropna(subset=[TARGET_W10]).reset_index(drop=True)


    run_id = f"{tag}|include_treatment={include_treatment}"
    metrics = {
        "run_id": run_id,
        "setting": tag,
        "include_treatment": bool(include_treatment),
        "target": TARGET_W10,
        "stage2_features_cv": "|".join(stage2_features),
        "stage2_features_external_present": "|".join(stage2_features_ext),
        "n_train_raw": int(len(df_tr)),
        "n_test_raw": int(len(df_te)),
        "cv_n": 0, "cv_bacc": np.nan, "cv_auc": np.nan, "cv_auc_lo": np.nan, "cv_auc_hi": np.nan,
        "cv_cm_tn": np.nan, "cv_cm_fp": np.nan, "cv_cm_fn": np.nan, "cv_cm_tp": np.nan,
        "ext_n": 0, "ext_bacc": np.nan, "ext_auc": np.nan, "ext_auc_lo": np.nan, "ext_auc_hi": np.nan,
        "ext_cm_tn": np.nan, "ext_cm_fp": np.nan, "ext_cm_fn": np.nan, "ext_cm_tp": np.nan,
    }


    if len(df_tr) < 12:
        print("Not enough training mice.")
        METRICS_ROWS.append(metrics)
        return


    # Stratify CV by median target
    strat = (df_tr[TARGET_W10] > df_tr[TARGET_W10].median()).astype(int).values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)


    y_true_all, y_pred_all, y_prob_all = [], [], []


    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(df_tr, strat), start=1):
        fold_train = df_tr.iloc[tr_idx].copy()
        fold_val   = df_tr.iloc[va_idx].copy()


        baseline_pipe, baseline_feats = fit_saline_baseline_model(fold_train, pooled=pooled)


        # If baseline model missing, fallback to SAL mean
        if baseline_pipe is None:
            sal_fold = fold_train[fold_train[TRT_COL]==CONTROL_TREAT][TARGET_W10]
            mean_sal = sal_fold.mean()
            if pd.isna(mean_sal):
                mean_sal = fold_train[TARGET_W10].mean()
            fold_train["_expected"] = mean_sal
            fold_val["_expected"]   = mean_sal
        else:
            fold_train = fold_train.dropna(subset=baseline_feats)
            fold_val   = fold_val.dropna(subset=baseline_feats)
            fold_train["_expected"] = predict_expected(baseline_pipe, baseline_feats, fold_train)
            fold_val["_expected"]   = predict_expected(baseline_pipe, baseline_feats, fold_val)


        # residuals
        fold_train["_resid"] = fold_train[TARGET_W10] - fold_train["_expected"]
        fold_val["_resid"]   = fold_val[TARGET_W10]   - fold_val["_expected"]


        # responder threshold from SAL residuals in fold_train ONLY
        sal_res = fold_train.loc[fold_train[TRT_COL]==CONTROL_TREAT, "_resid"].dropna().values
        thr = 0.0 if len(sal_res) < 3 else float(np.mean(sal_res) + np.std(sal_res, ddof=1))


        fold_train["Y"] = (fold_train["_resid"] > thr).astype(int)
        fold_val["Y"]   = (fold_val["_resid"]   > thr).astype(int)


        # Stage2 sets (no leakage)
        fold_train = fold_train.dropna(subset=stage2_features + ["Y"])
        fold_val   = fold_val.dropna(subset=stage2_features + ["Y"])


        if fold_train["Y"].nunique() < 2 or fold_val["Y"].nunique() < 2:
            continue


        X_tr = fold_train[stage2_features]
        y_tr = fold_train["Y"].astype(int)
        X_va = fold_val[stage2_features]
        y_va = fold_val["Y"].astype(int)


        stage2_pipe = Pipeline([
            ("prep", make_preprocessor(stage2_features, stage2_cat)),
            ("model", make_stage2_classifier())
        ])
        stage2_pipe.fit(X_tr, y_tr)


        pred = stage2_pipe.predict(X_va)
        prob = stage2_pipe.predict_proba(X_va)[:, 1]


        # Store per-mouse CV OOF preds
        for j, idx in enumerate(X_va.index):
            r = fold_val.loc[idx]
            PRED_ROWS.append({
                "run_id": run_id,
                "setting": tag,
                "include_treatment": bool(include_treatment),
                "split": "cv_oof",
                "fold": fold_i,
                "MOUSE_ID": _safe_get(r, "MOUSE_ID"),
                "GENOTYPE": _safe_get(r, "GENOTYPE"),
                "SEX": _safe_get(r, "SEX"),
                "TREATMENT": _safe_get(r, "TREATMENT"),
                "target": TARGET_W10,
                "y_value": float(r[TARGET_W10]) if pd.notna(r[TARGET_W10]) else np.nan,
                "expected": float(r["_expected"]) if pd.notna(r["_expected"]) else np.nan,
                "resid": float(r["_resid"]) if pd.notna(r["_resid"]) else np.nan,
                "thr": float(thr),
                "y_true_responder": int(r["Y"]) if pd.notna(r["Y"]) else np.nan,
                "y_pred_responder": int(pred[j]),
                "y_prob_responder": float(prob[j]),
            })


        y_true_all.extend(y_va.tolist())
        y_pred_all.extend(pred.tolist())
        y_prob_all.extend(prob.tolist())


    # CV report + save metrics + AUC boot dist
    if len(y_true_all) < 10 or len(np.unique(y_true_all)) < 2:
        print("CV not reliable (too few labeled samples or one class).")
    else:
        y_true = np.array(y_true_all)
        y_pred = np.array(y_pred_all)
        y_prob = np.array(y_prob_all)


        auc, lo, hi = auc_bootstrap_ci(y_true, y_prob)
        bacc = balanced_accuracy(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()


        print("\n[Phase 1 CV]")
        print(f"Balanced Acc: {bacc:.3f}")
        print(f"ROC AUC     : {auc:.3f} (95% CI {lo:.3f} to {hi:.3f})")
        print("Confusion Matrix:\n", cm)


        metrics.update({
            "cv_n": int(len(y_true)),
            "cv_bacc": float(bacc),
            "cv_auc": float(auc),
            "cv_auc_lo": float(lo) if lo is not None else np.nan,
            "cv_auc_hi": float(hi) if hi is not None else np.nan,
            "cv_cm_tn": int(tn), "cv_cm_fp": int(fp), "cv_cm_fn": int(fn), "cv_cm_tp": int(tp),
        })


        aucs = auc_bootstrap_dist(y_true, y_prob)
        for a in aucs:
            AUC_BOOT_ROWS.append({"run_id": run_id, "setting": tag, "include_treatment": bool(include_treatment),
                                  "split": "cv_oof", "auc_boot": float(a)})


    # External test + save metrics/preds/coefs
    baseline_pipe, baseline_feats = fit_saline_baseline_model(df_tr, pooled=pooled)


    if baseline_pipe is None:
        sal_all = df_tr[df_tr[TRT_COL]==CONTROL_TREAT][TARGET_W10]
        mean_sal = sal_all.mean()
        if pd.isna(mean_sal):
            mean_sal = df_tr[TARGET_W10].mean()
        df_tr["_expected"] = mean_sal
        df_te["_expected"] = mean_sal
    else:
        df_tr = df_tr.dropna(subset=baseline_feats)
        df_te = df_te.dropna(subset=baseline_feats)
        df_tr["_expected"] = predict_expected(baseline_pipe, baseline_feats, df_tr)
        df_te["_expected"] = predict_expected(baseline_pipe, baseline_feats, df_te)


    df_tr["_resid"] = df_tr[TARGET_W10] - df_tr["_expected"]
    df_te["_resid"] = df_te[TARGET_W10] - df_te["_expected"]


    sal_res = df_tr.loc[df_tr[TRT_COL]==CONTROL_TREAT, "_resid"].dropna().values
    thr = 0.0 if len(sal_res) < 3 else float(np.mean(sal_res) + np.std(sal_res, ddof=1))


    df_tr["Y"] = (df_tr["_resid"] > thr).astype(int)
    df_te["Y"] = (df_te["_resid"] > thr).astype(int)


    df_tr2 = df_tr.dropna(subset=stage2_features + ["Y"])
    df_te2 = df_te.dropna(subset=stage2_features_ext + ["Y"])


    if len(df_te2) < 8 or df_tr2["Y"].nunique() < 2 or df_te2["Y"].nunique() < 2:
        print("\nExternal test not reliable (too few samples or one class).")
        METRICS_ROWS.append(metrics)
        return


    df_te2 = ensure_columns(df_te2, stage2_features)


    X_train = df_tr2[stage2_features]
    y_train = df_tr2["Y"].astype(int)


    X_test  = df_te2[stage2_features]
    y_test  = df_te2["Y"].astype(int)


    ext_pipe = Pipeline([
        ("prep", make_preprocessor(stage2_features, stage2_cat)),
        ("model", make_stage2_classifier())
    ])
    ext_pipe.fit(X_train, y_train)


    y_pred = ext_pipe.predict(X_test)
    y_prob = ext_pipe.predict_proba(X_test)[:, 1]


    auc, lo, hi = auc_bootstrap_ci(y_test.values, y_prob)
    bacc = balanced_accuracy(y_test.values, y_pred)
    cm = confusion_matrix(y_test.values, y_pred)
    tn, fp, fn, tp = cm.ravel()


    print("\n[External Test (PS19)]")
    print(f"Balanced Acc: {bacc:.3f}")
    print(f"ROC AUC     : {auc:.3f} (95% CI {lo:.3f} to {hi:.3f})")
    print("Confusion Matrix:\n", cm)


    metrics.update({
        "ext_n": int(len(y_test)),
        "ext_bacc": float(bacc),
        "ext_auc": float(auc),
        "ext_auc_lo": float(lo) if lo is not None else np.nan,
        "ext_auc_hi": float(hi) if hi is not None else np.nan,
        "ext_cm_tn": int(tn), "ext_cm_fp": int(fp), "ext_cm_fn": int(fn), "ext_cm_tp": int(tp),
    })


    aucs = auc_bootstrap_dist(y_test.values, y_prob)
    for a in aucs:
        AUC_BOOT_ROWS.append({"run_id": run_id, "setting": tag, "include_treatment": bool(include_treatment),
                              "split": "external", "auc_boot": float(a)})


    # store external per-mouse predictions
    for i in range(len(df_te2)):
        r = df_te2.iloc[i]
        PRED_ROWS.append({
            "run_id": run_id,
            "setting": tag,
            "include_treatment": bool(include_treatment),
            "split": "external",
            "fold": np.nan,
            "MOUSE_ID": _safe_get(r, "MOUSE_ID"),
            "GENOTYPE": _safe_get(r, "GENOTYPE"),
            "SEX": _safe_get(r, "SEX"),
            "TREATMENT": _safe_get(r, "TREATMENT"),
            "target": TARGET_W10,
            "y_value": float(r[TARGET_W10]) if pd.notna(r[TARGET_W10]) else np.nan,
            "expected": float(r["_expected"]) if pd.notna(r["_expected"]) else np.nan,
            "resid": float(r["_resid"]) if pd.notna(r["_resid"]) else np.nan,
            "thr": float(thr),
            "y_true_responder": int(r["Y"]) if pd.notna(r["Y"]) else np.nan,
            "y_pred_responder": int(y_pred[i]),
            "y_prob_responder": float(y_prob[i]),
        })


    # store stage2 coefficients
    try:
        feat_names = ext_pipe.named_steps["prep"].get_feature_names_out()
    except Exception:
        feat_names = np.array([f"f{i}" for i in range(len(ext_pipe.named_steps["model"].coef_.ravel()))])


    coefs = ext_pipe.named_steps["model"].coef_.ravel()
    for f, c in zip(feat_names, coefs):
        COEF_ROWS.append({
            "run_id": run_id,
            "setting": tag,
            "include_treatment": bool(include_treatment),
            "feature": str(f),
            "coef": float(c),
        })


    METRICS_ROWS.append(metrics)


# -----------------------------
# Run
# -----------------------------
for include_treatment in [True, False]:
    run_two_stage(pooled=True, include_treatment=include_treatment)
    run_two_stage(pooled=False, sex_value="F", include_treatment=include_treatment)
    run_two_stage(pooled=False, sex_value="M", include_treatment=include_treatment)


# -----------------------------
# Save results
# -----------------------------
metrics_df = pd.DataFrame(METRICS_ROWS)
preds_df   = pd.DataFrame(PRED_ROWS)
coefs_df   = pd.DataFrame(COEF_ROWS)
aucboot_df = pd.DataFrame(AUC_BOOT_ROWS)


metrics_df.to_csv(OUT_METRICS, index=False)
preds_df.to_csv(OUT_PREDS, index=False)
coefs_df.to_csv(OUT_COEFS, index=False)
aucboot_df.to_csv(OUT_AUCBOOT, index=False)


print(f"\n[SAVED] {OUT_METRICS}  ({len(metrics_df)} rows)")
print(f"[SAVED] {OUT_PREDS}    ({len(preds_df)} rows)")
print(f"[SAVED] {OUT_COEFS}    ({len(coefs_df)} rows)")
print(f"[SAVED] {OUT_AUCBOOT}  ({len(aucboot_df)} rows)")


# ============================================================
# FIGURE GENERATION
# ============================================================


def pick_main_run(preds_df):
    rid = "POOLED|include_treatment=True"
    if (preds_df["run_id"] == rid).any():
        return rid
    return preds_df["run_id"].dropna().unique()[0]


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[FIG SAVED] {path}")


def annotate_cm(ax, cm, title):
    bacc = balanced_accuracy(
        np.repeat([0,0,1,1], [cm[0,0]+cm[0,1], 0, cm[1,0]+cm[1,1], 0])[:0] if False else [0],
        [0]
    )
    tn, fp, fn, tp = cm.ravel()
    sens, spec = sens_spec(cm)
    bacc = np.nanmean([sens, spec])


    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_xticklabels(["0","1"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["0","1"])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center", fontweight="bold")
    ax.text(0.02, -0.18, f"Balanced Acc={bacc:.3f} | Sens={sens:.3f} | Spec={spec:.3f}", transform=ax.transAxes)


def roc_with_ci(ax, y_true, y_prob, label, n_boot=2000):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    grid, mean_tpr, lo, hi = bootstrap_roc_ci(y_true, y_prob, n_boot=n_boot)
    if np.isfinite(lo).any() and np.isfinite(hi).any():
        ax.fill_between(grid, lo, hi, alpha=0.2)


def calibration_plot(ax, y_true, y_prob, label, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", label=label)


def residual_groups_plot(ax, df, thr_ref):
    d = df.copy()
    d["SEX"] = d["SEX"].astype(str).str.upper()
    d["resp"] = np.where(d["y_true_responder"].astype(int)==1, "Resp", "Non-resp")


    groups = [("M","Resp"), ("M","Non-resp"), ("F","Resp"), ("F","Non-resp")]
    data, labels = [], []
    for sx, rp in groups:
        x = d[(d["SEX"]==sx) & (d["resp"]==rp)]["resid"].dropna().values
        data.append(x)
        labels.append(f"{'Male' if sx=='M' else 'Female'} — {rp}")


    pos = np.arange(1, len(groups)+1)
    try:
        ax.violinplot(data, positions=pos, showextrema=False)
        ax.boxplot(data, positions=pos, widths=0.18, showfliers=False)
    except Exception:
        ax.boxplot(data, positions=pos, widths=0.55, showfliers=False)


    ax.axhline(thr_ref, linestyle="--")
    ax.text(0.01, 0.98, "Threshold = fold-wise SAL μ + 1σ\n(plotted as pooled reference)", transform=ax.transAxes, va="top")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Residual (Observed − Expected)")
    ax.set_title("Residual Distributions - CV (OOF)")


def responder_rate_bar(df, title, fname):
    d = df.dropna(subset=["SEX","TREATMENT","y_true_responder"]).copy()
    d["SEX"] = d["SEX"].astype(str).str.upper()
    d["TREATMENT"] = d["TREATMENT"].astype(str).str.upper()
    g = d.groupby(["TREATMENT","SEX"])["y_true_responder"].mean().reset_index()
    if len(g) == 0:
        return


    trts = sorted(g["TREATMENT"].unique())
    sexes = sorted(g["SEX"].unique())
    x = np.arange(len(trts))
    width = 0.35 if len(sexes) <= 2 else 0.25


    fig, ax = plt.subplots(figsize=(7.5,5.5))
    for i, sx in enumerate(sexes):
        vals = []
        for t in trts:
            v = g[(g["TREATMENT"]==t) & (g["SEX"]==sx)]["y_true_responder"]
            vals.append(float(v.iloc[0]) if len(v) else np.nan)
        ax.bar(x + (i - (len(sexes)-1)/2)*width, vals, width=width, label=sx)


    ax.set_xticks(x)
    ax.set_xticklabels(trts)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Responder rate (mean Y)")
    ax.set_title(title)
    ax.legend(title="SEX", loc="best")
    savefig(fig, fname)
    plt.close(fig)




main_run_id = pick_main_run(preds_df)
sub = preds_df[preds_df["run_id"] == main_run_id].copy()


cv_df  = sub[sub["split"]=="cv_oof"].dropna(subset=["y_true_responder","y_prob_responder"])
ext_df = sub[sub["split"]=="external"].dropna(subset=["y_true_responder","y_prob_responder"])


thr_ref = float(cv_df.groupby("fold")["thr"].first().mean()) if len(cv_df) and "fold" in cv_df.columns else np.nan


fig = plt.figure(figsize=(12,6))
gs = fig.add_gridspec(2, 3, height_ratios=[1,1.2], width_ratios=[1.4,1.0,1.0])


axA = fig.add_subplot(gs[0, 0])
axB1 = fig.add_subplot(gs[0, 1])
axB2 = fig.add_subplot(gs[0, 2])
axC = fig.add_subplot(gs[1, :])


# ROC
roc_with_ci(axA, cv_df["y_true_responder"].astype(int).values, cv_df["y_prob_responder"].values, "CV_OOF", n_boot=2000)
roc_with_ci(axA, ext_df["y_true_responder"].astype(int).values, ext_df["y_prob_responder"].values, "EXTERNAL", n_boot=2000)
axA.plot([0,1],[0,1], linestyle="--")
axA.set_xlabel("False Positive Rate")
axA.set_ylabel("True Positive Rate")
axA.set_title("ROC - Pooled")
axA.legend(loc="lower right")
axA.text(-0.12, 1.05, "A", transform=axA.transAxes, fontweight="bold")


# Confusion matrices
if len(cv_df) and cv_df["y_true_responder"].nunique() > 1:
    y_true = cv_df["y_true_responder"].astype(int).values
    y_pred = (cv_df["y_prob_responder"].values >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    annotate_cm(axB1, cm, "Confusion Matrix (CV OOF)")
else:
    axB1.set_axis_off()
    axB1.set_title("Confusion Matrix (CV OOF)\n(n/a)")


if len(ext_df) and ext_df["y_true_responder"].nunique() > 1:
    y_true = ext_df["y_true_responder"].astype(int).values
    y_pred = (ext_df["y_prob_responder"].values >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    annotate_cm(axB2, cm, "Confusion Matrix (External)")
else:
    axB2.set_axis_off()
    axB2.set_title("Confusion Matrix (External)\n(n/a)")


axB1.text(-0.25, 1.05, "B", transform=axB1.transAxes, fontweight="bold")


# Residual distributions
if len(cv_df):
    residual_groups_plot(axC, cv_df, thr_ref)
else:
    axC.set_axis_off()
    axC.set_title("Residual Distributions - CV (OOF)\n(n/a)")
axC.text(-0.02, 1.05, "C", transform=axC.transAxes, fontweight="bold")


fig.suptitle("Residual ML Framework", y=1.02)
savefig(fig, "core_AUC_CM_residuals.png")
plt.close(fig)


# Predicted vs expected scatter
if len(cv_df.dropna(subset=["expected","y_value"])) > 3:
    d = cv_df.dropna(subset=["expected","y_value","y_true_responder"]).copy()
    fig, ax = plt.subplots(figsize=(6.5,5.5))
    for ylab, lab in [(0, "Non-responder"), (1, "Responder")]:
        dd = d[d["y_true_responder"].astype(int) == ylab]
        ax.scatter(dd["expected"].values, dd["y_value"].values, label=lab, alpha=0.85)
    lo = np.nanmin(np.r_[d["expected"].values, d["y_value"].values])
    hi = np.nanmax(np.r_[d["expected"].values, d["y_value"].values])
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Expected Week10 (Stage 1 saline Ridge)")
    ax.set_ylabel("Observed Week10")
    ax.set_title("Stage 1: Observed vs Expected (Phase 1 CV OOF)")
    ax.legend(loc="best")
    savefig(fig, "support_stage1_observed_vs_expected_scatter_cv.png")
    plt.close(fig)




# Bootstrap AUC distribution
boot_sub = aucboot_df[aucboot_df["run_id"] == main_run_id].copy()
if len(boot_sub):
    fig, ax = plt.subplots(figsize=(7,5.5))
    for split in ["cv_oof","external"]:
        dd = boot_sub[boot_sub["split"] == split]["auc_boot"].dropna().values
        if len(dd):
            ax.hist(dd, bins=30, alpha=0.6, density=True, label=split)
    ax.set_xlabel("Bootstrapped AUC")
    ax.set_ylabel("Density")
    ax.set_title("Bootstrap AUC Distributions")
    ax.legend(loc="best")
    savefig(fig, "support_bootstrap_auc_distributions.png")
    plt.close(fig)


print(f"\nAll figures saved under: {FIG_DIR}")
