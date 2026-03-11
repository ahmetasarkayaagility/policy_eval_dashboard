from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist
from statsmodels.stats.multitest import multipletests


def z_value(confidence_level: float) -> float:
    alpha = 1.0 - confidence_level
    return norm.ppf(1.0 - alpha / 2.0)


def wilson_interval(successes: int, trials: int, confidence_level: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan

    p_hat = successes / trials
    z = z_value(confidence_level)
    z2 = z**2

    denominator = 1 + z2 / trials
    center = (p_hat + z2 / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt((p_hat * (1 - p_hat) / trials) + (z2 / (4 * trials**2)))
        / denominator
    )

    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return lower, upper


def two_proportion_p_value(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    if min(trials_a, trials_b) <= 0:
        return math.nan, math.nan

    p_a = successes_a / trials_a
    p_b = successes_b / trials_b
    pooled = (successes_a + successes_b) / (trials_a + trials_b)
    se = math.sqrt(max(1e-15, pooled * (1 - pooled) * (1 / trials_a + 1 / trials_b)))
    z_stat = (p_b - p_a) / se

    if alternative == "greater":
        p_value = 1 - norm.cdf(z_stat)
    elif alternative == "less":
        p_value = norm.cdf(z_stat)
    else:
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    return z_stat, p_value


def delta_ci_newcombe_wilson(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    low_a, high_a = wilson_interval(successes_a, trials_a, confidence_level)
    low_b, high_b = wilson_interval(successes_b, trials_b, confidence_level)
    return low_b - high_a, high_b - low_a


def prepare_policy_metrics(df: pd.DataFrame, confidence_level: float) -> pd.DataFrame:
    out = df.copy()
    out["successes"] = pd.to_numeric(out["successes"], errors="coerce")
    out["trials"] = pd.to_numeric(out["trials"], errors="coerce")
    out = out.dropna(subset=["model_name", "successes", "trials"]).copy()
    out["successes"] = out["successes"].astype(int)
    out["trials"] = out["trials"].astype(int)
    out = out[out["trials"] > 0].copy()
    out["successes"] = out[["successes", "trials"]].min(axis=1)
    out["successes"] = out["successes"].clip(lower=0)

    out["success_rate"] = out["successes"] / out["trials"]
    intervals = out.apply(
        lambda row: wilson_interval(int(row["successes"]), int(row["trials"]), confidence_level),
        axis=1,
    )
    out["wilson_low"] = intervals.apply(lambda x: x[0])
    out["wilson_high"] = intervals.apply(lambda x: x[1])
    return out.reset_index(drop=True)


def pairwise_adjusted_tests(df: pd.DataFrame, method: str = "holm") -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    if len(df) < 2:
        return pd.DataFrame(columns=["policy_a", "policy_b", "p_value", "p_value_adj"])

    for i, j in itertools.combinations(range(len(df)), 2):
        row_a = df.iloc[i]
        row_b = df.iloc[j]
        _, p_value = two_proportion_p_value(
            int(row_a["successes"]),
            int(row_a["trials"]),
            int(row_b["successes"]),
            int(row_b["trials"]),
            alternative="two-sided",
        )
        records.append(
            {
                "i": i,
                "j": j,
                "policy_a": str(row_a["model_name"]),
                "policy_b": str(row_b["model_name"]),
                "p_value": p_value,
            }
        )

    pvals = [r["p_value"] for r in records]
    _, pvals_adj, _, _ = multipletests(pvals, method=method)

    for rec, p_adj in zip(records, pvals_adj):
        rec["p_value_adj"] = float(p_adj)

    return pd.DataFrame(records)


def _letter_name(index: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    result = ""
    current = index
    while True:
        result = alphabet[current % 26] + result
        current = current // 26 - 1
        if current < 0:
            break
    return result


def compact_letter_display(df: pd.DataFrame, alpha: float = 0.05) -> tuple[dict[str, str], pd.DataFrame]:
    if len(df) == 0:
        return {}, pd.DataFrame()

    ordered = df.sort_values("success_rate", ascending=False).reset_index(drop=True)
    labels = ordered["model_name"].astype(str).tolist()

    tests = pairwise_adjusted_tests(ordered, method="holm")
    n = len(ordered)
    non_significant = np.eye(n, dtype=bool)

    for _, row in tests.iterrows():
        i, j = int(row["i"]), int(row["j"])
        is_non_sig = float(row["p_value_adj"]) >= alpha
        non_significant[i, j] = is_non_sig
        non_significant[j, i] = is_non_sig

    groups: list[set[int]] = []
    for idx in range(n):
        assigned = False
        for group in groups:
            if all(non_significant[idx, member] for member in group):
                group.add(idx)
                assigned = True
        if not assigned:
            groups.append({idx})

    for i in range(n):
        for j in range(i + 1, n):
            if not non_significant[i, j]:
                continue
            if any(i in group and j in group for group in groups):
                continue
            new_group = {i, j}
            for k in range(n):
                if k in new_group:
                    continue
                if all(non_significant[k, m] for m in new_group):
                    new_group.add(k)
            groups.append(new_group)

    pruned_groups: list[set[int]] = []
    for group in groups:
        if not any(group < other for other in groups):
            pruned_groups.append(group)

    letters = {label: "" for label in labels}
    for idx, group in enumerate(pruned_groups):
        letter = _letter_name(idx)
        for member in sorted(group):
            letters[labels[member]] += letter

    return letters, tests


def base_vs_policy_letter_pairs(
    df: pd.DataFrame,
    base_policy: str,
    alpha: float = 0.05,
    p_adjust_method: str | None = "holm",
) -> pd.DataFrame:
    required_columns = {"model_name", "successes", "trials"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for base-vs comparison: {sorted(missing)}")

    working = df.dropna(subset=["model_name", "successes", "trials"]).copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                "base_policy",
                "policy",
                "delta",
                "p_value",
                "p_value_adj",
                "is_significant",
                "pair_letters",
            ]
        )

    labels = working["model_name"].astype(str).tolist()
    if base_policy not in labels:
        raise ValueError(f"Base policy '{base_policy}' not found in selected policies.")

    base_row = working.loc[working["model_name"].astype(str) == base_policy].iloc[0]
    base_successes = int(base_row["successes"])
    base_trials = int(base_row["trials"])
    base_rate = base_successes / base_trials if base_trials > 0 else 0.0

    raw_records: list[dict[str, float | str]] = []
    raw_p_values: list[float] = []

    for _, row in working.iterrows():
        policy_name = str(row["model_name"])
        if policy_name == base_policy:
            continue

        policy_successes = int(row["successes"])
        policy_trials = int(row["trials"])
        policy_rate = policy_successes / policy_trials if policy_trials > 0 else 0.0

        _, p_value = two_proportion_p_value(
            base_successes,
            base_trials,
            policy_successes,
            policy_trials,
            alternative="two-sided",
        )
        raw_p_values.append(float(p_value))
        raw_records.append(
            {
                "base_policy": base_policy,
                "policy": policy_name,
                "delta": policy_rate - base_rate,
                "p_value": float(p_value),
            }
        )

    if not raw_records:
        return pd.DataFrame(
            columns=[
                "base_policy",
                "policy",
                "delta",
                "p_value",
                "p_value_adj",
                "is_significant",
                "pair_letters",
            ]
        )

    if p_adjust_method is None or str(p_adjust_method).lower() in {"none", "raw"}:
        reject = [p < alpha for p in raw_p_values]
        p_value_adj = raw_p_values
    else:
        reject, p_value_adj, _, _ = multipletests(raw_p_values, alpha=alpha, method=p_adjust_method)

    rows: list[dict[str, float | str | bool]] = []
    for rec, is_reject, p_adj in zip(raw_records, reject, p_value_adj):
        significant = bool(is_reject)
        rows.append(
            {
                **rec,
                "p_value_adj": float(p_adj),
                "is_significant": significant,
                "pair_letters": "a-b" if significant else "a-a",
            }
        )

    return pd.DataFrame(rows)


def quality_score_ci(
    mean_pct: float,
    std_pct: float,
    n: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Confidence interval for a quality-score mean via the *t*-distribution.

    All values are in percentage-point scale ([0, 100]).
    Returns ``(ci_low, ci_high)``, clamped to [0, 100].
    """
    if n <= 1 or not math.isfinite(mean_pct) or not math.isfinite(std_pct) or std_pct < 0:
        return math.nan, math.nan

    t_crit = float(t_dist.ppf(1.0 - (1.0 - confidence_level) / 2.0, df=n - 1))
    se = std_pct / math.sqrt(n)
    return max(0.0, mean_pct - t_crit * se), min(100.0, mean_pct + t_crit * se)


def welch_t_test(
    mean_a: float,
    std_a: float,
    n_a: int,
    mean_b: float,
    std_b: float,
    n_b: int,
    confidence_level: float = 0.95,
) -> tuple[float, float, float, float, float]:
    """Welch's *t*-test for two independent sample means.

    Returns ``(t_stat, p_value, dof, delta_ci_low, delta_ci_high)``
    where *delta = mean_b - mean_a*.
    """
    if min(n_a, n_b) <= 1:
        return math.nan, math.nan, math.nan, math.nan, math.nan

    for val in (mean_a, std_a, mean_b, std_b):
        if not math.isfinite(val):
            return math.nan, math.nan, math.nan, math.nan, math.nan

    se_a2 = (std_a ** 2) / n_a
    se_b2 = (std_b ** 2) / n_b
    se_diff = math.sqrt(se_a2 + se_b2)
    if se_diff < 1e-15:
        return math.nan, math.nan, math.nan, math.nan, math.nan

    delta = mean_b - mean_a
    t_stat = delta / se_diff

    # Welch-Satterthwaite degrees of freedom
    numerator = (se_a2 + se_b2) ** 2
    denominator = se_a2 ** 2 / (n_a - 1) + se_b2 ** 2 / (n_b - 1)
    if denominator < 1e-15:
        return math.nan, math.nan, math.nan, math.nan, math.nan
    dof = numerator / denominator

    p_value = float(2.0 * (1.0 - t_dist.cdf(abs(t_stat), df=dof)))

    t_crit = float(t_dist.ppf(1.0 - (1.0 - confidence_level) / 2.0, df=dof))
    delta_ci_low = delta - t_crit * se_diff
    delta_ci_high = delta + t_crit * se_diff

    return t_stat, p_value, dof, delta_ci_low, delta_ci_high
