from __future__ import annotations

import hashlib
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
from dash import dash_table

from data_utils import (
    CSV_SINGLE_SHEET_NAME,
    DEFAULT_GOOGLE_SHEET_URL,
    DEFAULT_TRIALS,
    extract_url_from_cell_value,
    get_google_auth_status,
    list_google_spreadsheet_sheets,
    list_local_spreadsheet_sheets,
    load_google_spreadsheet,
    load_local_spreadsheet,
    normalize_policy_dataframe,
    promote_header_row_if_needed,
)
from stats_utils import (
    base_vs_policy_letter_pairs,
    compact_letter_display,
    delta_ci_newcombe_wilson,
    prepare_policy_metrics,
    quality_score_ci,
    two_proportion_p_value,
    welch_t_test,
    wilson_interval,
)


CONFIDENCE_OPTIONS = [
    {"label": "90% (exploratory / low risk)", "value": 0.90},
    {"label": "95% (default)", "value": 0.95},
    {"label": "99% (very strict)", "value": 0.99},
]

SORT_MODE_LABELS = {
    "original": "Original sheet order",
    "success_rate": "Sorted by success rate \u2193",
    "success_rate_asc": "Sorted by success rate \u2191",
    "quality_score": "Sorted by Quality Score [%] \u2193",
    "quality_score_asc": "Sorted by Quality Score [%] \u2191",
    "dropin_ratio": "Sorted by Attempt Drop-in [%] \u2193",
    "dropin_ratio_asc": "Sorted by Attempt Drop-in [%] \u2191",
}

QUALITY_SCORE_COLUMN_CANDIDATES = ["Quality Score [%]", "Quality Score", "QualityScore", "Quality"]
QUALITY_SCORE_STD_COLUMN_CANDIDATES = [
    "Quality Score STD [%]",
    "Quality Score STD",
    "QualityScoreSTD",
    "Quality STD",
]
SUCCESS_RATE_INPUT_CANDIDATES = ["Success Rate [%]", "Success Rate", "Success_Rate", "Rate", "Accuracy"]
DROPIN_RATIO_COLUMN_CANDIDATES = [
    "Attempt to drop in Ratio [%]",
    "Attempt to drop in Ratio",
    "Attempt Drop-in Ratio [%]",
    "Attempt Drop-in Ratio",
    "Drop-in Ratio [%]",
    "Drop-in Ratio",
    "DropIn Ratio",
    "Drop in Ratio",
]
TESTING_GROUP_COLUMN_CANDIDATES = [
    "Testing Group",
    "TestingGroup",
    "Test Group",
    "Experiment Group",
    "Group",
    "Tag",
]
BASE_GROUP_TAGS = {
    "base",
    "baseline",
    "default",
    "control",
}
POLICY_COLOR_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

FAILURE_METRIC_OPTIONS = [
    {"label": "Failure rate (%)", "value": "failure_rate"},
    {"label": "Success rate (%)", "value": "success_rate"},
    {"label": "Quality score mean (%)", "value": "quality_score"},
    {"label": "Sample count (n)", "value": "sample_count"},
]

FAILURE_TASK_SUCCESS_COLUMN_CANDIDATES = [
    "Task Success",
    "TaskSuccess",
    "Success",
    "Succeeded",
    "Outcome",
    "Result",
]
FAILURE_QUALITY_COLUMN_CANDIDATES = [
    "Score based on rubric",
    "Score Based On Rubric",
    "Rubric Score",
    "Quality Score",
    "Quality",
    "Score",
]
FAILURE_DEFAULT_X_COLUMN_CANDIDATES = [
    "Relative Stance Offset",
    "Robot Initial Pose",
    "Robot Pose",
    "Stance Offset",
]
FAILURE_DEFAULT_Y_COLUMN_CANDIDATES = [
    "Tote on Pallet Offset",
    "Stack Pose",
    "Initial Stack Pose",
    "Pallet Offset",
]
FAILURE_LINK_COLUMN_CANDIDATES = [
    "eval_details_url",
    "Eval Details URL",
    "Evaluation Details URL",
    "Detail URL",
    "Details URL",
    "Rollout Details URL",
    "Eval Details",
    "Evaluation Details",
    "Rollout Details",
    "Details",
]


def _default_columns() -> list[dict[str, str | bool]]:
    return [
        {"name": "model_name", "id": "model_name", "editable": True},
        {"name": "successes", "id": "successes", "editable": True},
        {"name": "trials", "id": "trials", "editable": True},
        {"name": "notes", "id": "notes", "editable": True},
    ]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(col).lower()): str(col)
        for col in df.columns
    }
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    return None


def _percent_like_to_numeric(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    text = text.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    return pd.to_numeric(text, errors="coerce")


def _to_percent_points(series: pd.Series) -> pd.Series:
    numeric = _percent_like_to_numeric(series)
    unit_interval_mask = numeric.notna() & (numeric >= 0) & (numeric <= 1)
    return numeric.where(~unit_interval_mask, numeric * 100.0)


def _normalize_group_tag(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _is_base_group_tag(value: object) -> bool:
    token = _normalize_group_tag(value)
    return bool(token) and token in BASE_GROUP_TAGS


def _first_non_null(series: pd.Series) -> object:
    valid = series.dropna()
    if valid.empty:
        return pd.NA
    return valid.iloc[0]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    clean = hex_color.lstrip("#")
    if len(clean) != 6:
        return f"rgba(31,119,180,{alpha})"
    red = int(clean[0:2], 16)
    green = int(clean[2:4], 16)
    blue = int(clean[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"


def _policy_color(policy_name: str) -> str:
    digest = hashlib.md5(policy_name.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(POLICY_COLOR_PALETTE)
    return POLICY_COLOR_PALETTE[index]


def _policy_color_map(policy_names: list[str]) -> dict[str, str]:
    unique_names = sorted({str(name) for name in policy_names})
    return {name: _policy_color(name) for name in unique_names}


def _raw_to_clean_df(raw_records: list[dict] | None) -> pd.DataFrame:
    if not raw_records:
        return pd.DataFrame(
            columns=[
                "model_name",
                "successes",
                "trials",
                "source_order",
                "quality_score_pct",
                "quality_score_std_pct",
                "dropin_ratio_pct",
                "dropin_count",
                "has_success_rate_input",
                "testing_group",
                "is_base_group",
            ]
        )

    df = pd.DataFrame(raw_records)
    for required in ["model_name", "successes", "trials"]:
        if required not in df.columns:
            df[required] = pd.NA

    df["source_order"] = np.arange(len(df))

    df["model_name"] = df["model_name"].astype(str).str.strip()
    df["model_name"] = df["model_name"].replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    df["successes"] = pd.to_numeric(df["successes"], errors="coerce")
    df["trials"] = pd.to_numeric(df["trials"], errors="coerce")

    df = df.dropna(subset=["model_name", "successes", "trials"]).copy()
    df["successes"] = df["successes"].astype(int)
    df["trials"] = df["trials"].astype(int)
    df = df[df["trials"] > 0].copy()
    df["successes"] = df[["successes", "trials"]].min(axis=1).clip(lower=0)

    quality_col = _find_column(df, QUALITY_SCORE_COLUMN_CANDIDATES)
    quality_std_col = _find_column(df, QUALITY_SCORE_STD_COLUMN_CANDIDATES)
    if quality_col is not None:
        mean_raw = _percent_like_to_numeric(df[quality_col])
        std_raw = _percent_like_to_numeric(df[quality_std_col]) if quality_std_col else pd.Series(pd.NA, index=df.index)
        unit_mask = mean_raw.notna() & (mean_raw >= 0) & (mean_raw <= 1)
        df["quality_score_pct"] = mean_raw.where(~unit_mask, mean_raw * 100.0)
        df["quality_score_std_pct"] = std_raw.where(~unit_mask, std_raw * 100.0)
    else:
        df["quality_score_pct"] = pd.NA
        df["quality_score_std_pct"] = pd.NA

    dropin_col = _find_column(df, DROPIN_RATIO_COLUMN_CANDIDATES)
    if dropin_col is not None:
        dropin_raw = _percent_like_to_numeric(df[dropin_col])
        unit_mask_di = dropin_raw.notna() & (dropin_raw >= 0) & (dropin_raw <= 1)
        df["dropin_ratio_pct"] = dropin_raw.where(~unit_mask_di, dropin_raw * 100.0)
        # Back-compute dropin count from percentage and trials for Wilson CI
        df["dropin_count"] = (df["dropin_ratio_pct"] / 100.0 * df["trials"]).round().fillna(0).astype(int)
        df["dropin_count"] = df["dropin_count"].clip(lower=0)
        df["dropin_count"] = df[["dropin_count", "trials"]].min(axis=1)
    else:
        df["dropin_ratio_pct"] = pd.NA
        df["dropin_count"] = pd.NA

    success_rate_input_col = _find_column(df, SUCCESS_RATE_INPUT_CANDIDATES)
    if success_rate_input_col is not None:
        df["has_success_rate_input"] = _percent_like_to_numeric(df[success_rate_input_col]).notna()
    else:
        df["has_success_rate_input"] = True

    testing_group_col = _find_column(df, TESTING_GROUP_COLUMN_CANDIDATES)
    if testing_group_col is not None:
        df["testing_group"] = df[testing_group_col].astype(str).str.strip()
        df["testing_group"] = df["testing_group"].replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    else:
        df["testing_group"] = pd.NA
    df["is_base_group"] = df["testing_group"].map(_is_base_group_tag)

    grouped = (
        df.groupby("model_name", as_index=False)
        .agg(
            {
                "successes": "sum",
                "trials": "sum",
                "source_order": "min",
                "quality_score_pct": "mean",
                "quality_score_std_pct": "first",
                "dropin_ratio_pct": "mean",
                "dropin_count": "sum",
                "has_success_rate_input": "any",
                "testing_group": _first_non_null,
                "is_base_group": "any",
            }
        )
        .sort_values("source_order", kind="stable")
        .reset_index(drop=True)
    )
    return grouped


def _apply_sort_mode(
    metrics: pd.DataFrame,
    sort_mode: str | None,
    pin_first: str | None = None,
) -> pd.DataFrame:
    """Sort *metrics* and optionally pin *pin_first* policy to the front."""
    mode = sort_mode or "original"
    if mode == "success_rate":
        result = metrics.sort_values(["success_rate", "source_order"], ascending=[False, True], kind="stable").reset_index(drop=True)
    elif mode == "success_rate_asc":
        result = metrics.sort_values(["success_rate", "source_order"], ascending=[True, True], kind="stable").reset_index(drop=True)
    elif mode == "quality_score" and "quality_score_pct" in metrics.columns:
        result = metrics.sort_values(["quality_score_pct", "source_order"], ascending=[False, True], na_position="last", kind="stable").reset_index(drop=True)
    elif mode == "quality_score_asc" and "quality_score_pct" in metrics.columns:
        result = metrics.sort_values(["quality_score_pct", "source_order"], ascending=[True, True], na_position="first", kind="stable").reset_index(drop=True)
    elif mode == "dropin_ratio" and "dropin_ratio_pct" in metrics.columns:
        result = metrics.sort_values(["dropin_ratio_pct", "source_order"], ascending=[False, True], na_position="last", kind="stable").reset_index(drop=True)
    elif mode == "dropin_ratio_asc" and "dropin_ratio_pct" in metrics.columns:
        result = metrics.sort_values(["dropin_ratio_pct", "source_order"], ascending=[True, True], na_position="first", kind="stable").reset_index(drop=True)
    elif "source_order" in metrics.columns:
        result = metrics.sort_values("source_order", ascending=True, kind="stable").reset_index(drop=True)
    else:
        result = metrics.reset_index(drop=True)

    if pin_first and pin_first in set(result["model_name"].astype(str)):
        mask = result["model_name"].astype(str) == pin_first
        result = pd.concat([result[mask], result[~mask]], ignore_index=True)

    return result


def _common_prefix_at_boundary(names: list[str]) -> str:
    """Longest common prefix truncated to the last ``_`` or ``-`` boundary."""
    if len(names) < 2:
        return ""
    prefix = names[0]
    for name in names[1:]:
        i = 0
        while i < len(prefix) and i < len(name) and prefix[i] == name[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            return ""
    last_sep = -1
    for i, ch in enumerate(prefix):
        if ch in ("_", "-"):
            last_sep = i
    if last_sep < 0:
        return ""
    candidate = prefix[: last_sep + 1]
    if all(len(n) > len(candidate) for n in names):
        return candidate
    return ""


def _make_display_names(names: list[str]) -> tuple[dict[str, str], str]:
    """Return ``(full_name -> short_name, stripped_prefix)``."""
    prefix = _common_prefix_at_boundary(names)
    if not prefix:
        return {n: n for n in names}, ""
    return {n: n[len(prefix):] for n in names}, prefix


def _format_sort_status(sort_mode: str | None, prefix: str = "", active_group: str | None = None) -> str:
    status = f"Order mode: {SORT_MODE_LABELS.get(sort_mode or 'original', 'Original sheet order')}"
    if active_group:
        status += f" | testing group: {active_group} + Base"
    if prefix:
        status += f" | common prefix: {prefix}"
    return status


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Policy",
        yaxis_title="Success Rate (%)",
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
            }
        ],
    )
    return fig


def _build_posterior_violin(
    plot_df: pd.DataFrame,
    letters: dict[str, str],
    policy_colors: dict[str, str],
    title: str,
    display_names: dict[str, str] | None = None,
) -> go.Figure:
    if plot_df.empty:
        return _empty_figure("No selected policies for posterior uncertainty view")

    _dn = display_names or {}
    rng = np.random.default_rng(20260310)
    n_samples = 1200
    prior_alpha = 1.0
    prior_beta = 1.0

    fig = go.Figure()
    for _, row in plot_df.reset_index(drop=True).iterrows():
        model_name = str(row["model_name"])
        short = _dn.get(model_name, model_name)
        color = policy_colors.get(model_name, "#1f77b4")

        successes = int(row["successes"])
        trials = int(row["trials"])
        failures = max(0, trials - successes)

        posterior_samples = rng.beta(prior_alpha + successes, prior_beta + failures, size=n_samples) * 100.0

        fig.add_trace(
            go.Violin(
                x=[short] * n_samples,
                y=posterior_samples,
                legendgroup=model_name,
                scalegroup=model_name,
                name=short,
                showlegend=False,
                points=False,
                meanline_visible=True,
                line_color=_hex_to_rgba(color, 0.95),
                fillcolor=_hex_to_rgba(color, 0.42),
            )
        )

    annotations = []
    for model_name in plot_df["model_name"].astype(str).tolist():
        short = _dn.get(model_name, model_name)
        group_letters = letters.get(model_name, "")
        if not group_letters:
            continue
        annotations.append(
            {
                "x": short,
                "y": 103,
                "text": f"<b>{group_letters}</b>",
                "showarrow": False,
                "font": {"size": 16},
            }
        )

    fig.update_layout(
        template="plotly_white",
        violinmode="group",
        yaxis_title="Success Rate (%)",
        xaxis_title="Policy",
        title=title,
        yaxis_range=[0, 105],
        annotations=annotations,
    )
    return fig


def _build_base_vs_pairs_violin(
    plot_df: pd.DataFrame,
    base_policy: str,
    pair_letters_df: pd.DataFrame,
    policy_colors: dict[str, str],
    display_names: dict[str, str] | None = None,
    title_suffix: str = "",
) -> go.Figure:
    if plot_df.empty:
        return _empty_figure("No selected policies for base-vs-policy violin")

    if base_policy not in set(plot_df["model_name"].astype(str)):
        return _empty_figure("Base policy is not in selected policies")

    if pair_letters_df.empty:
        return _empty_figure("Select at least one non-base policy for pairwise violin")

    _dn = display_names or {}
    rng = np.random.default_rng(20260310)
    n_samples = 1000
    prior_alpha = 1.0
    prior_beta = 1.0

    row_by_model = plot_df.set_index("model_name")
    fig = go.Figure()
    shown_legend: set[str] = set()
    annotations = []

    for idx, row in pair_letters_df.reset_index(drop=True).iterrows():
        other_policy = str(row["policy"])
        base_short = _dn.get(base_policy, base_policy)
        other_short = _dn.get(other_policy, other_policy)
        pair_label = f"{base_short} vs {other_short}"

        for policy_name in [base_policy, other_policy]:
            if policy_name not in row_by_model.index:
                continue
            record = row_by_model.loc[policy_name]
            successes = int(record["successes"])
            trials = int(record["trials"])
            failures = max(0, trials - successes)
            posterior_samples = rng.beta(prior_alpha + successes, prior_beta + failures, size=n_samples) * 100.0

            short = _dn.get(policy_name, policy_name)
            color = policy_colors.get(policy_name, "#1f77b4")
            show_legend = policy_name not in shown_legend
            shown_legend.add(policy_name)

            fig.add_trace(
                go.Violin(
                    x=[pair_label] * n_samples,
                    y=posterior_samples,
                    legendgroup=policy_name,
                    scalegroup=f"{pair_label}:{policy_name}",
                    name=short,
                    showlegend=show_legend,
                    points=False,
                    meanline_visible=True,
                    line_color=_hex_to_rgba(color, 0.95),
                    fillcolor=_hex_to_rgba(color, 0.42),
                )
            )

        annotations.append(
            {
                "x": pair_label,
                "y": 103,
                "text": f"<b>{row['pair_letters']}</b>",
                "showarrow": False,
                "font": {"size": 15},
            }
        )

    fig.update_layout(
        template="plotly_white",
        violinmode="group",
        yaxis_title="Success Rate (%)",
        xaxis_title="Base-vs-policy pair",
        title=f"Base-vs-policy posterior uncertainty with pair letters{title_suffix}",
        yaxis_range=[0, 105],
        annotations=annotations,
    )
    return fig


def _format_numeric_token(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return str(int(rounded))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _normalize_header_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _make_unique_columns(columns: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique: list[str] = []

    for idx, column in enumerate(columns):
        base = str(column).strip()
        if not base:
            base = f"col_{idx + 1}"
        counter = counts.get(base, 0) + 1
        counts[base] = counter
        unique.append(base if counter == 1 else f"{base}_{counter}")

    return unique


def _find_detail_header_row(detail_df: pd.DataFrame, max_scan_rows: int = 40) -> int | None:
    if detail_df.empty:
        return None

    success_tokens = {_normalize_header_token(value) for value in FAILURE_TASK_SUCCESS_COLUMN_CANDIDATES}
    quality_tokens = {_normalize_header_token(value) for value in FAILURE_QUALITY_COLUMN_CANDIDATES}
    axis_tokens = {
        _normalize_header_token(value)
        for value in [
            *FAILURE_DEFAULT_X_COLUMN_CANDIDATES,
            *FAILURE_DEFAULT_Y_COLUMN_CANDIDATES,
            "Task",
            "# totes in stack",
            "Totes in stack",
        ]
    }

    best_row: int | None = None
    best_rank: tuple[int, int, int, int] = (-1, -1, -1, -1)

    scan_rows = min(max_scan_rows, len(detail_df))
    for row_index in range(scan_rows):
        tokens = [_normalize_header_token(value) for value in detail_df.iloc[row_index].tolist()]
        tokens = [token for token in tokens if token]
        token_set = set(tokens)

        has_success = bool(token_set & success_tokens)
        if not has_success:
            continue

        axis_hits = len(token_set & axis_tokens)
        quality_hits = 1 if bool(token_set & quality_tokens) else 0
        rank = (1, axis_hits, quality_hits, len(tokens))

        if rank > best_rank:
            best_rank = rank
            best_row = row_index

    return best_row


def _promote_detail_header_row_if_needed(detail_df: pd.DataFrame) -> pd.DataFrame:
    promoted = promote_header_row_if_needed(detail_df)
    if _find_column(promoted, FAILURE_TASK_SUCCESS_COLUMN_CANDIDATES) is not None:
        return promoted

    header_row = _find_detail_header_row(detail_df)
    if header_row is None:
        return promoted

    header_values = detail_df.iloc[header_row].tolist()
    fallback_columns = [str(col) for col in detail_df.columns]

    candidate_columns: list[str] = []
    for idx, value in enumerate(header_values):
        label = "" if pd.isna(value) else str(value).strip()
        token = _normalize_header_token(label)
        if not label or token in {"nan", "none"} or token.startswith("unnamed"):
            fallback = fallback_columns[idx] if idx < len(fallback_columns) else ""
            fallback = str(fallback).strip()
            fallback_token = _normalize_header_token(fallback)
            if fallback and not fallback_token.startswith("unnamed"):
                label = fallback
            else:
                label = f"col_{idx + 1}"
        candidate_columns.append(label)

    promoted_detail = detail_df.iloc[header_row + 1 :].copy()
    if promoted_detail.empty:
        return promoted

    promoted_detail.columns = _make_unique_columns(candidate_columns)
    promoted_detail = promoted_detail.dropna(how="all").reset_index(drop=True)
    return promoted_detail


def _normalize_condition_token(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return "NA"

    bracket_match = re.match(r"^[\[(]\s*(.*?)\s*[\])]$", text)
    if bracket_match:
        raw_parts = [part.strip() for part in bracket_match.group(1).split(",")]
        parsed_parts: list[str] = []
        for part in raw_parts:
            if not part:
                return text
            parsed = pd.to_numeric(pd.Series([part]), errors="coerce").iloc[0]
            if pd.isna(parsed):
                return text
            parsed_parts.append(_format_numeric_token(float(parsed)))
        return "[" + ", ".join(parsed_parts) + "]"

    parsed = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(parsed):
        return _format_numeric_token(float(parsed))

    return text


def _condition_sort_key(label: str) -> tuple:
    if label == "NA":
        return 3, ""

    bracket_match = re.match(r"^\[\s*(.*?)\s*\]$", str(label))
    if bracket_match:
        raw_parts = [part.strip() for part in bracket_match.group(1).split(",")]
        numbers: list[float] = []
        for part in raw_parts:
            parsed = pd.to_numeric(pd.Series([part]), errors="coerce").iloc[0]
            if pd.isna(parsed):
                numbers = []
                break
            numbers.append(float(parsed))
        if numbers:
            return 0, len(numbers), tuple(numbers)

    parsed = pd.to_numeric(pd.Series([label]), errors="coerce").iloc[0]
    if pd.notna(parsed):
        return 1, float(parsed)

    return 2, str(label).lower()


def _condition_key(column_name: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", column_name.strip().lower()).strip("_")
    token = token or "condition"
    digest = hashlib.md5(column_name.strip().lower().encode("utf-8")).hexdigest()[:8]
    return f"cond__{token}_{digest}"


def _safe_policy_name_from_row(row: dict[str, object]) -> str:
    for key in ["model_name", "Model Name", "Model", "Policy", "Policy Name"]:
        if key in row:
            text = str(row.get(key, "")).strip()
            if text and text.lower() not in {"nan", "none"}:
                return text
    return ""


def _collect_policy_detail_links(rows: list[dict] | None) -> list[dict[str, str]]:
    collected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for row in rows or []:
        if not isinstance(row, dict):
            continue

        policy_name = _safe_policy_name_from_row(row)
        if not policy_name:
            continue

        detail_url: str | None = None
        for key in FAILURE_LINK_COLUMN_CANDIDATES:
            if key not in row:
                continue
            detail_url = extract_url_from_cell_value(row.get(key))
            if detail_url:
                break

        if detail_url is None:
            for key, value in row.items():
                token = re.sub(r"[^a-z0-9]", "", str(key).lower())
                if "detail" not in token:
                    continue
                parsed = extract_url_from_cell_value(value)
                if parsed:
                    detail_url = parsed
                    break

        if not detail_url:
            continue
        if "/spreadsheets/d/" not in detail_url:
            continue

        dedup_key = (policy_name, detail_url)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        collected.append({"policy_name": policy_name, "detail_url": detail_url})

    return collected


def _parse_task_success_value(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0

    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "success", "succeeded", "pass", "passed"}:
        return 1.0
    if text in {"0", "false", "f", "no", "n", "fail", "failed", "failure"}:
        return 0.0

    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return 1.0 if float(parsed) > 0 else 0.0


def _normalize_detail_rollout_frame(
    detail_df: pd.DataFrame,
    policy_name: str,
    detail_url: str,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    working = _promote_detail_header_row_if_needed(detail_df)
    if working.empty:
        return [], []

    success_col = _find_column(working, FAILURE_TASK_SUCCESS_COLUMN_CANDIDATES)
    if success_col is None:
        raise ValueError("Missing 'Task Success' column in detail sheet")

    quality_col = _find_column(working, FAILURE_QUALITY_COLUMN_CANDIDATES)
    quality_values = (
        _to_percent_points(working[quality_col])
        if quality_col is not None
        else pd.Series(pd.NA, index=working.index)
    )

    condition_columns: list[tuple[str, str, pd.Series]] = []
    for column in working.columns:
        column_name = str(column).strip()
        if not column_name:
            continue
        if column_name == success_col:
            continue
        if quality_col is not None and column_name == quality_col:
            continue

        normalized = working[column].map(_normalize_condition_token)
        non_na = normalized[normalized != "NA"]
        unique_count = int(non_na.nunique(dropna=True))
        if unique_count <= 1:
            continue
        if unique_count > max(80, int(len(working) * 0.95)):
            continue

        key = _condition_key(column_name)
        condition_columns.append((key, column_name, normalized))

    records: list[dict[str, object]] = []
    for idx in range(len(working)):
        success_value = _parse_task_success_value(working.iloc[idx][success_col])
        if success_value is None:
            continue

        record: dict[str, object] = {
            "policy_name": policy_name,
            "detail_url": detail_url,
            "task_success": success_value,
            "quality_score_pct": math.nan,
        }

        quality_value = quality_values.iloc[idx]
        if pd.notna(quality_value):
            record["quality_score_pct"] = float(quality_value)

        for key, _label, series in condition_columns:
            record[key] = series.iloc[idx]

        records.append(record)

    metadata = [{"key": key, "label": label} for key, label, _series in condition_columns]
    return records, metadata


def _pick_default_condition_key(condition_columns: list[dict[str, str]], candidates: list[str]) -> str | None:
    normalized_map = {
        re.sub(r"[^a-z0-9]", "", str(entry.get("label", "")).lower()): str(entry.get("key", ""))
        for entry in condition_columns
        if str(entry.get("key", "")).strip()
    }
    for candidate in candidates:
        key = normalized_map.get(re.sub(r"[^a-z0-9]", "", candidate.lower()))
        if key:
            return key
    if condition_columns:
        return str(condition_columns[0].get("key"))
    return None


def _failure_empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Condition",
        yaxis_title="Condition",
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
            }
        ],
    )
    return fig


def _build_failure_policy_grid_figure(
    cell_df: pd.DataFrame,
    policy_names: list[str],
    x_values: list[str],
    y_values: list[str],
    x_key: str,
    y_key: str,
    metric_key: str,
    metric_label: str,
    colorscale: str,
    zmin: float | None,
    zmax: float | None,
    display_names: dict[str, str] | None = None,
) -> go.Figure:
    if not policy_names:
        return _failure_empty_figure("No policies selected for failure heatmaps")

    display = display_names or {}
    n_cols = 4
    n_rows = int(math.ceil(len(policy_names) / n_cols))
    subplot_titles = [display.get(policy, policy) for policy in policy_names]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.16,
    )

    for index, policy_name in enumerate(policy_names):
        row_idx = index // n_cols + 1
        col_idx = index % n_cols + 1

        policy_cells = cell_df[cell_df["policy_name"].astype(str) == str(policy_name)].copy()
        z = (
            policy_cells.pivot(index=y_key, columns=x_key, values=metric_key)
            .reindex(index=y_values, columns=x_values)
            .to_numpy()
        )
        n = (
            policy_cells.pivot(index=y_key, columns=x_key, values="n")
            .reindex(index=y_values, columns=x_values)
            .fillna(0)
            .to_numpy()
        )

        fig.add_trace(
            go.Heatmap(
                x=x_values,
                y=y_values,
                z=z,
                customdata=n,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                showscale=index == 0,
                colorbar={"title": metric_label} if index == 0 else None,
                hovertemplate=(
                    "%{y} × %{x}<br>"
                    + metric_label
                    + ": %{z:.2f}<br>n: %{customdata:.0f}<extra>"
                    + display.get(policy_name, policy_name)
                    + "</extra>"
                ),
            ),
            row=row_idx,
            col=col_idx,
        )
        fig.update_xaxes(tickangle=35, row=row_idx, col=col_idx)

    fig.update_layout(
        template="plotly_white",
        title=f"Policy failure-mode mini-heatmaps ({metric_label})",
        height=max(420, n_rows * 300),
        margin={"l": 30, "r": 30, "t": 90, "b": 40},
    )
    return fig


def _build_failure_aggregate_figure(
    aggregate_df: pd.DataFrame,
    x_values: list[str],
    y_values: list[str],
    x_key: str,
    y_key: str,
    metric_key: str,
    metric_label: str,
    colorscale: str,
    zmin: float | None,
    zmax: float | None,
) -> go.Figure:
    if aggregate_df.empty:
        return _failure_empty_figure("No aggregate failure data available")

    def _format_cell_label(value: float) -> str:
        if metric_key in {"failure_rate_pct", "success_rate_pct", "quality_score_pct"}:
            return f"{value:.1f}%"
        if metric_key == "n":
            return str(int(round(value)))
        return f"{value:.2f}"

    z = (
        aggregate_df.pivot(index=y_key, columns=x_key, values=metric_key)
        .reindex(index=y_values, columns=x_values)
        .to_numpy()
    )
    n = (
        aggregate_df.pivot(index=y_key, columns=x_key, values="n")
        .reindex(index=y_values, columns=x_values)
        .fillna(0)
        .to_numpy()
    )

    fig = go.Figure(
        data=[
            go.Heatmap(
                x=x_values,
                y=y_values,
                z=z,
                customdata=n,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                colorbar={"title": metric_label},
                hovertemplate=(
                    "%{y} × %{x}<br>"
                    + metric_label
                    + ": %{z:.2f}<br>n: %{customdata:.0f}<extra>All selected policies</extra>"
                ),
            )
        ]
    )

    if zmin is not None and math.isfinite(zmin):
        zmin_eff = float(zmin)
    else:
        zmin_eff = float(np.nanmin(z)) if np.isfinite(np.nanmin(z)) else 0.0
    if zmax is not None and math.isfinite(zmax):
        zmax_eff = float(zmax)
    else:
        zmax_eff = float(np.nanmax(z)) if np.isfinite(np.nanmax(z)) else zmin_eff + 1.0
    if zmax_eff <= zmin_eff:
        zmax_eff = zmin_eff + 1.0

    annotations: list[dict[str, object]] = []
    for y_idx, y_value in enumerate(y_values):
        for x_idx, x_value in enumerate(x_values):
            cell_value = z[y_idx][x_idx]
            if pd.isna(cell_value):
                continue

            try:
                numeric_value = float(cell_value)
            except (TypeError, ValueError):
                continue

            normalized = (numeric_value - zmin_eff) / (zmax_eff - zmin_eff)
            normalized = max(0.0, min(1.0, normalized))
            text_color = "#FFFFFF" if normalized >= 0.58 else "#111111"

            annotations.append(
                {
                    "x": x_value,
                    "y": y_value,
                    "xref": "x",
                    "yref": "y",
                    "text": _format_cell_label(numeric_value),
                    "showarrow": False,
                    "font": {"size": 11, "color": text_color},
                }
            )

    fig.update_layout(
        template="plotly_white",
        title=f"Aggregate condition heatmap ({metric_label})",
        xaxis_title="Robot condition (X)",
        yaxis_title="Stack condition (Y)",
        margin={"l": 30, "r": 30, "t": 70, "b": 50},
        annotations=annotations,
    )
    fig.update_xaxes(tickangle=35)
    return fig


app = Dash(__name__)
app.title = "Robot Policy Evaluation Dashboard"

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px"},
    children=[
        dcc.Store(id="uploaded-file-store", data=None),
        dcc.Store(id="sort-mode-store", data="original"),
        dcc.Store(id="active-testing-group-store", data=None),
        dcc.Store(
            id="failure-detail-store",
            data={"records": [], "condition_columns": [], "default_x": None, "default_y": None},
        ),
        dcc.Tabs(
            id="page-tabs",
            value="main",
            children=[
                dcc.Tab(label="Main Dashboard", value="main"),
                dcc.Tab(label="Failure Mode Analysis", value="failure"),
            ],
        ),
        html.Div(
            id="main-page",
            children=[
                html.H2("Robot Policy Evaluation Dashboard"),
                html.P(
                    "Upload a local CSV/XLSX or paste a Google Sheets link, then edit/log rollout results "
                    "and compare policies with Wilson confidence intervals."
                ),
                html.Hr(),
                html.Div(
                    style={"display": "grid", "gap": "8px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap"},
                            children=[
                                dcc.Upload(
                                    id="local-file-upload",
                                    children=html.Button("Upload CSV/XLSX"),
                                    accept=".csv,.xlsx,.xls",
                                    multiple=False,
                                ),
                                html.Div(
                                    style={"minWidth": "250px"},
                                    children=[
                                        html.Label("Sheet", style={"fontSize": "12px", "marginBottom": "2px"}),
                                        dcc.Dropdown(id="sheet-name-dropdown", options=[], value=None, clearable=False, disabled=True),
                                    ],
                                ),
                                html.Button("Add Row", id="add-row-btn"),
                                html.Button("Download CSV", id="download-btn"),
                                dcc.Download(id="download-csv"),
                            ],
                        ),
                        html.Div(
                            style={"display": "flex", "gap": "10px", "alignItems": "flex-end", "flexWrap": "wrap"},
                            children=[
                                html.Div(
                                    style={"minWidth": "360px", "flex": "1 1 600px"},
                                    children=[
                                        html.Label("Google Sheets URL", style={"fontSize": "12px", "marginBottom": "2px"}),
                                        dcc.Input(
                                            id="google-sheet-url",
                                            type="text",
                                            value=DEFAULT_GOOGLE_SHEET_URL,
                                            placeholder="https://docs.google.com/spreadsheets/d/<id>/edit",
                                            debounce=True,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                ),
                                html.Button("Load/Refresh Google Sheet", id="load-google-sheet-btn"),
                                html.Div(
                                    style={"display": "grid", "gap": "2px", "paddingBottom": "4px"},
                                    children=[
                                        html.Div(
                                            "First load may open a Google sign-in browser window.",
                                            style={"fontSize": "12px", "color": "#666"},
                                        ),
                                        html.Div(
                                            id="google-auth-status",
                                            style={"fontSize": "12px", "color": "#666"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(id="load-status", style={"marginTop": "8px"}),
                dash_table.DataTable(
                    id="raw-table",
                    columns=_default_columns(),
                    data=[],
                    editable=True,
                    row_deletable=True,
                    page_size=12,
                    style_table={"overflowX": "auto", "marginTop": "8px"},
                    style_cell={"textAlign": "left", "fontFamily": "sans-serif", "fontSize": 13},
                ),
                html.Hr(),
                html.H4("Analysis settings"),
                html.Div(
                    style={"maxWidth": "420px"},
                    children=[
                        html.Label("Confidence level"),
                        dcc.Dropdown(
                            id="confidence-level",
                            options=CONFIDENCE_OPTIONS,
                            value=0.95,
                            clearable=False,
                        ),
                    ],
                ),
                html.H4("Per-policy intervals (success rate & quality)"),
                dash_table.DataTable(
                    id="summary-table",
                    columns=[],
                    data=[],
                    page_size=12,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "fontFamily": "sans-serif", "fontSize": 13},
                ),
                html.Hr(),
                html.H4("A/B comparison (base vs experimental)"),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px", "maxWidth": "760px"},
                    children=[
                        html.Div(
                            [
                                html.Label("Policy A (base)"),
                                dcc.Dropdown(id="policy-a-dropdown", options=[], value=None),
                            ]
                        ),
                        html.Div(
                            [
                                html.Label("Policy B (experimental)"),
                                dcc.Dropdown(id="policy-b-dropdown", options=[], value=None),
                            ]
                        ),
                    ],
                ),
                html.Div(id="ab-output", style={"marginTop": "10px"}),
                html.Div(id="ab-failure-peek", style={"marginTop": "8px"}),
                dcc.Graph(id="ab-comparison-graph"),
                dcc.Graph(id="ab-quality-graph"),
                dcc.Graph(id="ab-dropin-graph"),
                dcc.Graph(id="ab-violin-graph"),
                html.H4("Failure mode highlights (quick sneak peek)"),
                html.Div(
                    id="failure-main-highlights",
                    style={
                        "background": "#fafafa",
                        "border": "1px solid #e0e0e0",
                        "borderRadius": "6px",
                        "padding": "10px 12px",
                    },
                ),
                html.Hr(),
                html.H4("Multi-policy comparison + compact letter display"),
                html.Div(
                    style={"display": "grid", "gap": "8px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
                            children=[
                                html.Button("Select All", id="select-all-btn"),
                                html.Button("Deselect All", id="deselect-all-btn"),
                                html.Div(
                                    style={"minWidth": "260px", "maxWidth": "360px"},
                                    children=[
                                        html.Label("Testing Group", style={"fontSize": "12px", "marginBottom": "2px"}),
                                        dcc.Dropdown(
                                            id="testing-group-dropdown",
                                            options=[],
                                            value=None,
                                            clearable=True,
                                            placeholder="Select a tag",
                                            disabled=True,
                                        ),
                                    ],
                                ),
                                html.Button("Plot Tag + Base", id="apply-testing-group-btn"),
                                html.Button("Clear Tag Filter", id="clear-testing-group-btn"),
                            ],
                        ),
                        html.Div(
                            style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
                            children=[
                                html.Button("Original Order", id="sort-original-btn"),
                                html.Button("Sort by Success Rate \u2193", id="sort-success-btn"),
                                html.Button("Sort by Success Rate \u2191", id="sort-success-asc-btn"),
                                html.Button("Sort by Quality Score [%] \u2193", id="sort-quality-btn"),
                                html.Button("Sort by Quality Score [%] \u2191", id="sort-quality-asc-btn"),
                                html.Button("Sort by Attempt Drop-in \u2193", id="sort-dropin-btn"),
                                html.Button("Sort by Attempt Drop-in \u2191", id="sort-dropin-asc-btn"),
                            ],
                        ),
                    ],
                ),
                html.Div(id="sort-status", style={"marginTop": "6px", "fontSize": "13px"}),
                html.Div(
                    style={"marginTop": "8px"},
                    children=[
                        html.Label("Policies to include", style={"fontSize": "13px", "fontWeight": "600"}),
                        dcc.Checklist(
                            id="policy-checklist",
                            options=[],
                            value=[],
                            inline=False,
                            style={
                                "marginTop": "6px",
                                "maxHeight": "180px",
                                "overflowY": "auto",
                                "border": "1px solid #e0e0e0",
                                "borderRadius": "6px",
                                "padding": "8px 10px",
                                "background": "#fafafa",
                                "columnCount": 2,
                                "columnGap": "16px",
                            },
                            labelStyle={"display": "block", "marginBottom": "4px", "whiteSpace": "nowrap"},
                        ),
                    ],
                ),
                dcc.Checklist(
                    id="show-final-violin-toggle",
                    options=[{"label": "Show final base-vs-policy violin panel", "value": "show"}],
                    value=["show"],
                    inline=True,
                    style={"marginTop": "8px"},
                ),
                dcc.Graph(id="performance-graph"),
                dcc.Graph(id="quality-score-graph"),
                dcc.Graph(id="dropin-ratio-graph"),
                dcc.Graph(id="sr-vs-quality-graph"),
                html.Div(id="final-violin-wrapper", children=[dcc.Graph(id="cld-violin-graph")]),
                dash_table.DataTable(
                    id="cld-table",
                    columns=[],
                    data=[],
                    page_size=12,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "fontFamily": "sans-serif", "fontSize": 13},
                    style_data_conditional=[],
                ),
                html.Hr(),
                html.H4("All-vs-all compact letter display (CLD)"),
                html.P(
                    "All selected policies compared pairwise (Holm-adjusted). "
                    "Policies sharing a letter are not significantly different.",
                    style={"fontSize": "13px", "color": "#666"},
                ),
                dash_table.DataTable(
                    id="allvsall-cld-table",
                    columns=[],
                    data=[],
                    page_size=20,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "fontFamily": "sans-serif", "fontSize": 13},
                    style_data_conditional=[],
                ),
                dcc.Checklist(
                    id="show-allvsall-violin-toggle",
                    options=[{"label": "Show all-vs-all CLD violin", "value": "show"}],
                    value=["show"],
                    inline=True,
                    style={"marginTop": "8px"},
                ),
                html.Div(id="allvsall-violin-wrapper", children=[dcc.Graph(id="allvsall-violin-graph")]),
            ],
        ),
        html.Div(
            id="failure-page",
            style={"display": "none"},
            children=[
                html.H2("Failure Mode Analysis"),
                html.P(
                    "Dedicated page for detailed rollout-level diagnostics. "
                    "Load per-policy detail links, then inspect the aggregate condition grid across all completed policies."
                ),
                html.Div(
                    style={"display": "grid", "gap": "8px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "gap": "8px", "alignItems": "flex-end", "flexWrap": "wrap"},
                            children=[
                                html.Button("Load/Refresh detailed rollout sheets", id="load-failure-details-btn"),
                                html.Div(
                                    style={"minWidth": "220px"},
                                    children=[
                                        html.Label("Metric", style={"fontSize": "12px", "marginBottom": "2px"}),
                                        dcc.Dropdown(
                                            id="failure-metric-dropdown",
                                            options=FAILURE_METRIC_OPTIONS,
                                            value="failure_rate",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(id="failure-load-status", style={"marginTop": "8px", "fontSize": "13px"}),
                dcc.Graph(id="failure-aggregate-graph"),
                html.H4("Top hardest conditions"),
                dash_table.DataTable(
                    id="failure-top-conditions-table",
                    columns=[],
                    data=[],
                    page_size=12,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "fontFamily": "sans-serif", "fontSize": 13},
                ),
                html.H4("Top easiest conditions"),
                dash_table.DataTable(
                    id="failure-easiest-conditions-table",
                    columns=[],
                    data=[],
                    page_size=12,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "fontFamily": "sans-serif", "fontSize": 13},
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("main-page", "style"),
    Output("failure-page", "style"),
    Input("page-tabs", "value"),
)
def toggle_pages(active_tab: str | None):
    if active_tab == "failure":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


@app.callback(
    Output("failure-detail-store", "data"),
    Output("failure-load-status", "children"),
    Input("load-failure-details-btn", "n_clicks"),
    State("raw-table", "data"),
    prevent_initial_call=True,
)
def load_failure_detail_data(
    _load_clicks: int,
    rows: list[dict] | None,
):
    empty_store = {
        "records": [],
        "condition_columns": [],
        "default_x": None,
        "default_y": None,
    }

    if not rows:
        return empty_store, "Load the main policy table first."

    all_policy_links = _collect_policy_detail_links(rows)
    clean_df = _raw_to_clean_df(rows)
    if "has_success_rate_input" in clean_df.columns:
        completed_policy_names = {
            str(name).strip()
            for name in clean_df.loc[clean_df["has_success_rate_input"].fillna(True), "model_name"].astype(str).tolist()
        }
    else:
        completed_policy_names = {str(name).strip() for name in clean_df["model_name"].astype(str).tolist()}

    policy_links = [
        item
        for item in all_policy_links
        if str(item.get("policy_name", "")).strip() in completed_policy_names
    ]
    planning_rows_skipped = max(0, len(all_policy_links) - len(policy_links))

    if not policy_links:
        message = "No valid detail-sheet URLs found for completed policies."
        if planning_rows_skipped > 0:
            message += " Rows without success rate were skipped by design."
        return empty_store, message

    all_records: list[dict[str, object]] = []
    condition_by_key: dict[str, str] = {}
    policy_order: list[str] = []
    errors: list[str] = []

    for item in policy_links:
        policy_name = str(item["policy_name"])
        detail_url = str(item["detail_url"])
        try:
            detail_df = load_google_spreadsheet(detail_url, enrich_hyperlinks=False)
            records, condition_columns = _normalize_detail_rollout_frame(
                detail_df,
                policy_name=policy_name,
                detail_url=detail_url,
            )
        except Exception as exc:
            errors.append(f"{policy_name}: {exc}")
            continue

        if not records:
            errors.append(f"{policy_name}: no usable rollout rows")
            continue

        policy_order.append(policy_name)
        all_records.extend(records)
        for entry in condition_columns:
            key = str(entry.get("key", "")).strip()
            label = str(entry.get("label", "")).strip()
            if key and label and key not in condition_by_key:
                condition_by_key[key] = label

    if not all_records:
        message = "Could not load rollout details from any policy links."
        if errors:
            preview = "; ".join(errors[:2])
            if len(errors) > 2:
                preview += "; ..."
            message += f" ({preview})"
        return empty_store, message

    condition_columns = [{"key": key, "label": label} for key, label in condition_by_key.items()]
    default_x = _pick_default_condition_key(condition_columns, FAILURE_DEFAULT_X_COLUMN_CANDIDATES)
    default_y = _pick_default_condition_key(condition_columns, FAILURE_DEFAULT_Y_COLUMN_CANDIDATES)
    if default_x and default_y and default_x == default_y and len(condition_columns) > 1:
        alternatives = [entry["key"] for entry in condition_columns if entry["key"] != default_x]
        default_y = alternatives[0] if alternatives else default_y
    condition_label_map = {entry["key"]: entry["label"] for entry in condition_columns}
    x_label = condition_label_map.get(str(default_x), str(default_x or "N/A"))
    y_label = condition_label_map.get(str(default_y), str(default_y or "N/A"))
    unique_policies = list(dict.fromkeys(policy_order))

    status = (
        f"Loaded {len(all_records)} rollout rows from {len(unique_policies)} policy detail sheets. "
        f"Detected {len(condition_columns)} condition columns. "
        f"Using axes: {y_label} × {x_label}."
    )
    if errors:
        status += f" Skipped {len(errors)} sheet(s)."
    if planning_rows_skipped > 0:
        status += f" Ignored {planning_rows_skipped} planning row(s) without success rate."

    store_data = {
        "records": all_records,
        "condition_columns": condition_columns,
        "default_x": default_x,
        "default_y": default_y,
    }

    return (
        store_data,
        status,
    )


@app.callback(
    Output("failure-aggregate-graph", "figure"),
    Output("failure-top-conditions-table", "data"),
    Output("failure-top-conditions-table", "columns"),
    Output("failure-easiest-conditions-table", "data"),
    Output("failure-easiest-conditions-table", "columns"),
    Output("failure-main-highlights", "children"),
    Output("ab-failure-peek", "children"),
    Input("failure-detail-store", "data"),
    Input("failure-metric-dropdown", "value"),
    Input("policy-a-dropdown", "value"),
    Input("policy-b-dropdown", "value"),
)
def update_failure_views(
    failure_store: dict | None,
    metric_mode: str | None,
    policy_a: str | None,
    policy_b: str | None,
):
    empty_message = "Load detailed rollout sheets from the Failure Mode Analysis tab to see highlights."
    empty_ab_peek = html.Div(
        "Failure-mode sneak peek appears here after detailed sheets are loaded.",
        style={
            "fontSize": "13px",
            "color": "#555",
            "background": "#fafafa",
            "border": "1px solid #e0e0e0",
            "borderRadius": "6px",
            "padding": "8px 12px",
        },
    )

    if not failure_store or not failure_store.get("records"):
        return (
            _failure_empty_figure("Load detail sheets to render aggregate heatmap"),
            [],
            [],
            [],
            [],
            empty_message,
            empty_ab_peek,
        )

    detail_df = pd.DataFrame(failure_store.get("records") or [])
    if detail_df.empty:
        return (
            _failure_empty_figure("No detail rollout rows available"),
            [],
            [],
            [],
            [],
            empty_message,
            empty_ab_peek,
        )

    condition_columns = {
        str(entry.get("key", "")): str(entry.get("label", ""))
        for entry in (failure_store.get("condition_columns") or [])
        if str(entry.get("key", "")).strip()
    }
    x_key = str(failure_store.get("default_x") or "")
    y_key = str(failure_store.get("default_y") or "")
    available_keys = [key for key in condition_columns if key in detail_df.columns]

    if (not x_key or x_key not in detail_df.columns) and available_keys:
        x_key = available_keys[0]
    if (
        (not y_key or y_key not in detail_df.columns or y_key == x_key)
        and len(available_keys) > 1
    ):
        y_key = next((key for key in available_keys if key != x_key), "")

    if not x_key or not y_key or x_key not in detail_df.columns or y_key not in detail_df.columns:
        return (
            _failure_empty_figure("Condition axes could not be auto-detected for failure heatmap"),
            [],
            [],
            [],
            [],
            "Condition axes are not available yet. Reload detail sheets and try again.",
            empty_ab_peek,
        )

    detail_df = detail_df.copy()
    detail_df["policy_name"] = detail_df["policy_name"].astype(str)
    detail_df["task_success"] = pd.to_numeric(detail_df["task_success"], errors="coerce")
    detail_df = detail_df[detail_df["task_success"].notna()].copy()
    if detail_df.empty:
        return (
            _failure_empty_figure("No valid Task Success rows found in detail sheets"),
            [],
            [],
            [],
            [],
            "No valid Task Success values detected in loaded detail sheets.",
            empty_ab_peek,
        )

    detail_df[x_key] = detail_df[x_key].fillna("NA").astype(str)
    detail_df[y_key] = detail_df[y_key].fillna("NA").astype(str)
    detail_df["quality_score_pct"] = pd.to_numeric(detail_df.get("quality_score_pct"), errors="coerce")

    policy_order = list(dict.fromkeys(detail_df["policy_name"].tolist()))

    grouped = (
        detail_df.groupby(["policy_name", x_key, y_key], as_index=False)
        .agg(
            n=("task_success", "size"),
            success_rate=("task_success", "mean"),
            quality_score_pct=("quality_score_pct", "mean"),
        )
    )
    grouped["failure_rate"] = 1.0 - grouped["success_rate"]
    grouped["success_rate_pct"] = grouped["success_rate"] * 100.0
    grouped["failure_rate_pct"] = grouped["failure_rate"] * 100.0

    aggregate = (
        detail_df.groupby([x_key, y_key], as_index=False)
        .agg(
            n=("task_success", "size"),
            success_rate=("task_success", "mean"),
            quality_score_pct=("quality_score_pct", "mean"),
            policy_count=("policy_name", "nunique"),
        )
    )
    aggregate["failure_rate"] = 1.0 - aggregate["success_rate"]
    aggregate["success_rate_pct"] = aggregate["success_rate"] * 100.0
    aggregate["failure_rate_pct"] = aggregate["failure_rate"] * 100.0

    metric_mode = metric_mode or "failure_rate"
    if metric_mode == "success_rate":
        metric_key = "success_rate_pct"
        metric_label = "Success rate (%)"
        colorscale = "Greys"
        zmin, zmax = 0.0, 100.0
    elif metric_mode == "quality_score":
        metric_key = "quality_score_pct"
        metric_label = "Quality score mean (%)"
        colorscale = "Greys"
        finite_values = pd.to_numeric(grouped[metric_key], errors="coerce").dropna()
        if finite_values.empty:
            no_quality_msg = "Quality score column is missing in loaded detail sheets"
            return (
                _failure_empty_figure(no_quality_msg),
                [],
                [],
                [],
                [],
                no_quality_msg,
                empty_ab_peek,
            )
        zmin = max(0.0, float(math.floor(finite_values.min() / 5.0) * 5.0))
        zmax = min(100.0, float(math.ceil(finite_values.max() / 5.0) * 5.0))
        if zmax <= zmin:
            zmax = zmin + 1.0
    elif metric_mode == "sample_count":
        metric_key = "n"
        metric_label = "Sample count (n)"
        colorscale = "Greys"
        zmin = 0.0
        zmax = float(max(1.0, grouped[metric_key].max()))
    else:
        metric_key = "failure_rate_pct"
        metric_label = "Failure rate (%)"
        colorscale = "Greys"
        zmin, zmax = 0.0, 100.0

    x_values = sorted({str(v) for v in grouped[x_key].dropna().tolist()}, key=_condition_sort_key)
    y_values = sorted({str(v) for v in grouped[y_key].dropna().tolist()}, key=_condition_sort_key)

    aggregate_fig = _build_failure_aggregate_figure(
        aggregate,
        x_values,
        y_values,
        x_key,
        y_key,
        metric_key,
        metric_label,
        colorscale,
        zmin,
        zmax,
    )

    x_label = condition_columns.get(x_key, x_key)
    y_label = condition_columns.get(y_key, y_key)

    hardest = aggregate.sort_values(["failure_rate_pct", "n"], ascending=[False, False]).copy()
    hardest = hardest.head(12)
    hardest["failure_rate_pct"] = hardest["failure_rate_pct"].round(2)
    hardest["success_rate_pct"] = hardest["success_rate_pct"].round(2)
    hardest["quality_score_pct"] = pd.to_numeric(hardest["quality_score_pct"], errors="coerce").round(2)
    hardest_data = hardest.rename(columns={x_key: "x_condition", y_key: "y_condition"})[
        ["x_condition", "y_condition", "failure_rate_pct", "success_rate_pct", "quality_score_pct", "n", "policy_count"]
    ].to_dict("records")
    hardest_columns = [
        {"name": x_label, "id": "x_condition"},
        {"name": y_label, "id": "y_condition"},
        {"name": "Failure (%)", "id": "failure_rate_pct"},
        {"name": "Success (%)", "id": "success_rate_pct"},
        {"name": "Quality (%)", "id": "quality_score_pct"},
        {"name": "n", "id": "n"},
        {"name": "Policies", "id": "policy_count"},
    ]

    easiest = aggregate.sort_values(["failure_rate_pct", "n"], ascending=[True, False]).copy()
    easiest = easiest.head(12)
    easiest["failure_rate_pct"] = easiest["failure_rate_pct"].round(2)
    easiest["success_rate_pct"] = easiest["success_rate_pct"].round(2)
    easiest["quality_score_pct"] = pd.to_numeric(easiest["quality_score_pct"], errors="coerce").round(2)
    easiest_data = easiest.rename(columns={x_key: "x_condition", y_key: "y_condition"})[
        ["x_condition", "y_condition", "failure_rate_pct", "success_rate_pct", "quality_score_pct", "n", "policy_count"]
    ].to_dict("records")
    easiest_columns = hardest_columns

    hardest_items: list[html.Li] = []
    for _, row in hardest.head(3).iterrows():
        hardest_items.append(
            html.Li(
                f"{row[y_key]} × {row[x_key]}: {row['failure_rate_pct']:.1f}% failure "
                f"(n={int(row['n'])}, {int(row['policy_count'])} policies)"
            )
        )

    easiest_items: list[html.Li] = []
    for _, row in easiest.head(3).iterrows():
        easiest_items.append(
            html.Li(
                f"{row[y_key]} × {row[x_key]}: {row['failure_rate_pct']:.1f}% failure "
                f"(n={int(row['n'])}, {int(row['policy_count'])} policies)"
            )
        )

    main_highlights = html.Div(
        [
            html.Div(
                f"Detailed rollout data loaded: {len(detail_df)} rows, {len(policy_order)} policies. "
                f"Grid size detected: {len(y_values)} × {len(x_values)} ({y_label} × {x_label}).",
                style={"fontWeight": "600"},
            ),
            html.Div("Hardest conditions:", style={"fontWeight": "600", "marginTop": "6px"}),
            html.Ul(hardest_items, style={"marginTop": "4px", "marginBottom": "6px"}),
            html.Div("Easiest conditions:", style={"fontWeight": "600", "marginTop": "6px"}),
            html.Ul(easiest_items, style={"marginTop": "4px", "marginBottom": "6px"}),
            html.Div(
                "Showing a single aggregate grayscale heatmap averaged across all completed policies.",
                style={"fontSize": "12px", "color": "#666", "marginTop": "4px"},
            ),
        ]
    )

    ab_peek = empty_ab_peek
    if policy_a and policy_b and policy_a != policy_b:
        pair_df = grouped[grouped["policy_name"].isin([policy_a, policy_b])].copy()
        a_df = pair_df[pair_df["policy_name"] == policy_a][[x_key, y_key, "failure_rate_pct", "n"]].rename(
            columns={"failure_rate_pct": "failure_a", "n": "n_a"}
        )
        b_df = pair_df[pair_df["policy_name"] == policy_b][[x_key, y_key, "failure_rate_pct", "n"]].rename(
            columns={"failure_rate_pct": "failure_b", "n": "n_b"}
        )
        merged = a_df.merge(b_df, on=[x_key, y_key], how="inner")
        if not merged.empty:
            merged["delta_b_minus_a"] = merged["failure_b"] - merged["failure_a"]
            worst_reg = merged.sort_values("delta_b_minus_a", ascending=False).iloc[0]
            best_gain = merged.sort_values("delta_b_minus_a", ascending=True).iloc[0]

            reg_text = (
                f"Largest B regression: +{worst_reg['delta_b_minus_a']:.1f} pp failure at "
                f"{worst_reg[y_key]} × {worst_reg[x_key]} (A n={int(worst_reg['n_a'])}, B n={int(worst_reg['n_b'])})."
            )
            gain_text = (
                f"Largest B improvement: {best_gain['delta_b_minus_a']:.1f} pp failure at "
                f"{best_gain[y_key]} × {best_gain[x_key]} (A n={int(best_gain['n_a'])}, B n={int(best_gain['n_b'])})."
            )
            ab_peek = html.Div(
                [
                    html.Div("Failure-mode sneak peek (A/B)", style={"fontWeight": "600", "marginBottom": "4px"}),
                    html.Div(reg_text),
                    html.Div(gain_text),
                ],
                style={
                    "fontSize": "13px",
                    "background": "#fafafa",
                    "border": "1px solid #e0e0e0",
                    "borderRadius": "6px",
                    "padding": "8px 12px",
                },
            )

    return (
        aggregate_fig,
        hardest_data,
        hardest_columns,
        easiest_data,
        easiest_columns,
        main_highlights,
        ab_peek,
    )


@app.callback(
    Output("sort-mode-store", "data"),
    Input("sort-original-btn", "n_clicks"),
    Input("sort-success-btn", "n_clicks"),
    Input("sort-success-asc-btn", "n_clicks"),
    Input("sort-quality-btn", "n_clicks"),
    Input("sort-quality-asc-btn", "n_clicks"),
    Input("sort-dropin-btn", "n_clicks"),
    Input("sort-dropin-asc-btn", "n_clicks"),
    State("sort-mode-store", "data"),
    prevent_initial_call=True,
)
def update_sort_mode(
    _sort_original_clicks: int,
    _sort_success_clicks: int,
    _sort_success_asc_clicks: int,
    _sort_quality_clicks: int,
    _sort_quality_asc_clicks: int,
    _sort_dropin_clicks: int,
    _sort_dropin_asc_clicks: int,
    current_mode: str | None,
):
    trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
    if trigger == "sort-success-btn":
        return "success_rate"
    if trigger == "sort-success-asc-btn":
        return "success_rate_asc"
    if trigger == "sort-quality-btn":
        return "quality_score"
    if trigger == "sort-quality-asc-btn":
        return "quality_score_asc"
    if trigger == "sort-dropin-btn":
        return "dropin_ratio"
    if trigger == "sort-dropin-asc-btn":
        return "dropin_ratio_asc"
    if trigger == "sort-original-btn":
        return "original"
    return current_mode or "original"


@app.callback(
    Output("google-auth-status", "children"),
    Output("google-auth-status", "style"),
    Input("google-sheet-url", "value"),
    Input("load-google-sheet-btn", "n_clicks"),
)
def update_google_auth_status(
    _google_sheet_url: str | None,
    _load_google_sheet_clicks: int | None,
):
    status = get_google_auth_status()
    message = f"Google auth: {status.get('message', 'status unavailable')}"

    base_style = {
        "fontSize": "12px",
        "paddingBottom": "0px",
    }
    if bool(status.get("ready")):
        color = "#2e7d32"
    elif str(status.get("state")) in {"signin", "setup"}:
        color = "#ef6c00"
    else:
        color = "#c62828"

    return message, {**base_style, "color": color}


@app.callback(
    Output("raw-table", "data"),
    Output("raw-table", "columns"),
    Output("load-status", "children"),
    Output("uploaded-file-store", "data"),
    Output("sheet-name-dropdown", "options"),
    Output("sheet-name-dropdown", "value"),
    Output("sheet-name-dropdown", "disabled"),
    Input("local-file-upload", "contents"),
    Input("sheet-name-dropdown", "value"),
    Input("load-google-sheet-btn", "n_clicks"),
    State("local-file-upload", "filename"),
    State("google-sheet-url", "value"),
    State("uploaded-file-store", "data"),
    prevent_initial_call=True,
)
def load_file_to_table(
    upload_contents: str | None,
    selected_sheet_name: str | None,
    _load_google_sheet_clicks: int,
    filename: str | None,
    google_sheet_url: str | None,
    stored_upload: dict | None,
):
    trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None

    if trigger == "local-file-upload":
        if not upload_contents:
            return (
                no_update,
                no_update,
                "Upload a CSV/XLSX file first.",
                no_update,
                no_update,
                no_update,
                no_update,
            )

        try:
            sheets = list_local_spreadsheet_sheets(upload_contents, filename)
        except Exception as exc:
            return no_update, no_update, f"Error: {exc}", no_update, no_update, no_update, no_update

        default_sheet = sheets[0] if sheets else CSV_SINGLE_SHEET_NAME
        upload_state = {
            "source": "local",
            "contents": upload_contents,
            "filename": filename,
            "sheets": sheets,
            "default_sheet": default_sheet,
        }
    elif trigger == "load-google-sheet-btn":
        clean_url = str(google_sheet_url or "").strip()
        if not clean_url:
            return (
                no_update,
                no_update,
                "Paste a Google Sheets URL first.",
                no_update,
                no_update,
                no_update,
                no_update,
            )

        try:
            metadata = list_google_spreadsheet_sheets(clean_url)
        except Exception as exc:
            return no_update, no_update, f"Error: {exc}", no_update, no_update, no_update, no_update

        sheet_entries = metadata.get("sheets") or []
        sheets = [str(entry.get("name")) for entry in sheet_entries if str(entry.get("name", "")).strip()]
        default_sheet = str(metadata.get("default_sheet") or (sheets[0] if sheets else CSV_SINGLE_SHEET_NAME))
        upload_state = {
            "source": "google",
            "url": clean_url,
            "spreadsheet_id": metadata.get("spreadsheet_id"),
            "sheet_title": metadata.get("title") or "Google Sheet",
            "sheets": sheets,
            "default_sheet": default_sheet,
        }
    else:
        if not stored_upload:
            return (
                no_update,
                no_update,
                "Upload a CSV/XLSX file or load a Google Sheet first.",
                no_update,
                no_update,
                no_update,
                no_update,
            )
        upload_state = stored_upload
        sheets = upload_state.get("sheets") or [CSV_SINGLE_SHEET_NAME]
        default_sheet = selected_sheet_name or upload_state.get("default_sheet") or sheets[0]

    if selected_sheet_name and selected_sheet_name in sheets:
        sheet_name = selected_sheet_name
    else:
        sheet_name = default_sheet if default_sheet in sheets else (sheets[0] if sheets else CSV_SINGLE_SHEET_NAME)

    options = [{"label": name, "value": name} for name in sheets]
    disabled = len(sheets) <= 1

    try:
        source = str(upload_state.get("source") or "local")
        if source == "google":
            df = load_google_spreadsheet(upload_state.get("url", ""), sheet_name=sheet_name)
        else:
            df = load_local_spreadsheet(upload_state["contents"], upload_state.get("filename"), sheet_name=sheet_name)
        normalized = normalize_policy_dataframe(df)
    except Exception as exc:
        return no_update, no_update, f"Error: {exc}", upload_state, options, sheet_name, disabled

    columns = [{"name": col, "id": col, "editable": col not in {"success_rate", "wilson_low", "wilson_high"}} for col in normalized.columns]
    source = str(upload_state.get("source") or "local")
    if source == "google":
        display_name = upload_state.get("sheet_title") or "Google Sheet"
        if disabled:
            status = f"Loaded {len(normalized)} rows from Google Sheet: {display_name}."
        else:
            status = f"Loaded {len(normalized)} rows from Google Sheet: {display_name} (tab: {sheet_name})."
    else:
        display_name = upload_state.get("filename") or "uploaded file"
        if disabled:
            status = f"Loaded {len(normalized)} rows from {display_name}."
        else:
            status = f"Loaded {len(normalized)} rows from {display_name} (sheet: {sheet_name})."

    return normalized.to_dict("records"), columns, status, upload_state, options, sheet_name, disabled


@app.callback(
    Output("raw-table", "data", allow_duplicate=True),
    Input("add-row-btn", "n_clicks"),
    State("raw-table", "data"),
    State("raw-table", "columns"),
    prevent_initial_call=True,
)
def add_table_row(_: int, rows: list[dict] | None, columns: list[dict] | None):
    if columns is None:
        columns = _default_columns()
    rows = rows or []

    new_row = {}
    for col in columns:
        col_id = col.get("id")
        if col_id == "trials":
            new_row[col_id] = DEFAULT_TRIALS
        elif col_id == "successes":
            new_row[col_id] = 0
        else:
            new_row[col_id] = ""
    rows.append(new_row)
    return rows


@app.callback(
    Output("download-csv", "data"),
    Input("download-btn", "n_clicks"),
    State("raw-table", "data"),
    prevent_initial_call=True,
)
def download_table(_: int, rows: list[dict] | None):
    if not rows:
        return no_update
    df = pd.DataFrame(rows)
    return dcc.send_data_frame(df.to_csv, "policy_rollout_log.csv", index=False)


@app.callback(
    Output("policy-a-dropdown", "options"),
    Output("policy-a-dropdown", "value"),
    Output("policy-b-dropdown", "options"),
    Output("policy-b-dropdown", "value"),
    Output("policy-checklist", "options"),
    Output("policy-checklist", "value"),
    Output("testing-group-dropdown", "options"),
    Output("testing-group-dropdown", "value"),
    Output("testing-group-dropdown", "disabled"),
    Output("active-testing-group-store", "data"),
    Input("raw-table", "data"),
    Input("select-all-btn", "n_clicks"),
    Input("deselect-all-btn", "n_clicks"),
    Input("apply-testing-group-btn", "n_clicks"),
    Input("clear-testing-group-btn", "n_clicks"),
    State("policy-a-dropdown", "value"),
    State("policy-b-dropdown", "value"),
    State("policy-checklist", "value"),
    State("testing-group-dropdown", "value"),
    State("active-testing-group-store", "data"),
)
def sync_policy_selectors(
    rows: list[dict] | None,
    _select_all_clicks: int,
    _deselect_all_clicks: int,
    _apply_testing_group_clicks: int,
    _clear_testing_group_clicks: int,
    current_a: str | None,
    current_b: str | None,
    current_checked: list[str] | None,
    selected_testing_group: str | None,
    current_active_group: str | None,
):
    clean_df = _raw_to_clean_df(rows)
    if "has_success_rate_input" in clean_df.columns and not clean_df.empty:
        eligible_models = clean_df.loc[clean_df["has_success_rate_input"].fillna(True), "model_name"].astype(str).tolist()
    else:
        eligible_models = clean_df["model_name"].astype(str).tolist() if not clean_df.empty else []

    models = eligible_models
    display_map, _pfx = _make_display_names(models)
    options = [{"label": display_map.get(model, model), "value": model} for model in models]

    tag_df = clean_df.copy()
    if not tag_df.empty:
        tag_df = tag_df[tag_df["model_name"].astype(str).isin(models)].copy()
        tag_df = tag_df.sort_values("source_order", kind="stable")
    if "testing_group" not in tag_df.columns:
        tag_df["testing_group"] = pd.NA
    if "is_base_group" not in tag_df.columns:
        tag_df["is_base_group"] = False

    tag_df["testing_group"] = tag_df["testing_group"].astype(str).str.strip()
    tag_df["testing_group"] = tag_df["testing_group"].replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    tag_df["is_base_group"] = tag_df["is_base_group"].fillna(False).astype(bool)

    non_base_tag_df = tag_df[tag_df["testing_group"].notna() & (~tag_df["is_base_group"])].copy()
    tag_order = non_base_tag_df["testing_group"].astype(str).drop_duplicates().tolist()
    tag_options = [{"label": tag, "value": tag} for tag in tag_order]
    tag_set = set(tag_order)

    base_models = tag_df.loc[tag_df["is_base_group"], "model_name"].astype(str).drop_duplicates().tolist()
    if not base_models and current_a in models:
        base_models = [str(current_a)]

    group_to_models: dict[str, list[str]] = {}
    for tag in tag_order:
        tag_models = (
            non_base_tag_df.loc[non_base_tag_df["testing_group"].astype(str) == tag, "model_name"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        group_to_models[tag] = [model for model in models if model in set(tag_models)]

    trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else "raw-table"

    if current_a in models:
        policy_a = current_a
    else:
        policy_a = models[0] if models else None

    if not base_models and policy_a in models:
        base_models = [str(policy_a)]

    remaining_for_b = [m for m in models if m != policy_a]
    if current_b in models and current_b != policy_a:
        policy_b = current_b
    else:
        policy_b = remaining_for_b[0] if remaining_for_b else (models[0] if models else None)

    current_checked = current_checked or []
    active_group = str(current_active_group) if current_active_group in tag_set else None

    def _group_plus_base_selection(group_name: str) -> list[str]:
        selected = set(group_to_models.get(group_name, []))
        selected.update(base_models)
        return [model for model in models if model in selected]

    if trigger == "select-all-btn":
        checked = eligible_models
        active_group = None
    elif trigger == "deselect-all-btn":
        checked = []
        active_group = None
    elif trigger == "apply-testing-group-btn":
        requested_group = str(selected_testing_group) if selected_testing_group in tag_set else None
        if requested_group is not None:
            checked = _group_plus_base_selection(requested_group)
            active_group = requested_group
        else:
            checked = [m for m in current_checked if m in models]
            if not checked:
                checked = eligible_models
            active_group = None
    elif trigger == "clear-testing-group-btn":
        checked = eligible_models
        active_group = None
    elif trigger == "raw-table" and active_group in tag_set:
        checked = _group_plus_base_selection(active_group)
    else:
        checked = [m for m in current_checked if m in models]
        if not checked:
            checked = eligible_models

    dropdown_disabled = len(tag_options) == 0
    if dropdown_disabled:
        dropdown_value = None
        active_group = None
    elif trigger == "clear-testing-group-btn":
        dropdown_value = None
    elif selected_testing_group in tag_set:
        dropdown_value = str(selected_testing_group)
    elif active_group in tag_set:
        dropdown_value = str(active_group)
    else:
        dropdown_value = None

    return (
        options,
        policy_a,
        options,
        policy_b,
        options,
        checked,
        tag_options,
        dropdown_value,
        dropdown_disabled,
        active_group,
    )


@app.callback(
    Output("summary-table", "data"),
    Output("summary-table", "columns"),
    Output("ab-output", "children"),
    Output("ab-comparison-graph", "figure"),
    Output("ab-quality-graph", "figure"),
    Output("ab-dropin-graph", "figure"),
    Output("ab-violin-graph", "figure"),
    Output("performance-graph", "figure"),
    Output("quality-score-graph", "figure"),
    Output("dropin-ratio-graph", "figure"),
    Output("sr-vs-quality-graph", "figure"),
    Output("final-violin-wrapper", "style"),
    Output("cld-violin-graph", "figure"),
    Output("cld-table", "data"),
    Output("cld-table", "columns"),
    Output("cld-table", "style_data_conditional"),
    Output("allvsall-cld-table", "data"),
    Output("allvsall-cld-table", "columns"),
    Output("allvsall-cld-table", "style_data_conditional"),
    Output("allvsall-violin-wrapper", "style"),
    Output("allvsall-violin-graph", "figure"),
    Output("sort-status", "children"),
    Input("raw-table", "data"),
    Input("confidence-level", "value"),
    Input("policy-a-dropdown", "value"),
    Input("policy-b-dropdown", "value"),
    Input("policy-checklist", "value"),
    Input("active-testing-group-store", "data"),
    Input("show-final-violin-toggle", "value"),
    Input("sort-mode-store", "data"),
    Input("show-allvsall-violin-toggle", "value"),
)
def update_analysis(
    rows: list[dict] | None,
    confidence_level: float,
    policy_a: str | None,
    policy_b: str | None,
    selected_policies: list[str] | None,
    active_testing_group: str | None,
    show_final_violin_toggle: list[str] | None,
    sort_mode: str | None,
    show_allvsall_violin_toggle: list[str] | None,
):
    clean_df = _raw_to_clean_df(rows)
    show_final_violin = "show" in (show_final_violin_toggle or [])
    show_allvsall_violin = "show" in (show_allvsall_violin_toggle or [])
    active_group_label = str(active_testing_group).strip() if active_testing_group else ""
    active_group_label = active_group_label if active_group_label else None
    final_violin_style = {"display": "block"} if show_final_violin else {"display": "none"}
    allvsall_violin_style = {"display": "block"} if show_allvsall_violin else {"display": "none"}

    if clean_df.empty:
        return (
            [],
            [],
            "Add policy rows to start analysis.",
            _empty_figure("Pick two policies for A/B plot"),
            _empty_figure("No quality score data for selected A/B policies"),
            _empty_figure("No drop-in ratio data for selected A/B policies"),
            _empty_figure("Pick two policies for posterior uncertainty view"),
            _empty_figure("No policy data"),
            _empty_figure("No quality score data for selected policies"),
            _empty_figure("No drop-in ratio data for selected policies"),
            _empty_figure("No quality data for scatter view"),
            final_violin_style,
            _empty_figure("No policy data"),
            [],
            [],
            [],
            [],
            [],
            [],
            allvsall_violin_style,
            _empty_figure("No policy data"),
            _format_sort_status(sort_mode, active_group=active_group_label),
        )

    analysis_df = clean_df.copy()
    if "has_success_rate_input" in analysis_df.columns:
        analysis_df = analysis_df[analysis_df["has_success_rate_input"].fillna(True)].copy()

    if analysis_df.empty:
        return (
            [],
            [],
            "No concluded policies available: success rate is empty for all rows.",
            _empty_figure("Pick two concluded policies for A/B plot"),
            _empty_figure("No quality score data for selected A/B policies"),
            _empty_figure("No drop-in ratio data for selected A/B policies"),
            _empty_figure("Pick two concluded policies for posterior uncertainty view"),
            _empty_figure("No concluded policies selected"),
            _empty_figure("No quality score data for selected policies"),
            _empty_figure("No drop-in ratio data for selected policies"),
            _empty_figure("No quality data for scatter view"),
            final_violin_style,
            _empty_figure("No concluded policies selected"),
            [],
            [],
            [],
            [],
            [],
            [],
            allvsall_violin_style,
            _empty_figure("No concluded policies selected"),
            _format_sort_status(sort_mode, active_group=active_group_label),
        )

    metrics = prepare_policy_metrics(analysis_df, confidence_level)
    metrics = _apply_sort_mode(metrics, sort_mode, pin_first=policy_a)
    all_names = metrics["model_name"].astype(str).tolist()
    display_map, prefix = _make_display_names(all_names)
    _prefix_sub = f"<br><sup>common prefix: {prefix}</sup>" if prefix else ""
    _multi_sub_parts: list[str] = []
    if active_group_label:
        _multi_sub_parts.append(f"testing group: {active_group_label} + Base")
    if prefix:
        _multi_sub_parts.append(f"common prefix: {prefix}")
    _multi_sub = f"<br><sup>{' | '.join(_multi_sub_parts)}</sup>" if _multi_sub_parts else ""
    policy_colors = _policy_color_map(all_names)
    alpha = 1.0 - confidence_level
    has_quality_std = (
        "quality_score_std_pct" in metrics.columns
        and pd.to_numeric(metrics.get("quality_score_std_pct"), errors="coerce").notna().any()
    )
    has_dropin = (
        "dropin_ratio_pct" in metrics.columns
        and pd.to_numeric(metrics.get("dropin_ratio_pct"), errors="coerce").notna().any()
    )

    summary = metrics.copy()
    summary["success_rate"] = (summary["success_rate"] * 100).round(2)
    summary["wilson_low"] = (summary["wilson_low"] * 100).round(2)
    summary["wilson_high"] = (summary["wilson_high"] * 100).round(2)
    if "quality_score_pct" in summary.columns:
        summary["quality_score_pct"] = pd.to_numeric(summary["quality_score_pct"], errors="coerce").round(2)
    if has_quality_std:
        _qci = metrics.apply(
            lambda row: quality_score_ci(
                float(row["quality_score_pct"]) if pd.notna(row.get("quality_score_pct")) else math.nan,
                float(row["quality_score_std_pct"]) if pd.notna(row.get("quality_score_std_pct")) else math.nan,
                int(row["trials"]),
                confidence_level,
            ),
            axis=1,
            result_type="expand",
        )
        summary["quality_ci_low"] = _qci[0].round(2)
        summary["quality_ci_high"] = _qci[1].round(2)
        summary["quality_score_std_pct"] = pd.to_numeric(metrics["quality_score_std_pct"], errors="coerce").round(2)
    summary_columns = [
        {"name": "model_name", "id": "model_name"},
        {"name": "successes", "id": "successes"},
        {"name": "trials", "id": "trials"},
        {"name": "success_rate_%", "id": "success_rate"},
        {"name": "wilson_low_%", "id": "wilson_low"},
        {"name": "wilson_high_%", "id": "wilson_high"},
    ]
    if "quality_score_pct" in summary.columns:
        summary_columns.append({"name": "quality_score_%", "id": "quality_score_pct"})
    if has_quality_std:
        summary_columns.append({"name": "quality_std_%", "id": "quality_score_std_pct"})
        summary_columns.append({"name": "quality_ci_low_%", "id": "quality_ci_low"})
        summary_columns.append({"name": "quality_ci_high_%", "id": "quality_ci_high"})
    if has_dropin:
        summary["dropin_ratio_pct"] = pd.to_numeric(metrics["dropin_ratio_pct"], errors="coerce").round(2)
        _di_count = pd.to_numeric(metrics.get("dropin_count"), errors="coerce").fillna(0).astype(int)
        _di_trials = metrics["trials"].astype(int)
        _di_ci = pd.DataFrame(
            [wilson_interval(int(dc), int(tr), confidence_level) for dc, tr in zip(_di_count, _di_trials)],
            columns=["dropin_wilson_low", "dropin_wilson_high"],
        )
        summary["dropin_wilson_low"] = (_di_ci["dropin_wilson_low"] * 100).round(2)
        summary["dropin_wilson_high"] = (_di_ci["dropin_wilson_high"] * 100).round(2)
        summary_columns.append({"name": "dropin_%", "id": "dropin_ratio_pct"})
        summary_columns.append({"name": "dropin_ci_low_%", "id": "dropin_wilson_low"})
        summary_columns.append({"name": "dropin_ci_high_%", "id": "dropin_wilson_high"})

    ab_output = "Pick two policies to compare."
    ab_fig = _empty_figure("Pick two policies for A/B plot")
    ab_quality_fig = _empty_figure("No quality score data for selected A/B policies")
    ab_dropin_fig = _empty_figure("No drop-in ratio data for selected A/B policies")
    ab_violin_fig = _empty_figure("Pick two policies for A/B posterior view")

    if policy_a and policy_b and policy_a in set(metrics["model_name"]) and policy_b in set(metrics["model_name"]):
        row_a = metrics.loc[metrics["model_name"] == policy_a].iloc[0]
        row_b = metrics.loc[metrics["model_name"] == policy_b].iloc[0]

        delta = float(row_b["success_rate"] - row_a["success_rate"])
        delta_low, delta_high = delta_ci_newcombe_wilson(
            int(row_a["successes"]),
            int(row_a["trials"]),
            int(row_b["successes"]),
            int(row_b["trials"]),
            confidence_level,
        )
        _, p_value = two_proportion_p_value(
            int(row_a["successes"]),
            int(row_a["trials"]),
            int(row_b["successes"]),
            int(row_b["trials"]),
            alternative="two-sided",
        )

        if delta_low > 0:
            decision = "B has significantly higher success rate."
            decision_color = "#2e7d32"
            sr_better = True
            sr_worse = False
        elif delta_high < 0:
            decision = "B has significantly lower success rate."
            decision_color = "#c62828"
            sr_better = False
            sr_worse = True
        else:
            decision = "Inconclusive success-rate difference."
            decision_color = "#9e9e9e"
            sr_better = False
            sr_worse = False

        ab_output = html.Div(
            [
                html.Div("Success Rate", style={"fontWeight": "bold", "marginBottom": "4px"}),
                html.Div(
                    f"\u0394 (B \u2212 A): {delta * 100:.2f} pp | "
                    f"{confidence_level * 100:.0f}% CI: [{delta_low * 100:.2f}, {delta_high * 100:.2f}] pp"
                ),
                html.Div(
                    decision,
                    style={"fontWeight": "bold", "marginTop": "4px", "color": decision_color},
                ),
            ],
            style={
                "background": "#fafafa",
                "border": "1px solid #e0e0e0",
                "borderRadius": "6px",
                "padding": "12px 16px",
                "marginTop": "10px",
            },
        )

        ab_policies = [policy_a, policy_b]
        ab_df = metrics.set_index("model_name").loc[ab_policies].reset_index()
        ab_df["success_rate"] = ab_df["successes"] / ab_df["trials"]
        ab_ci = ab_df.apply(
            lambda row: wilson_interval(int(row["successes"]), int(row["trials"]), confidence_level),
            axis=1,
        )
        ab_df["wilson_low"] = ab_ci.apply(lambda x: x[0])
        ab_df["wilson_high"] = ab_ci.apply(lambda x: x[1])

        ab_fig = go.Figure()
        ab_fig.add_bar(
            x=[display_map.get(n, n) for n in ab_df["model_name"]],
            y=(ab_df["success_rate"] * 100),
            marker_color=[policy_colors.get(name, "#1f77b4") for name in ab_df["model_name"]],
            error_y={
                "type": "data",
                "array": ((ab_df["wilson_high"] - ab_df["success_rate"]) * 100),
                "arrayminus": ((ab_df["success_rate"] - ab_df["wilson_low"]) * 100),
                "visible": True,
            },
            text=[f"{rate * 100:.1f}%" for rate in ab_df["success_rate"]],
            textposition="outside",
        )
        ab_fig.update_layout(
            template="plotly_white",
            yaxis_title="Success Rate (%)",
            xaxis_title="Policy",
            title=f"A/B policy comparison with {int(confidence_level * 100)}% Wilson CIs{_prefix_sub}",
            yaxis_range=[0, min(105, max(5, math.ceil((ab_df["wilson_high"].max() * 100) / 5) * 5 + 5))],
        )

        q_better = q_worse = False
        di_better = di_worse = False
        quality_card = None
        dropin_card = None

        if "quality_score_pct" in ab_df.columns and pd.to_numeric(ab_df["quality_score_pct"], errors="coerce").notna().any():
            ab_quality_values = pd.to_numeric(ab_df["quality_score_pct"], errors="coerce")
            ab_has_qstd = (
                "quality_score_std_pct" in ab_df.columns
                and pd.to_numeric(ab_df.get("quality_score_std_pct"), errors="coerce").notna().any()
            )
            ab_quality_fig = go.Figure()
            q_error_y = None
            q_max_y = ab_quality_values.max(skipna=True) or 0
            if ab_has_qstd:
                _abqci = ab_df.apply(
                    lambda row: quality_score_ci(
                        float(row["quality_score_pct"]) if pd.notna(row.get("quality_score_pct")) else math.nan,
                        float(row["quality_score_std_pct"]) if pd.notna(row.get("quality_score_std_pct")) else math.nan,
                        int(row["trials"]),
                        confidence_level,
                    ),
                    axis=1,
                    result_type="expand",
                )
                _abq_lo, _abq_hi = _abqci[0], _abqci[1]
                q_error_y = {
                    "type": "data",
                    "array": (_abq_hi - ab_quality_values).clip(lower=0),
                    "arrayminus": (ab_quality_values - _abq_lo).clip(lower=0),
                    "visible": True,
                }
                q_max_y = max(q_max_y, _abq_hi.max(skipna=True) or 0)
            _qbar_kw: dict = dict(
                x=[display_map.get(n, n) for n in ab_df["model_name"]],
                y=ab_quality_values,
                marker_color=[policy_colors.get(name, "#1f77b4") for name in ab_df["model_name"]],
                text=[f"{val:.1f}%" if pd.notna(val) else "NA" for val in ab_quality_values],
                textposition="outside",
            )
            if q_error_y:
                _qbar_kw["error_y"] = q_error_y
            ab_quality_fig.add_bar(**_qbar_kw)
            ab_quality_fig.update_layout(
                template="plotly_white",
                yaxis_title="Quality Score (%)",
                xaxis_title="Policy",
                title="A/B quality score comparison" + (f" with {int(confidence_level * 100)}% CIs" if ab_has_qstd else "") + _prefix_sub,
                yaxis_range=[0, min(105, max(5, math.ceil(q_max_y / 5) * 5 + 5))],
            )

            # Welch t-test for quality scores
            if ab_has_qstd:
                _ra = ab_df.loc[ab_df["model_name"] == policy_a].iloc[0]
                _rb = ab_df.loc[ab_df["model_name"] == policy_b].iloc[0]
                _t, _pq, _dof, _dcl, _dch = welch_t_test(
                    float(_ra["quality_score_pct"]) if pd.notna(_ra.get("quality_score_pct")) else math.nan,
                    float(_ra["quality_score_std_pct"]) if pd.notna(_ra.get("quality_score_std_pct")) else math.nan,
                    int(_ra["trials"]),
                    float(_rb["quality_score_pct"]) if pd.notna(_rb.get("quality_score_pct")) else math.nan,
                    float(_rb["quality_score_std_pct"]) if pd.notna(_rb.get("quality_score_std_pct")) else math.nan,
                    int(_rb["trials"]),
                    confidence_level,
                )
                if math.isfinite(_pq):
                    _dq = (
                        (float(_rb["quality_score_pct"]) if pd.notna(_rb.get("quality_score_pct")) else 0)
                        - (float(_ra["quality_score_pct"]) if pd.notna(_ra.get("quality_score_pct")) else 0)
                    )
                    if _dcl > 0:
                        _qdec = "B has significantly higher quality."
                        _qcolor = "#2e7d32"
                        q_better = True
                        q_worse = False
                    elif _dch < 0:
                        _qdec = "B has significantly lower quality."
                        _qcolor = "#c62828"
                        q_better = False
                        q_worse = True
                    else:
                        _qdec = "Inconclusive quality difference."
                        _qcolor = "#9e9e9e"
                        q_better = False
                        q_worse = False

                    quality_card = html.Div(
                        [
                            html.Div("Quality Score", style={"fontWeight": "bold", "marginBottom": "4px"}),
                            html.Div(
                                f"\u0394 (B \u2212 A): {_dq:.2f} pp, "
                                f"{confidence_level * 100:.0f}% CI: [{_dcl:.2f}, {_dch:.2f}] pp"
                            ),
                            html.Div(_qdec, style={"fontWeight": "bold", "marginTop": "4px", "color": _qcolor}),
                        ],
                        style={
                            "background": "#fafafa",
                            "border": "1px solid #e0e0e0",
                            "borderRadius": "6px",
                            "padding": "12px 16px",
                            "marginTop": "10px",
                        },
                    )

        # ── Attempt drop-in ratio A/B ──────────────────────────────────
        if "dropin_ratio_pct" in ab_df.columns and pd.to_numeric(ab_df["dropin_ratio_pct"], errors="coerce").notna().any():
            ab_di_values = pd.to_numeric(ab_df["dropin_ratio_pct"], errors="coerce")
            ab_dropin_fig = go.Figure()
            _di_ci_ab = ab_df.apply(
                lambda row: wilson_interval(
                    int(row["dropin_count"]) if pd.notna(row.get("dropin_count")) else 0,
                    int(row["trials"]),
                    confidence_level,
                ),
                axis=1,
                result_type="expand",
            )
            _di_lo_ab, _di_hi_ab = _di_ci_ab[0] * 100, _di_ci_ab[1] * 100
            ab_dropin_fig.add_bar(
                x=[display_map.get(n, n) for n in ab_df["model_name"]],
                y=ab_di_values,
                marker_color=[policy_colors.get(name, "#1f77b4") for name in ab_df["model_name"]],
                error_y={
                    "type": "data",
                    "array": (_di_hi_ab - ab_di_values).clip(lower=0),
                    "arrayminus": (ab_di_values - _di_lo_ab).clip(lower=0),
                    "visible": True,
                },
                text=[f"{val:.1f}%" if pd.notna(val) else "NA" for val in ab_di_values],
                textposition="outside",
            )
            _di_max_y_ab = max(
                ab_di_values.max(skipna=True) or 0,
                _di_hi_ab.max(skipna=True) or 0,
            )
            ab_dropin_fig.update_layout(
                template="plotly_white",
                yaxis_title="Attempt Drop-in Ratio (%)",
                xaxis_title="Policy",
                title=f"A/B attempt drop-in comparison with {int(confidence_level * 100)}% Wilson CIs{_prefix_sub}",
                yaxis_range=[0, min(105, max(5, math.ceil(_di_max_y_ab / 5) * 5 + 5))],
            )

            # Newcombe-Wilson CI for drop-in difference (lower is better)
            _di_a = ab_df.loc[ab_df["model_name"] == policy_a].iloc[0]
            _di_b = ab_df.loc[ab_df["model_name"] == policy_b].iloc[0]
            _di_cnt_b = int(_di_b["dropin_count"]) if pd.notna(_di_b.get("dropin_count")) else 0
            _di_cnt_a = int(_di_a["dropin_count"]) if pd.notna(_di_a.get("dropin_count")) else 0
            _di_dlow, _di_dhigh = delta_ci_newcombe_wilson(
                _di_cnt_a,
                int(_di_a["trials"]),
                _di_cnt_b,
                int(_di_b["trials"]),
                confidence_level,
            )
            _di_delta = (
                (_di_cnt_b / int(_di_b["trials"])) - (_di_cnt_a / int(_di_a["trials"]))
            )
            # Lower is better: delta_high < 0 → B lower (good); delta_low > 0 → B higher (bad)
            if _di_dhigh < 0:
                _didec = "B has significantly lower drop-in ratio (better)."
                _dicolor = "#2e7d32"
                di_better = True
            elif _di_dlow > 0:
                _didec = "B has significantly higher drop-in ratio (worse)."
                _dicolor = "#c62828"
                di_worse = True
            else:
                _didec = "Inconclusive drop-in ratio difference."
                _dicolor = "#9e9e9e"

            dropin_card = html.Div(
                [
                    html.Div("Attempt Drop-in Ratio", style={"fontWeight": "bold", "marginBottom": "4px"}),
                    html.Div(
                        f"\u0394 (B \u2212 A): {_di_delta * 100:.2f} pp, "
                        f"{confidence_level * 100:.0f}% CI: [{_di_dlow * 100:.2f}, {_di_dhigh * 100:.2f}] pp"
                    ),
                    html.Div(_didec, style={"fontWeight": "bold", "marginTop": "4px", "color": _dicolor}),
                ],
                style={
                    "background": "#fafafa",
                    "border": "1px solid #e0e0e0",
                    "borderRadius": "6px",
                    "padding": "12px 16px",
                    "marginTop": "10px",
                },
            )

        # ── Combined verdict ───────────────────────────────────────────
        _has_extra_verdict = quality_card is not None or dropin_card is not None
        if _has_extra_verdict:
            _b_wins: list[str] = []
            _a_wins: list[str] = []

            if sr_better:
                _b_wins.append("success rate")
            elif sr_worse:
                _a_wins.append("success rate")

            if q_better:
                _b_wins.append("quality")
            elif q_worse:
                _a_wins.append("quality")

            if di_better:
                _b_wins.append("drop-in ratio")
            elif di_worse:
                _a_wins.append("drop-in ratio")

            if _b_wins and not _a_wins:
                _ov = f"B is significantly better on {', '.join(_b_wins)}."
                _oc = "#2e7d32"
            elif _a_wins and not _b_wins:
                _ov = f"A is significantly better on {', '.join(_a_wins)}."
                _oc = "#c62828"
            elif _b_wins and _a_wins:
                _ov = f"Trade-off: B is better on {', '.join(_b_wins)} but worse on {', '.join(_a_wins)}."
                _oc = "#e65100"
            else:
                _ov = "No significant difference on any metric."
                _oc = "#9e9e9e"

            overall_card = html.Div(
                html.Div(_ov, style={"fontWeight": "bold", "color": _oc}),
                style={
                    "background": "#f5f5f5",
                    "border": "1px solid #bdbdbd",
                    "borderRadius": "6px",
                    "padding": "10px 16px",
                    "marginTop": "10px",
                },
            )
            _cards = [ab_output]
            if quality_card is not None:
                _cards.append(quality_card)
            if dropin_card is not None:
                _cards.append(dropin_card)
            _cards.append(overall_card)
            ab_output = html.Div(_cards)

        ab_pair_letters = base_vs_policy_letter_pairs(
            ab_df,
            base_policy=policy_a,
            alpha=alpha,
            p_adjust_method=None,
        )
        ab_letters = {policy_a: "a", policy_b: "a"}
        if not ab_pair_letters.empty:
            policy_b_letter = str(ab_pair_letters.iloc[0]["pair_letters"]).split("-")[-1]
            ab_letters[policy_b] = policy_b_letter

        ab_violin_fig = _build_posterior_violin(
            ab_df,
            ab_letters,
            policy_colors,
            f"A/B posterior uncertainty (Bayesian){_prefix_sub}",
            display_names=display_map,
        )

    selected_policies = selected_policies or []
    plot_df = metrics[metrics["model_name"].isin(selected_policies)].copy()

    cld_data = []
    cld_columns = []
    letters = {model: "" for model in plot_df["model_name"]}
    quality_letters: dict[str, str] = {}
    base_policy_for_pairs: str | None = None
    pair_letters_df = pd.DataFrame()
    pair_df_for_pairs = plot_df.copy()

    if not plot_df.empty:
        if policy_a and policy_a in set(metrics["model_name"]):
            base_policy_for_pairs = policy_a
        else:
            base_policy_for_pairs = str(plot_df["model_name"].iloc[0])

        if base_policy_for_pairs not in set(pair_df_for_pairs["model_name"]) and base_policy_for_pairs in set(metrics["model_name"]):
            base_row = metrics.loc[metrics["model_name"] == base_policy_for_pairs]
            pair_df_for_pairs = pd.concat([base_row, pair_df_for_pairs], ignore_index=True)

        pair_df_for_pairs = pair_df_for_pairs.drop_duplicates(subset=["model_name"], keep="first").reset_index(drop=True)

        if len(pair_df_for_pairs) >= 2:
            pair_letters_df = base_vs_policy_letter_pairs(
                pair_df_for_pairs,
                base_policy=base_policy_for_pairs,
                alpha=alpha,
                p_adjust_method=None,
            )

            if not pair_letters_df.empty:
                letters = {model: "" for model in plot_df["model_name"]}
                if base_policy_for_pairs in letters:
                    letters[base_policy_for_pairs] = "a"
                for _, row in pair_letters_df.iterrows():
                    policy_name = str(row["policy"])
                    letter = str(row["pair_letters"]).split("-")[-1]
                    if policy_name in letters:
                        letters[policy_name] = letter

                pair_table = pair_letters_df.copy()
                pair_table["delta_pct"] = (pair_table["delta"] * 100).round(2)
                pair_table["p_value"] = pair_table["p_value"].round(4)
                pair_table["p_value_adj"] = pair_table["p_value_adj"].round(4)

                # Quality Welch tests (base vs each policy)
                _has_q_std = (
                    "quality_score_std_pct" in pair_df_for_pairs.columns
                    and pd.to_numeric(pair_df_for_pairs.get("quality_score_std_pct"), errors="coerce").notna().any()
                )
                if _has_q_std and base_policy_for_pairs:
                    quality_letters[base_policy_for_pairs] = "a"
                    _brow = pair_df_for_pairs.loc[
                        pair_df_for_pairs["model_name"] == base_policy_for_pairs
                    ].iloc[0]
                    _bm = float(_brow["quality_score_pct"]) if pd.notna(_brow.get("quality_score_pct")) else math.nan
                    _bs = float(_brow["quality_score_std_pct"]) if pd.notna(_brow.get("quality_score_std_pct")) else math.nan
                    _bn = int(_brow["trials"])
                    q_deltas, q_pvals, q_ltrs = [], [], []
                    for _, prow in pair_table.iterrows():
                        _pname = str(prow["policy"])
                        _pr = pair_df_for_pairs.loc[pair_df_for_pairs["model_name"] == _pname]
                        if _pr.empty:
                            q_deltas.append(math.nan)
                            q_pvals.append(math.nan)
                            q_ltrs.append("?")
                            continue
                        _pr = _pr.iloc[0]
                        _pm = float(_pr["quality_score_pct"]) if pd.notna(_pr.get("quality_score_pct")) else math.nan
                        _ps = float(_pr["quality_score_std_pct"]) if pd.notna(_pr.get("quality_score_std_pct")) else math.nan
                        _pn = int(_pr["trials"])
                        _, _pv, _, _, _ = welch_t_test(_bm, _bs, _bn, _pm, _ps, _pn, confidence_level)
                        _is_sig = _pv < alpha if math.isfinite(_pv) else False
                        _ql = "a-b" if _is_sig else "a-a"
                        q_deltas.append(round(_pm - _bm, 2) if math.isfinite(_pm) and math.isfinite(_bm) else math.nan)
                        q_pvals.append(round(_pv, 4) if math.isfinite(_pv) else math.nan)
                        q_ltrs.append(_ql)
                        quality_letters[_pname] = _ql.split("-")[-1]
                    pair_table["quality_delta_pct"] = q_deltas
                    pair_table["quality_p_value"] = q_pvals
                    pair_table["quality_pair_letters"] = q_ltrs

                table_cols = ["base_policy", "policy", "delta_pct", "pair_letters"]
                if _has_q_std:
                    table_cols.extend(["quality_delta_pct", "quality_pair_letters"])
                pair_table = pair_table[[c for c in table_cols if c in pair_table.columns]]

                # Apply display names to table
                pair_table["base_policy"] = pair_table["base_policy"].map(lambda n: display_map.get(str(n), str(n)))
                pair_table["policy"] = pair_table["policy"].map(lambda n: display_map.get(str(n), str(n)))

                _COL_DISPLAY_NAMES = {
                    "base_policy": "Base",
                    "policy": "Policy",
                    "delta_pct": "\u0394 SR (pp)",
                    "pair_letters": "SR Letters",
                    "quality_delta_pct": "\u0394 Quality (pp)",
                    "quality_pair_letters": "Quality Letters",
                }
                cld_data = pair_table.to_dict("records")
                cld_columns = [
                    {"name": _COL_DISPLAY_NAMES.get(c, c), "id": c}
                    for c in pair_table.columns
                ]

    if plot_df.empty:
        fig = _empty_figure("No selected policies to plot")
        quality_fig = _empty_figure("No quality score data for selected policies")
        dropin_fig = _empty_figure("No drop-in ratio data for selected policies")
        sr_vs_q_fig = _empty_figure("No data for scatter view")
        if show_final_violin:
            violin_fig = _empty_figure("No selected policies for base-vs-policy violin")
        else:
            violin_fig = go.Figure()
    else:
        plot_df["success_rate"] = plot_df["successes"] / plot_df["trials"]
        plot_ci = plot_df.apply(
            lambda row: wilson_interval(int(row["successes"]), int(row["trials"]), confidence_level),
            axis=1,
        )
        plot_df["wilson_low"] = plot_ci.apply(lambda x: x[0])
        plot_df["wilson_high"] = plot_ci.apply(lambda x: x[1])

        fig = go.Figure()
        fig.add_bar(
            x=[display_map.get(n, n) for n in plot_df["model_name"]],
            y=(plot_df["success_rate"] * 100),
            marker_color=[policy_colors.get(name, "#1f77b4") for name in plot_df["model_name"]],
            error_y={
                "type": "data",
                "array": ((plot_df["wilson_high"] - plot_df["success_rate"]) * 100),
                "arrayminus": ((plot_df["success_rate"] - plot_df["wilson_low"]) * 100),
                "visible": True,
            },
            text=[
                f"{rate * 100:.1f}%" + (f" | {letters.get(model, '')}" if letters.get(model, "") else "")
                for model, rate in zip(plot_df["model_name"], plot_df["success_rate"])
            ],
            textposition="outside",
        )
        fig.update_layout(
            template="plotly_white",
            yaxis_title="Success Rate (%)",
            xaxis_title="Policy",
            title=f"Selected policies with {int(confidence_level * 100)}% Wilson CIs{_multi_sub}",
            yaxis_range=[0, min(105, max(5, math.ceil((plot_df["wilson_high"].max() * 100) / 5) * 5 + 5))],
        )

        quality_values = pd.to_numeric(plot_df.get("quality_score_pct"), errors="coerce")
        if quality_values.notna().any():
            quality_fig = go.Figure()
            mp_has_q_std = (
                "quality_score_std_pct" in plot_df.columns
                and pd.to_numeric(plot_df.get("quality_score_std_pct"), errors="coerce").notna().any()
            )
            mp_error_y = None
            mp_q_max = quality_values.max(skipna=True) or 0
            if mp_has_q_std:
                _mpqci = plot_df.apply(
                    lambda row: quality_score_ci(
                        float(row["quality_score_pct"]) if pd.notna(row.get("quality_score_pct")) else math.nan,
                        float(row["quality_score_std_pct"]) if pd.notna(row.get("quality_score_std_pct")) else math.nan,
                        int(row["trials"]),
                        confidence_level,
                    ),
                    axis=1,
                    result_type="expand",
                )
                _mpq_lo, _mpq_hi = _mpqci[0], _mpqci[1]
                mp_error_y = {
                    "type": "data",
                    "array": (_mpq_hi - quality_values).clip(lower=0),
                    "arrayminus": (quality_values - _mpq_lo).clip(lower=0),
                    "visible": True,
                }
                mp_q_max = max(mp_q_max, _mpq_hi.max(skipna=True) or 0)
            bar_text = []
            for model, val in zip(plot_df["model_name"], quality_values):
                label = f"{val:.1f}%" if pd.notna(val) else "NA"
                ql = quality_letters.get(model, "")
                if ql:
                    label += f" | {ql}"
                bar_text.append(label)
            _mpq_kw: dict = dict(
                x=[display_map.get(n, n) for n in plot_df["model_name"]],
                y=quality_values,
                marker_color=[policy_colors.get(name, "#1f77b4") for name in plot_df["model_name"]],
                text=bar_text,
                textposition="outside",
            )
            if mp_error_y:
                _mpq_kw["error_y"] = mp_error_y
            quality_fig.add_bar(**_mpq_kw)
            quality_fig.update_layout(
                template="plotly_white",
                yaxis_title="Quality Score (%)",
                xaxis_title="Policy",
                title="Selected policies quality score" + (f" with {int(confidence_level * 100)}% CIs" if mp_has_q_std else "") + _multi_sub,
                yaxis_range=[0, min(105, max(5, math.ceil(mp_q_max / 5) * 5 + 5))],
            )
        else:
            quality_fig = _empty_figure("No quality score data for selected policies")

        # ── Multi-policy drop-in ratio bar chart ──────────────────────
        _mp_di_values = pd.to_numeric(plot_df.get("dropin_ratio_pct"), errors="coerce")
        if _mp_di_values.notna().any():
            dropin_fig = go.Figure()
            _mp_di_ci = plot_df.apply(
                lambda row: wilson_interval(
                    int(row["dropin_count"]) if pd.notna(row.get("dropin_count")) else 0,
                    int(row["trials"]),
                    confidence_level,
                ),
                axis=1,
                result_type="expand",
            )
            _mp_di_lo, _mp_di_hi = _mp_di_ci[0] * 100, _mp_di_ci[1] * 100
            _mp_di_max = max(
                _mp_di_values.max(skipna=True) or 0,
                _mp_di_hi.max(skipna=True) or 0,
            )
            dropin_fig.add_bar(
                x=[display_map.get(n, n) for n in plot_df["model_name"]],
                y=_mp_di_values,
                marker_color=[policy_colors.get(name, "#1f77b4") for name in plot_df["model_name"]],
                error_y={
                    "type": "data",
                    "array": (_mp_di_hi - _mp_di_values).clip(lower=0),
                    "arrayminus": (_mp_di_values - _mp_di_lo).clip(lower=0),
                    "visible": True,
                },
                text=[f"{val:.1f}%" if pd.notna(val) else "NA" for val in _mp_di_values],
                textposition="outside",
            )
            dropin_fig.update_layout(
                template="plotly_white",
                yaxis_title="Attempt Drop-in Ratio (%)",
                xaxis_title="Policy",
                title=f"Selected policies attempt drop-in with {int(confidence_level * 100)}% Wilson CIs{_multi_sub}",
                yaxis_range=[0, min(105, max(5, math.ceil(_mp_di_max / 5) * 5 + 5))],
            )
        else:
            dropin_fig = _empty_figure("No drop-in ratio data for selected policies")

        # SR vs Quality scatter plot
        _q_vals_scatter = pd.to_numeric(plot_df.get("quality_score_pct"), errors="coerce")
        if _q_vals_scatter.notna().any():
            sr_vs_q_fig = go.Figure()
            _mp_has_qstd = (
                "quality_score_std_pct" in plot_df.columns
                and pd.to_numeric(plot_df.get("quality_score_std_pct"), errors="coerce").notna().any()
            )
            for _, _srow in plot_df.iterrows():
                _sname = str(_srow["model_name"])
                _sshort = display_map.get(_sname, _sname)
                _sr = float(_srow["success_rate"]) * 100
                _qv = float(_srow.get("quality_score_pct", math.nan)) if pd.notna(_srow.get("quality_score_pct")) else math.nan
                if not math.isfinite(_qv):
                    continue
                _sc = policy_colors.get(_sname, "#1f77b4")
                _ex = dict(
                    type="data",
                    array=[float(_srow["wilson_high"] - _srow["success_rate"]) * 100],
                    arrayminus=[float(_srow["success_rate"] - _srow["wilson_low"]) * 100],
                    visible=True,
                )
                _trace_kw: dict = dict(
                    x=[_sr],
                    y=[_qv],
                    mode="markers+text",
                    marker=dict(color=_sc, size=12),
                    text=[_sshort],
                    textposition="top center",
                    name=_sshort,
                    showlegend=False,
                    error_x=_ex,
                )
                if _mp_has_qstd and pd.notna(_srow.get("quality_score_std_pct")):
                    _qci_lo, _qci_hi = quality_score_ci(
                        float(_srow["quality_score_pct"]),
                        float(_srow["quality_score_std_pct"]),
                        int(_srow["trials"]),
                        confidence_level,
                    )
                    if math.isfinite(_qci_lo) and math.isfinite(_qci_hi):
                        _trace_kw["error_y"] = dict(
                            type="data",
                            array=[_qci_hi - _qv],
                            arrayminus=[_qv - _qci_lo],
                            visible=True,
                        )
                sr_vs_q_fig.add_trace(go.Scatter(**_trace_kw))
            sr_vs_q_fig.update_layout(
                template="plotly_white",
                xaxis_title="Success Rate (%)",
                yaxis_title="Quality Score (%)",
                title=f"Success Rate vs Quality Score{_multi_sub}",
            )
        else:
            sr_vs_q_fig = _empty_figure("No quality data for scatter view")

        if show_final_violin:
            violin_fig = _build_base_vs_pairs_violin(
                pair_df_for_pairs,
                base_policy=base_policy_for_pairs or str(plot_df["model_name"].iloc[0]),
                pair_letters_df=pair_letters_df,
                policy_colors=policy_colors,
                display_names=display_map,
                title_suffix=_multi_sub,
            )
        else:
            violin_fig = go.Figure()

    sort_status = _format_sort_status(sort_mode, prefix=prefix, active_group=active_group_label)

    # Conditional row styling: highlight significant comparisons
    cld_row_styles: list[dict] = []
    if cld_data:
        for idx, rec in enumerate(cld_data):
            is_sr_sig = str(rec.get("pair_letters", "")) == "a-b"
            is_q_sig = str(rec.get("quality_pair_letters", "")) == "a-b"
            if is_sr_sig and is_q_sig:
                cld_row_styles.append(
                    {"if": {"row_index": idx}, "backgroundColor": "#e8f5e9", "fontWeight": "500"}
                )
            elif is_sr_sig or is_q_sig:
                cld_row_styles.append(
                    {"if": {"row_index": idx}, "backgroundColor": "#fff8e1"}
                )

    # All-vs-all CLD
    allvsall_data: list[dict] = []
    allvsall_columns: list[dict] = []
    allvsall_styles: list[dict] = []
    allvsall_violin_fig = _empty_figure("Select at least 2 policies for all-vs-all violin")
    if not plot_df.empty and len(plot_df) >= 2:
        cld_letters, _cld_tests = compact_letter_display(plot_df, alpha=alpha)
        # Build the table: one row per policy with SR, Quality, CLD group
        _ava_rows: list[dict] = []
        for _, _row in plot_df.iterrows():
            _name = str(_row["model_name"])
            _sr = float(_row["successes"]) / float(_row["trials"]) * 100 if float(_row["trials"]) > 0 else 0
            _entry: dict = {
                "policy": display_map.get(_name, _name),
                "success_rate_pct": round(_sr, 2),
                "sr_group": cld_letters.get(_name, ""),
            }
            if "quality_score_pct" in _row.index and pd.notna(_row.get("quality_score_pct")):
                _entry["quality_pct"] = round(float(_row["quality_score_pct"]), 2)
            else:
                _entry["quality_pct"] = None
            _ava_rows.append(_entry)

        allvsall_data = _ava_rows
        _ava_cols = ["policy", "success_rate_pct", "sr_group"]
        _has_quality_in_ava = any(r.get("quality_pct") is not None for r in _ava_rows)
        if _has_quality_in_ava:
            _ava_cols.append("quality_pct")
        _AVA_COL_NAMES = {
            "policy": "Policy",
            "success_rate_pct": "SR (%)",
            "sr_group": "SR Group",
            "quality_pct": "Quality (%)",
        }
        allvsall_columns = [{"name": _AVA_COL_NAMES.get(c, c), "id": c} for c in _ava_cols]

        # Determine distinct groups and count them
        _unique_groups = sorted({v for v in cld_letters.values() if v})
        _n_groups = len(_unique_groups)
        # Color-code rows by group membership
        _group_colors = ["#e3f2fd", "#fce4ec", "#e8f5e9", "#fff3e0", "#f3e5f5", "#e0f7fa"]
        if _n_groups > 1:
            _group_color_map = {g: _group_colors[i % len(_group_colors)] for i, g in enumerate(_unique_groups)}
            for idx, rec in enumerate(allvsall_data):
                grp = rec.get("sr_group", "")
                if grp in _group_color_map:
                    allvsall_styles.append(
                        {"if": {"row_index": idx}, "backgroundColor": _group_color_map[grp]}
                    )

        # All-vs-all violin
        if show_allvsall_violin:
            allvsall_violin_fig = _build_posterior_violin(
                plot_df,
                cld_letters,
                policy_colors,
                f"All-vs-all posterior uncertainty (Bayesian CLD){_multi_sub}",
                display_names=display_map,
            )
        else:
            allvsall_violin_fig = go.Figure()

    return (
        summary.to_dict("records"),
        summary_columns,
        ab_output,
        ab_fig,
        ab_quality_fig,
        ab_dropin_fig,
        ab_violin_fig,
        fig,
        quality_fig,
        dropin_fig,
        sr_vs_q_fig,
        final_violin_style,
        violin_fig,
        cld_data,
        cld_columns,
        cld_row_styles,
        allvsall_data,
        allvsall_columns,
        allvsall_styles,
        allvsall_violin_style,
        allvsall_violin_fig,
        sort_status,
    )


if __name__ == "__main__":
    app.run(debug=True)
