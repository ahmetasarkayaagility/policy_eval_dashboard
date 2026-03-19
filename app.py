from __future__ import annotations

import functools
import hashlib
import math
import re
import time

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
    EVAL_DETAILS_COLUMN_CANDIDATES,
    EVAL_DETAILS_URL_COLUMN_CANDIDATES,
    MODEL_COLUMN_CANDIDATES,
    extract_url_from_cell_value,
    find_column as _find_column,
    get_google_auth_status,
    list_google_spreadsheet_sheets,
    list_local_spreadsheet_sheets,
    load_google_spreadsheet,
    load_local_spreadsheet,
    normalize_policy_dataframe,
    percent_like_to_numeric as _percent_like_to_numeric,
    promote_header_row_if_needed,
    to_percent_points as _to_percent_points,
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
    "dropin_ratio": "Sorted by Attempt Drop-in [%] \u2193 (higher/worse first)",
    "dropin_ratio_asc": "Sorted by Attempt Drop-in [%] \u2191 (lower/better first)",
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

FAILURE_CONDITION_ORDER_OPTIONS = [
    {"label": "Original spreadsheet order (default)", "value": "original"},
    {"label": "Sort by failure severity", "value": "failure"},
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
FAILURE_LINK_COLUMN_CANDIDATES = list(
    dict.fromkeys([*EVAL_DETAILS_URL_COLUMN_CANDIDATES, *EVAL_DETAILS_COLUMN_CANDIDATES])
)
SAFE_POLICY_NAME_COLUMN_CANDIDATES = ["model_name", *MODEL_COLUMN_CANDIDATES]


def _default_columns() -> list[dict[str, str | bool]]:
    return [
        {"name": "model_name", "id": "model_name", "editable": True},
        {"name": "successes", "id": "successes", "editable": True},
        {"name": "trials", "id": "trials", "editable": True},
        {"name": "notes", "id": "notes", "editable": True},
    ]


def _normalize_group_tag(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _is_base_group_tag(value: object) -> bool:
    token = _normalize_group_tag(value)
    return bool(token) and token in BASE_GROUP_TAGS


def _split_testing_group_tags(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []

    parts = [part.strip() for part in re.split(r"[\n,;|/]+", text)]
    tags: list[str] = []
    seen: set[str] = set()
    for part in parts:
        clean = re.sub(r"\s+", " ", part).strip()
        if not clean:
            continue
        token = clean.lower()
        if token in {"nan", "none"} or token in seen:
            continue
        tags.append(clean)
        seen.add(token)
    return tags


def _merge_unique_group_tags(series: pd.Series) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()

    for value in series:
        tags = value if isinstance(value, list) else _split_testing_group_tags(value)
        for tag in tags:
            token = _normalize_group_tag(tag)
            if not token or token in seen:
                continue
            merged.append(tag)
            seen.add(token)

    return merged


def _normalize_group_selection(value: object) -> list[str]:
    """Normalize a single/multi-select UI value into unique display-preserving tags."""
    if isinstance(value, list):
        candidates = value
    elif value is None or (isinstance(value, float) and pd.isna(value)):
        candidates = []
    else:
        candidates = [value]

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate).strip()
        if not text:
            continue
        token = _normalize_group_tag(text)
        if not token or token in seen:
            continue
        out.append(text)
        seen.add(token)
    return out


def _select_models_for_group_tags(
    group_names: list[str],
    base_models: list[str],
    group_to_models: dict[str, list[str]],
    ordered_models: list[str],
) -> list[str]:
    """Return models matching selected tags plus base rows, preserving source order."""
    selected = set(base_models)
    for group_name in group_names:
        selected.update(group_to_models.get(group_name, []))
    return [model for model in ordered_models if model in selected]


def _build_testing_group_index(
    entries: list[tuple[str, list[str], bool]],
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Build ordered testing-group index: tags, tag→models map, and base models."""
    tag_order: list[str] = []
    group_to_models: dict[str, list[str]] = {}
    base_models: list[str] = []
    base_seen: set[str] = set()

    for model_name, tags, row_is_base in entries:
        if row_is_base and model_name not in base_seen:
            base_models.append(model_name)
            base_seen.add(model_name)

        for tag in tags:
            if _is_base_group_tag(tag):
                continue
            if tag not in group_to_models:
                group_to_models[tag] = []
                tag_order.append(tag)
            if model_name not in group_to_models[tag]:
                group_to_models[tag].append(model_name)

    return tag_order, group_to_models, base_models


def _build_testing_group_index_for_models(
    clean_df: pd.DataFrame,
    model_names: list[str],
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Build testing-group index for a filtered model set from normalized rows."""
    tag_df = clean_df.copy()
    if not tag_df.empty:
        tag_df = tag_df[tag_df["model_name"].astype(str).isin(model_names)].copy()
        tag_df = tag_df.sort_values("source_order", kind="stable")

    if "testing_group_tags" not in tag_df.columns:
        if "testing_group" in tag_df.columns:
            tag_df["testing_group_tags"] = tag_df["testing_group"].map(_split_testing_group_tags)
        else:
            tag_df["testing_group_tags"] = [[] for _ in range(len(tag_df))]
    if "is_base_group" not in tag_df.columns:
        tag_df["is_base_group"] = False
    tag_df["is_base_group"] = tag_df["is_base_group"].fillna(False).astype(bool)

    group_entries: list[tuple[str, list[str], bool]] = []
    for _, row in tag_df.iterrows():
        model_name = str(row.get("model_name", "")).strip()
        if not model_name:
            continue

        tags = row.get("testing_group_tags")
        if not isinstance(tags, list):
            tags = _split_testing_group_tags(tags)

        row_is_base = bool(row.get("is_base_group", False)) or any(_is_base_group_tag(tag) for tag in tags)
        group_entries.append((model_name, tags, row_is_base))

    return _build_testing_group_index(group_entries)


def _resolve_group_dropdown_state(
    trigger: str,
    selected_groups: list[str],
    active_groups: list[str],
    tag_options: list[dict[str, str]],
    clear_trigger_id: str,
) -> tuple[list[str], bool, list[str]]:
    """Resolve testing-group dropdown value/disabled state and active-group reset rules."""
    dropdown_disabled = len(tag_options) == 0
    if dropdown_disabled:
        return [], True, []
    if trigger == clear_trigger_id:
        return [], False, active_groups
    if selected_groups:
        return selected_groups, False, active_groups
    if active_groups:
        return active_groups, False, active_groups
    return [], False, active_groups


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


@functools.lru_cache(maxsize=1024)
def _policy_color(policy_name: str) -> str:
    digest = hashlib.md5(policy_name.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(POLICY_COLOR_PALETTE)
    return POLICY_COLOR_PALETTE[index]


def _policy_color_map(policy_names: list[str]) -> dict[str, str]:
    unique_names = sorted({str(name) for name in policy_names})
    return {name: _policy_color(name) for name in unique_names}


def _raw_to_clean_df(raw_records: list[dict] | None) -> pd.DataFrame:
    """Coerce raw table rows into canonical policy metrics used by all dashboards."""
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
                "testing_group_tags",
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
        df["testing_group_tags"] = df[testing_group_col].map(_split_testing_group_tags)
    else:
        df["testing_group_tags"] = [[] for _ in range(len(df))]

    df["testing_group"] = df["testing_group_tags"].map(lambda tags: " | ".join(tags) if tags else pd.NA)
    df["is_base_group"] = df["testing_group_tags"].map(
        lambda tags: any(_is_base_group_tag(tag) for tag in tags)
    )

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
                "testing_group_tags": _merge_unique_group_tags,
                "is_base_group": "any",
            }
        )
        .sort_values("source_order", kind="stable")
        .reset_index(drop=True)
    )

    grouped["testing_group"] = grouped["testing_group_tags"].map(
        lambda tags: " | ".join(tags) if isinstance(tags, list) and tags else pd.NA
    )
    grouped["is_base_group"] = grouped.apply(
        lambda row: bool(row.get("is_base_group", False))
        or any(_is_base_group_tag(tag) for tag in (row.get("testing_group_tags") or [])),
        axis=1,
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


def _format_sort_status(
    sort_mode: str | None,
    prefix: str = "",
    active_group: str | list[str] | None = None,
) -> str:
    status = f"Order mode: {SORT_MODE_LABELS.get(sort_mode or 'original', 'Original sheet order')}"
    active_groups = _normalize_group_selection(active_group)
    if active_groups:
        noun = "testing groups" if len(active_groups) > 1 else "testing group"
        status += f" | {noun}: {', '.join(active_groups)} + Base"
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
        violingap=0.0,
        violingroupgap=0.0,
        yaxis_title="Success Rate (%)",
        xaxis_title="Policy",
        title=title,
        yaxis_range=[0, 105],
        annotations=annotations,
    )
    fig.update_traces(width=0.95, selector={"type": "violin"})
    return fig


def _build_quality_posterior_violin(
    plot_df: pd.DataFrame,
    letters: dict[str, str] | None,
    policy_colors: dict[str, str],
    title: str,
    display_names: dict[str, str] | None = None,
) -> go.Figure:
    if plot_df.empty:
        return _empty_figure("No selected policies for quality uncertainty view")
    if "quality_score_pct" not in plot_df.columns or "quality_score_std_pct" not in plot_df.columns:
        return _empty_figure("No quality-score uncertainty data for selected policies")

    _dn = display_names or {}
    rng = np.random.default_rng(20260313)
    n_samples = 1200

    fig = go.Figure()
    for _, row in plot_df.reset_index(drop=True).iterrows():
        model_name = str(row["model_name"])
        short = _dn.get(model_name, model_name)
        color = policy_colors.get(model_name, "#1f77b4")

        mean_pct = _coerce_scalar_float(row.get("quality_score_pct"))
        std_pct = _coerce_scalar_float(row.get("quality_score_std_pct"))
        trials = int(row.get("trials", 0)) if pd.notna(row.get("trials")) else 0
        if mean_pct is None or std_pct is None or std_pct < 0 or trials <= 1:
            continue

        mean_pct = max(0.0, min(100.0, mean_pct))
        standard_error = max(1e-6, std_pct / math.sqrt(trials))
        posterior_samples = rng.normal(loc=mean_pct, scale=standard_error, size=n_samples)
        posterior_samples = np.clip(posterior_samples, 0.0, 100.0)

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

    if not fig.data:
        return _empty_figure("No quality-score uncertainty data for selected policies")

    letter_map = letters or {}
    annotations = []
    for model_name in plot_df["model_name"].astype(str).tolist():
        short = _dn.get(model_name, model_name)
        group_letters = letter_map.get(model_name, "")
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
        violingap=0.0,
        violingroupgap=0.0,
        yaxis_title="Quality Score (%)",
        xaxis_title="Policy",
        title=title,
        yaxis_range=[0, 105],
        annotations=annotations,
    )
    fig.update_traces(width=0.95, selector={"type": "violin"})
    return fig


def _build_dropin_posterior_violin(
    plot_df: pd.DataFrame,
    letters: dict[str, str] | None,
    policy_colors: dict[str, str],
    title: str,
    display_names: dict[str, str] | None = None,
) -> go.Figure:
    if plot_df.empty:
        return _empty_figure("No selected policies for drop-in uncertainty view")
    if "dropin_count" not in plot_df.columns:
        return _empty_figure("No drop-in uncertainty data for selected policies")

    _dn = display_names or {}
    rng = np.random.default_rng(20260314)
    n_samples = 1200
    prior_alpha = 1.0
    prior_beta = 1.0

    fig = go.Figure()
    for _, row in plot_df.reset_index(drop=True).iterrows():
        model_name = str(row["model_name"])
        short = _dn.get(model_name, model_name)
        color = policy_colors.get(model_name, "#1f77b4")

        trials = int(row.get("trials", 0)) if pd.notna(row.get("trials")) else 0
        dropin_count = int(row.get("dropin_count", 0)) if pd.notna(row.get("dropin_count")) else 0
        if trials <= 0:
            continue

        dropin_count = max(0, min(dropin_count, trials))
        non_dropin_count = max(0, trials - dropin_count)
        posterior_samples = rng.beta(prior_alpha + dropin_count, prior_beta + non_dropin_count, size=n_samples) * 100.0

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

    if not fig.data:
        return _empty_figure("No drop-in uncertainty data for selected policies")

    letter_map = letters or {}
    annotations = []
    for model_name in plot_df["model_name"].astype(str).tolist():
        short = _dn.get(model_name, model_name)
        group_letters = letter_map.get(model_name, "")
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
        violingap=0.0,
        violingroupgap=0.0,
        yaxis_title="Attempt Drop-in Ratio (%) (lower is better)",
        xaxis_title="Policy",
        title=title,
        yaxis_range=[0, 105],
        annotations=annotations,
    )
    fig.update_traces(width=0.95, selector={"type": "violin"})
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
        violingap=0.0,
        violingroupgap=0.0,
        yaxis_title="Success Rate (%)",
        xaxis_title="Base-vs-policy pair",
        title=f"Base-vs-policy posterior uncertainty with pair letters{title_suffix}",
        yaxis_range=[0, 105],
        annotations=annotations,
    )
    fig.update_traces(width=0.48, selector={"type": "violin"})
    return fig


def _build_base_vs_quality_pairs_violin(
    plot_df: pd.DataFrame,
    base_policy: str,
    quality_pair_letters_df: pd.DataFrame,
    policy_colors: dict[str, str],
    display_names: dict[str, str] | None = None,
    title_suffix: str = "",
) -> go.Figure:
    if plot_df.empty:
        return _empty_figure("No selected policies for base-vs-policy quality violin")
    if base_policy not in set(plot_df["model_name"].astype(str)):
        return _empty_figure("Base policy is not in selected policies")
    if quality_pair_letters_df.empty:
        return _empty_figure("No quality-score pair letters available for selected base-vs-policy pairs")
    if "quality_score_pct" not in plot_df.columns or "quality_score_std_pct" not in plot_df.columns:
        return _empty_figure("Quality score mean and STD are required for quality base-vs-policy violin")

    _dn = display_names or {}
    rng = np.random.default_rng(20260311)
    n_samples = 1000

    row_by_model = plot_df.set_index("model_name")
    fig = go.Figure()
    shown_legend: set[str] = set()
    annotations: list[dict[str, object]] = []

    for _, row in quality_pair_letters_df.reset_index(drop=True).iterrows():
        other_policy = str(row["policy"])
        base_short = _dn.get(base_policy, base_policy)
        other_short = _dn.get(other_policy, other_policy)
        pair_label = f"{base_short} vs {other_short}"

        pair_has_trace = False
        for policy_name in [base_policy, other_policy]:
            if policy_name not in row_by_model.index:
                continue

            record = row_by_model.loc[policy_name]
            mean_pct = _coerce_scalar_float(record.get("quality_score_pct"))
            std_pct = _coerce_scalar_float(record.get("quality_score_std_pct"))
            trials = int(record.get("trials", 0)) if pd.notna(record.get("trials")) else 0

            if mean_pct is None or std_pct is None or trials <= 1 or std_pct < 0:
                continue

            mean_pct = max(0.0, min(100.0, mean_pct))
            standard_error = max(1e-6, std_pct / math.sqrt(trials))
            quality_samples = rng.normal(loc=mean_pct, scale=standard_error, size=n_samples)
            quality_samples = np.clip(quality_samples, 0.0, 100.0)

            short = _dn.get(policy_name, policy_name)
            color = policy_colors.get(policy_name, "#1f77b4")
            show_legend = policy_name not in shown_legend
            shown_legend.add(policy_name)

            fig.add_trace(
                go.Violin(
                    x=[pair_label] * n_samples,
                    y=quality_samples,
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
            pair_has_trace = True

        if pair_has_trace:
            annotations.append(
                {
                    "x": pair_label,
                    "y": 103,
                    "text": f"<b>{row['pair_letters']}</b>",
                    "showarrow": False,
                    "font": {"size": 15},
                }
            )

    if not fig.data:
        return _empty_figure("No valid quality-score STD rows available for selected base-vs-policy pairs")

    fig.update_layout(
        template="plotly_white",
        violinmode="group",
        violingap=0.0,
        violingroupgap=0.0,
        yaxis_title="Quality Score (%)",
        xaxis_title="Base-vs-policy pair",
        title=f"Base-vs-policy quality uncertainty with pair letters{title_suffix}",
        yaxis_range=[0, 105],
        annotations=annotations,
    )
    fig.update_traces(width=0.48, selector={"type": "violin"})
    return fig


def _build_base_vs_dropin_pairs_violin(
    plot_df: pd.DataFrame,
    base_policy: str,
    dropin_pair_letters_df: pd.DataFrame,
    policy_colors: dict[str, str],
    display_names: dict[str, str] | None = None,
    title_suffix: str = "",
) -> go.Figure:
    if plot_df.empty:
        return _empty_figure("No selected policies for base-vs-policy drop-in violin")
    if base_policy not in set(plot_df["model_name"].astype(str)):
        return _empty_figure("Base policy is not in selected policies")
    if dropin_pair_letters_df.empty:
        return _empty_figure("No drop-in pair letters available for selected base-vs-policy pairs")
    if "dropin_count" not in plot_df.columns:
        return _empty_figure("Drop-in count values are required for drop-in base-vs-policy violin")

    _dn = display_names or {}
    rng = np.random.default_rng(20260312)
    n_samples = 1000
    prior_alpha = 1.0
    prior_beta = 1.0

    row_by_model = plot_df.set_index("model_name")
    fig = go.Figure()
    shown_legend: set[str] = set()
    annotations: list[dict[str, object]] = []

    for _, row in dropin_pair_letters_df.reset_index(drop=True).iterrows():
        other_policy = str(row["policy"])
        base_short = _dn.get(base_policy, base_policy)
        other_short = _dn.get(other_policy, other_policy)
        pair_label = f"{base_short} vs {other_short}"

        pair_has_trace = False
        for policy_name in [base_policy, other_policy]:
            if policy_name not in row_by_model.index:
                continue

            record = row_by_model.loc[policy_name]
            trials = int(record.get("trials", 0)) if pd.notna(record.get("trials")) else 0
            dropin_count = int(record.get("dropin_count", 0)) if pd.notna(record.get("dropin_count")) else 0
            if trials <= 0:
                continue

            dropin_count = max(0, min(dropin_count, trials))
            non_dropin_count = max(0, trials - dropin_count)
            posterior_samples = (
                rng.beta(prior_alpha + dropin_count, prior_beta + non_dropin_count, size=n_samples) * 100.0
            )

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
            pair_has_trace = True

        if pair_has_trace:
            annotations.append(
                {
                    "x": pair_label,
                    "y": 103,
                    "text": f"<b>{row['pair_letters']}</b>",
                    "showarrow": False,
                    "font": {"size": 15},
                }
            )

    if not fig.data:
        return _empty_figure("No valid drop-in rows available for selected base-vs-policy pairs")

    fig.update_layout(
        template="plotly_white",
        violinmode="group",
        violingap=0.0,
        violingroupgap=0.0,
        yaxis_title="Attempt Drop-in Ratio (%) (lower is better)",
        xaxis_title="Base-vs-policy pair",
        title=f"Base-vs-policy drop-in uncertainty with pair letters (lower is better){title_suffix}",
        yaxis_range=[0, 105],
        annotations=annotations,
    )
    fig.update_traces(width=0.48, selector={"type": "violin"})
    return fig


def _format_numeric_token(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return str(int(rounded))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _coerce_scalar_float(value: object) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return 1.0 if value else 0.0

    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return None
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    clean = text.replace(",", "")
    if clean.endswith("%"):
        clean = clean[:-1].strip()
    if not clean:
        return None

    try:
        return float(clean)
    except ValueError:
        return None


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
            parsed = _coerce_scalar_float(part)
            if parsed is None or math.isnan(parsed):
                return text
            parsed_parts.append(_format_numeric_token(parsed))
        return "[" + ", ".join(parsed_parts) + "]"

    parsed = _coerce_scalar_float(text)
    if parsed is not None and not math.isnan(parsed):
        return _format_numeric_token(parsed)

    return text


def _condition_sort_key(label: str) -> tuple:
    if label == "NA":
        return 3, ""

    bracket_match = re.match(r"^\[\s*(.*?)\s*\]$", str(label))
    if bracket_match:
        raw_parts = [part.strip() for part in bracket_match.group(1).split(",")]
        numbers: list[float] = []
        for part in raw_parts:
            parsed = _coerce_scalar_float(part)
            if parsed is None or math.isnan(parsed):
                numbers = []
                break
            numbers.append(parsed)
        if numbers:
            return 0, len(numbers), tuple(numbers)

    parsed = _coerce_scalar_float(label)
    if parsed is not None and not math.isnan(parsed):
        return 1, parsed

    return 2, str(label).lower()


def _condition_key(column_name: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", column_name.strip().lower()).strip("_")
    token = token or "condition"
    digest = hashlib.md5(column_name.strip().lower().encode("utf-8")).hexdigest()[:8]
    return f"cond__{token}_{digest}"


def _safe_policy_name_from_row(row: dict[str, object]) -> str:
    for key in SAFE_POLICY_NAME_COLUMN_CANDIDATES:
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

    parsed = _coerce_scalar_float(value)
    if parsed is None or not math.isfinite(parsed):
        return None
    return 1.0 if parsed > 0 else 0.0


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


def _format_failure_metric_cell_label(metric_key: str, value: float) -> str:
    """Format heatmap cell text consistently across failure-analysis plots."""
    if metric_key in {"failure_rate_pct", "success_rate_pct", "quality_score_pct"}:
        return f"{value:.1f}%"
    if metric_key == "n":
        return str(int(round(value)))
    return f"{value:.2f}"


def _resolve_failure_heatmap_bounds(
    z: np.ndarray,
    zmin: float | None,
    zmax: float | None,
) -> tuple[float, float]:
    """Resolve stable finite z-bounds for failure heatmaps."""
    finite_values = np.asarray(z, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]

    if zmin is not None and math.isfinite(zmin):
        zmin_eff = float(zmin)
    else:
        zmin_eff = float(finite_values.min()) if finite_values.size else 0.0

    if zmax is not None and math.isfinite(zmax):
        zmax_eff = float(zmax)
    else:
        zmax_eff = float(finite_values.max()) if finite_values.size else zmin_eff + 1.0

    if zmax_eff <= zmin_eff:
        zmax_eff = zmin_eff + 1.0

    return zmin_eff, zmax_eff


def _with_failure_rate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach failure-rate convenience columns used by failure-analysis visualizations."""
    out = frame.copy()
    out["failure_rate"] = 1.0 - out["success_rate"]
    out["success_rate_pct"] = out["success_rate"] * 100.0
    out["failure_rate_pct"] = out["failure_rate"] * 100.0
    return out


def _empty_failure_view_result(message: str) -> tuple:
    """Return the standard empty-state payload for failure-analysis outputs."""
    return (
        _failure_empty_figure("Load spreadsheet data to render aggregate heatmap"),
        _failure_empty_figure("Load spreadsheet data to render stack-condition aggregate heatmap"),
        _failure_empty_figure("Load spreadsheet data to render robot-condition aggregate heatmap"),
        _failure_empty_figure("Select two policies to compare aggregate condition heatmaps"),
        _failure_empty_figure("Select two policies to compare stack-condition heatmaps"),
        _failure_empty_figure("Select two policies to compare robot-condition heatmaps"),
        [],
        [],
        [],
        [],
        message,
    )


def _resolve_failure_axes(
    detail_df: pd.DataFrame,
    condition_columns: dict[str, str],
    default_x: str | None,
    default_y: str | None,
) -> tuple[str | None, str | None]:
    """Resolve valid X/Y condition keys from defaults and available columns."""
    x_key = str(default_x or "")
    y_key = str(default_y or "")
    available_keys = [key for key in condition_columns if key in detail_df.columns]

    if (not x_key or x_key not in detail_df.columns) and available_keys:
        x_key = available_keys[0]
    if (not y_key or y_key not in detail_df.columns or y_key == x_key) and len(available_keys) > 1:
        y_key = next((key for key in available_keys if key != x_key), "")

    if not x_key or not y_key or x_key not in detail_df.columns or y_key not in detail_df.columns:
        return None, None
    return x_key, y_key


def _filter_failure_policy_selection(
    detail_df: pd.DataFrame,
    selected_failure_policies: list[str] | None,
) -> tuple[pd.DataFrame, list[str], str | None]:
    """Filter rollout-detail rows to selected policies and preserve source order."""
    policy_order = list(dict.fromkeys(detail_df["policy_name"].tolist()))
    policy_name_set = set(policy_order)

    if selected_failure_policies is None:
        selected_policy_set = set(policy_order)
    else:
        normalized_selection = [str(name).strip() for name in (selected_failure_policies or []) if str(name).strip()]
        if not normalized_selection:
            return detail_df.iloc[0:0].copy(), [], "No policies selected for failure aggregation."

        selected_policy_set = {name for name in normalized_selection if name in policy_name_set}
        if not selected_policy_set:
            return detail_df.iloc[0:0].copy(), [], "Selected failure policies have no rollout-detail rows."

    filtered_df = detail_df[detail_df["policy_name"].isin(selected_policy_set)].copy()
    if filtered_df.empty:
        return filtered_df, [], "No rollout-detail rows available for selected failure policies."

    filtered_policy_order = [name for name in policy_order if name in selected_policy_set]
    return filtered_df, filtered_policy_order, None


def _aggregate_failure_frames(
    detail_df: pd.DataFrame,
    x_key: str,
    y_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build grouped and aggregate frames used by failure-analysis visualizations."""
    grouped = _with_failure_rate_columns(
        detail_df.groupby(["policy_name", x_key, y_key], as_index=False)
        .agg(
            n=("task_success", "size"),
            success_rate=("task_success", "mean"),
            quality_score_pct=("quality_score_pct", "mean"),
        )
    )

    aggregate = _with_failure_rate_columns(
        detail_df.groupby([x_key, y_key], as_index=False)
        .agg(
            n=("task_success", "size"),
            success_rate=("task_success", "mean"),
            quality_score_pct=("quality_score_pct", "mean"),
            policy_count=("policy_name", "nunique"),
        )
    )

    stack_aggregate = _with_failure_rate_columns(
        detail_df.groupby([y_key], as_index=False)
        .agg(
            n=("task_success", "size"),
            success_rate=("task_success", "mean"),
            quality_score_pct=("quality_score_pct", "mean"),
        )
    )

    robot_aggregate = _with_failure_rate_columns(
        detail_df.groupby([x_key], as_index=False)
        .agg(
            n=("task_success", "size"),
            success_rate=("task_success", "mean"),
            quality_score_pct=("quality_score_pct", "mean"),
        )
    )

    return grouped, aggregate, stack_aggregate, robot_aggregate


def _resolve_failure_metric_settings(
    metric_mode: str | None,
    grouped: pd.DataFrame,
) -> tuple[str, str, str, float, float, str | None]:
    """Resolve metric key/label/colors and z-range for failure heatmaps."""
    mode = metric_mode or "failure_rate"
    if mode == "success_rate":
        return "success_rate_pct", "Success rate (%)", "Greys", 0.0, 100.0, None

    if mode == "quality_score":
        metric_key = "quality_score_pct"
        finite_values = pd.to_numeric(grouped[metric_key], errors="coerce").dropna()
        if finite_values.empty:
            return "", "", "Greys", 0.0, 1.0, "Quality score column is missing in loaded detail sheets"
        zmin = max(0.0, float(math.floor(finite_values.min() / 5.0) * 5.0))
        zmax = min(100.0, float(math.ceil(finite_values.max() / 5.0) * 5.0))
        if zmax <= zmin:
            zmax = zmin + 1.0
        return metric_key, "Quality score mean (%)", "Greys", zmin, zmax, None

    if mode == "sample_count":
        metric_key = "n"
        zmax = float(max(1.0, grouped[metric_key].max()))
        return metric_key, "Sample count (n)", "Greys", 0.0, zmax, None

    return "failure_rate_pct", "Failure rate (%)", "Greys", 0.0, 100.0, None


def _resolve_failure_condition_axis_values(
    detail_df: pd.DataFrame,
    stack_aggregate: pd.DataFrame,
    robot_aggregate: pd.DataFrame,
    x_key: str,
    y_key: str,
    condition_order_mode: str | None,
) -> tuple[str, list[str], list[str], list[str], list[str]]:
    """Resolve axis ordering for aggregate and axis-level failure heatmaps."""
    mode = condition_order_mode or "original"
    if mode == "failure":
        x_ranked = (
            detail_df.groupby([x_key], as_index=False)
            .agg(success_rate=("task_success", "mean"))
        )
        x_ranked[x_key] = x_ranked[x_key].astype(str)
        x_ranked["failure_rate_pct"] = (1.0 - pd.to_numeric(x_ranked["success_rate"], errors="coerce")) * 100.0
        x_ranked["condition_sort_key"] = x_ranked[x_key].map(_condition_sort_key)
        x_ranked = x_ranked.sort_values(
            ["failure_rate_pct", "condition_sort_key"],
            ascending=[False, True],
            kind="stable",
        )
        x_values = x_ranked[x_key].drop_duplicates().tolist()

        y_ranked = (
            detail_df.groupby([y_key], as_index=False)
            .agg(success_rate=("task_success", "mean"))
        )
        y_ranked[y_key] = y_ranked[y_key].astype(str)
        y_ranked["failure_rate_pct"] = (1.0 - pd.to_numeric(y_ranked["success_rate"], errors="coerce")) * 100.0
        y_ranked["condition_sort_key"] = y_ranked[y_key].map(_condition_sort_key)
        y_ranked = y_ranked.sort_values(
            ["failure_rate_pct", "condition_sort_key"],
            ascending=[False, True],
            kind="stable",
        )
        y_values = y_ranked[y_key].drop_duplicates().tolist()

        stack_ranked = stack_aggregate.copy()
        stack_ranked[y_key] = stack_ranked[y_key].astype(str)
        stack_ranked["failure_rate_pct"] = pd.to_numeric(stack_ranked["failure_rate_pct"], errors="coerce")
        stack_ranked["condition_sort_key"] = stack_ranked[y_key].map(_condition_sort_key)
        stack_ranked = stack_ranked.sort_values(
            ["failure_rate_pct", "condition_sort_key"],
            ascending=[False, True],
            kind="stable",
        )
        stack_values = stack_ranked[y_key].drop_duplicates().tolist()

        robot_ranked = robot_aggregate.copy()
        robot_ranked[x_key] = robot_ranked[x_key].astype(str)
        robot_ranked["failure_rate_pct"] = pd.to_numeric(robot_ranked["failure_rate_pct"], errors="coerce")
        robot_ranked["condition_sort_key"] = robot_ranked[x_key].map(_condition_sort_key)
        robot_ranked = robot_ranked.sort_values(
            ["failure_rate_pct", "condition_sort_key"],
            ascending=[False, True],
            kind="stable",
        )
        robot_values = robot_ranked[x_key].drop_duplicates().tolist()
        return mode, x_values, y_values, stack_values, robot_values

    x_values = detail_df[x_key].astype(str).drop_duplicates().tolist()
    y_values = detail_df[y_key].astype(str).drop_duplicates().tolist()
    return mode, x_values, y_values, y_values.copy(), x_values.copy()


def _select_failure_policy_pair(
    policy_order: list[str],
    selected_policy_a: str | None,
    selected_policy_b: str | None,
) -> list[str]:
    """Select up to two distinct policies for side-by-side failure comparisons."""
    selected_pair: list[str] = []
    for candidate in [selected_policy_a, selected_policy_b]:
        text = str(candidate).strip() if candidate is not None else ""
        if text and text in policy_order and text not in selected_pair:
            selected_pair.append(text)
    if len(selected_pair) < 2:
        selected_pair = policy_order[:2]
    return selected_pair[:2]


def _failure_condition_table_columns(x_label: str, y_label: str) -> list[dict[str, str]]:
    """Build column metadata for hardest/easiest failure condition tables."""
    return [
        {"name": x_label, "id": "x_condition"},
        {"name": y_label, "id": "y_condition"},
        {"name": "Failure (%)", "id": "failure_rate_pct"},
        {"name": "Success (%)", "id": "success_rate_pct"},
        {"name": "Quality (%)", "id": "quality_score_pct"},
        {"name": "n", "id": "n"},
        {"name": "Policies", "id": "policy_count"},
    ]


def _build_failure_condition_table(
    aggregate: pd.DataFrame,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    ascending_failure: bool,
) -> tuple[list[dict[str, object]], list[dict[str, str]], pd.DataFrame]:
    """Build one failure-condition table payload and keep a top slice for highlights."""
    top = aggregate.sort_values(["failure_rate_pct", "n"], ascending=[ascending_failure, False]).copy()
    top = top.head(12)
    top["failure_rate_pct"] = top["failure_rate_pct"].round(2)
    top["success_rate_pct"] = top["success_rate_pct"].round(2)
    top["quality_score_pct"] = pd.to_numeric(top["quality_score_pct"], errors="coerce").round(2)

    data = top.rename(columns={x_key: "x_condition", y_key: "y_condition"})[
        ["x_condition", "y_condition", "failure_rate_pct", "success_rate_pct", "quality_score_pct", "n", "policy_count"]
    ].to_dict("records")
    return data, _failure_condition_table_columns(x_label, y_label), top


def _build_failure_highlight_items(rows: pd.DataFrame, x_key: str, y_key: str) -> list[html.Li]:
    """Build bullet items for top hardest/easiest failure conditions."""
    items: list[html.Li] = []
    for _, row in rows.head(3).iterrows():
        items.append(
            html.Li(
                f"{row[y_key]} × {row[x_key]}: {row['failure_rate_pct']:.1f}% failure "
                f"(n={int(row['n'])}, {int(row['policy_count'])} policies)"
            )
        )
    return items


def _build_failure_main_highlights(
    detail_df: pd.DataFrame,
    policy_order: list[str],
    y_values: list[str],
    x_values: list[str],
    y_label: str,
    x_label: str,
    hardest_rows: pd.DataFrame,
    easiest_rows: pd.DataFrame,
    condition_order_mode: str,
    x_key: str,
    y_key: str,
) -> html.Div:
    """Build the summary/highlights panel shown above failure heatmaps."""
    hardest_items = _build_failure_highlight_items(hardest_rows, x_key=x_key, y_key=y_key)
    easiest_items = _build_failure_highlight_items(easiest_rows, x_key=x_key, y_key=y_key)
    return html.Div(
        [
            html.Div(
                f"Detailed rollout data loaded: {len(detail_df)} rows across {len(policy_order)} selected policies. "
                f"Grid size detected: {len(y_values)} × {len(x_values)} ({y_label} × {x_label}).",
                style={"fontWeight": "600"},
            ),
            html.Div("Hardest conditions:", style={"fontWeight": "600", "marginTop": "6px"}),
            html.Ul(hardest_items, style={"marginTop": "4px", "marginBottom": "6px"}),
            html.Div("Easiest conditions:", style={"fontWeight": "600", "marginTop": "6px"}),
            html.Ul(easiest_items, style={"marginTop": "4px", "marginBottom": "6px"}),
            html.Div(
                (
                    "Heatmap ordering: failure severity (highest to lowest)."
                    if condition_order_mode == "failure"
                    else "Heatmap ordering: original spreadsheet condition order."
                ),
                style={"fontSize": "12px", "color": "#666", "marginTop": "4px"},
            ),
        ]
    )


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
    n_cols: int = 4,
    title: str | None = None,
    height: int | None = None,
    share_yaxes: bool = False,
) -> go.Figure:
    if not policy_names:
        return _failure_empty_figure("No policies selected for failure heatmaps")

    display = display_names or {}
    n_cols = max(1, int(n_cols))
    n_rows = int(math.ceil(len(policy_names) / n_cols))
    subplot_titles = [display.get(policy, policy) for policy in policy_names]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.16,
        shared_yaxes=share_yaxes,
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
        if share_yaxes and col_idx > 1:
            fig.update_yaxes(showticklabels=False, row=row_idx, col=col_idx)

    fig.update_layout(
        template="plotly_white",
        title=title or f"Policy failure-mode mini-heatmaps ({metric_label})",
        height=height if height is not None else max(420, n_rows * 300),
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

    zmin_eff, zmax_eff = _resolve_failure_heatmap_bounds(z, zmin=zmin, zmax=zmax)

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
                    "text": _format_failure_metric_cell_label(metric_key, numeric_value),
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


def _build_failure_axis_aggregate_figure(
    aggregate_df: pd.DataFrame,
    axis_key: str,
    axis_values: list[str],
    metric_key: str,
    metric_label: str,
    colorscale: str,
    zmin: float | None,
    zmax: float | None,
    axis_label: str,
    row_label: str,
    title: str,
) -> go.Figure:
    if aggregate_df.empty or not axis_values:
        return _failure_empty_figure("No aggregate condition data available")

    z_row: list[float] = []
    n_row: list[float] = []
    key_series = aggregate_df[axis_key].astype(str)
    for axis_value in axis_values:
        matches = aggregate_df[key_series == str(axis_value)]
        if matches.empty:
            z_row.append(math.nan)
            n_row.append(0.0)
            continue

        metric_value = pd.to_numeric(matches.iloc[0][metric_key], errors="coerce")
        sample_count = pd.to_numeric(matches.iloc[0]["n"], errors="coerce")
        z_row.append(float(metric_value) if pd.notna(metric_value) else math.nan)
        n_row.append(float(sample_count) if pd.notna(sample_count) else 0.0)

    z = np.array([z_row], dtype=float)
    n = np.array([n_row], dtype=float)

    fig = go.Figure(
        data=[
            go.Heatmap(
                x=axis_values,
                y=[row_label],
                z=z,
                customdata=n,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                colorbar={"title": metric_label},
                hovertemplate=(
                    "%{x}<br>"
                    + metric_label
                    + ": %{z:.2f}<br>n: %{customdata:.0f}<extra>"
                    + row_label
                    + "</extra>"
                ),
            )
        ]
    )

    zmin_eff, zmax_eff = _resolve_failure_heatmap_bounds(z, zmin=zmin, zmax=zmax)

    annotations: list[dict[str, object]] = []
    for x_idx, x_value in enumerate(axis_values):
        cell_value = z[0][x_idx]
        if pd.isna(cell_value):
            continue

        numeric_value = float(cell_value)
        normalized = (numeric_value - zmin_eff) / (zmax_eff - zmin_eff)
        normalized = max(0.0, min(1.0, normalized))
        text_color = "#FFFFFF" if normalized >= 0.58 else "#111111"

        annotations.append(
            {
                "x": x_value,
                "y": row_label,
                "xref": "x",
                "yref": "y",
                "text": _format_failure_metric_cell_label(metric_key, numeric_value),
                "showarrow": False,
                "font": {"size": 11, "color": text_color},
            }
        )

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title=axis_label,
        yaxis_title="",
        margin={"l": 30, "r": 30, "t": 70, "b": 50},
        height=250,
        annotations=annotations,
    )
    fig.update_xaxes(tickangle=35)
    return fig


def _build_failure_policy_axis_pair_figure(
    axis_df: pd.DataFrame,
    policy_names: list[str],
    axis_key: str,
    axis_values: list[str],
    metric_key: str,
    metric_label: str,
    colorscale: str,
    zmin: float | None,
    zmax: float | None,
    axis_label: str,
    title: str,
    row_label: str | None = None,
    display_names: dict[str, str] | None = None,
    height: int = 240,
) -> go.Figure:
    if axis_df.empty or not policy_names or not axis_values:
        return _failure_empty_figure("No selected policy condition data available")

    display = display_names or {}
    _row_label_text = str(row_label).strip() if row_label is not None and str(row_label).strip() else None
    subplot_titles = [display.get(policy, policy) for policy in policy_names]
    fig = make_subplots(rows=1, cols=len(policy_names), subplot_titles=subplot_titles, horizontal_spacing=0.09)

    for index, policy_name in enumerate(policy_names):
        policy_slice = axis_df[axis_df["policy_name"].astype(str) == str(policy_name)].copy()
        z_row: list[float] = []
        n_row: list[float] = []
        key_series = policy_slice[axis_key].astype(str) if axis_key in policy_slice.columns else pd.Series(dtype=str)
        for axis_value in axis_values:
            matches = policy_slice[key_series == str(axis_value)]
            if matches.empty:
                z_row.append(math.nan)
                n_row.append(0.0)
                continue
            metric_value = pd.to_numeric(matches.iloc[0][metric_key], errors="coerce")
            sample_count = pd.to_numeric(matches.iloc[0]["n"], errors="coerce")
            z_row.append(float(metric_value) if pd.notna(metric_value) else math.nan)
            n_row.append(float(sample_count) if pd.notna(sample_count) else 0.0)

        z = np.array([z_row], dtype=float)
        n = np.array([n_row], dtype=float)
        short_name = display.get(policy_name, policy_name)
        y_row = _row_label_text or short_name

        fig.add_trace(
            go.Heatmap(
                x=axis_values,
                y=[y_row],
                z=z,
                customdata=n,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                showscale=index == 0,
                colorbar={"title": metric_label} if index == 0 else None,
                hovertemplate=(
                    "%{x}<br>"
                    + metric_label
                    + ": %{z:.2f}<br>n: %{customdata:.0f}<extra>"
                    + short_name
                    + "</extra>"
                ),
            ),
            row=1,
            col=index + 1,
        )
        fig.update_xaxes(tickangle=35, title_text=axis_label, row=1, col=index + 1)
        if index > 0:
            fig.update_yaxes(showticklabels=False, row=1, col=index + 1)

    fig.update_layout(
        template="plotly_white",
        title=title,
        margin={"l": 30, "r": 30, "t": 70, "b": 45},
        height=height,
    )
    return fig


app = Dash(__name__)
app.title = "Robot Policy Evaluation Dashboard"

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px"},
    children=[
        dcc.Store(id="uploaded-file-store", data=None),
        dcc.Store(id="sort-mode-store", data="original"),
        dcc.Store(id="active-testing-group-store", data=None),
        dcc.Store(id="leaderboard-active-testing-group-store", data=None),
        dcc.Store(id="failure-active-testing-group-store", data=None),
        dcc.Store(
            id="failure-detail-store",
            data={"records": [], "condition_columns": [], "default_x": None, "default_y": None, "policy_meta": []},
        ),
        dcc.Tabs(
            id="page-tabs",
            value="ab",
            children=[
                dcc.Tab(label="A/B Testing", value="ab"),
                dcc.Tab(label="Leaderboard", value="leaderboard"),
                dcc.Tab(label="Failure Mode Analysis", value="failure"),
            ],
        ),
        html.Div(
            id="main-page",
            children=[
                html.Div(
                    id="ab-page",
                    children=[
                        html.H2("A/B Testing"),
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
                        html.Div(id="ab-common-legend", style={"marginTop": "8px"}),
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(3, minmax(0, 1fr))",
                                "gap": "10px",
                                "alignItems": "start",
                            },
                            children=[
                                dcc.Graph(id="ab-comparison-graph", style={"height": "360px"}),
                                dcc.Graph(id="ab-quality-graph", style={"height": "360px"}),
                                dcc.Graph(id="ab-dropin-graph", style={"height": "360px"}),
                                dcc.Graph(id="ab-violin-graph", style={"height": "360px"}),
                                dcc.Graph(id="ab-quality-violin-graph", style={"height": "360px"}),
                                dcc.Graph(id="ab-dropin-violin-graph", style={"height": "360px"}),
                            ],
                        ),
                        dcc.Graph(id="ab-failure-heatmap-graph"),
                        html.Hr(),
                        dcc.Checklist(
                            id="show-raw-table-toggle",
                            options=[{"label": "Show loaded source table (optional sanity check)", "value": "show"}],
                            value=[],
                            inline=True,
                            style={"marginTop": "4px"},
                        ),
                        html.Div(
                            id="raw-table-wrapper",
                            style={"display": "none", "marginTop": "8px"},
                            children=[
                                html.Div(
                                    style={"display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap"},
                                    children=[
                                        html.Button("Download CSV", id="download-btn"),
                                        dcc.Download(id="download-csv"),
                                    ],
                                ),
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
                            ],
                        ),
                    ],
                ),
                html.Div(
                    id="leaderboard-page",
                    style={"display": "none"},
                    children=[
                        html.H2("Leaderboard"),
                        html.P("Rank all policies and compare selected policies against the chosen base policy."),
                        html.H4("Policy leaderboard"),
                        html.Div(
                            style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap", "marginBottom": "8px"},
                            children=[
                                html.Div(
                                    style={"minWidth": "260px", "maxWidth": "360px"},
                                    children=[
                                        html.Label("Testing Group", style={"fontSize": "12px", "marginBottom": "2px"}),
                                        dcc.Dropdown(
                                            id="leaderboard-testing-group-dropdown",
                                            options=[],
                                            value=[],
                                            multi=True,
                                            clearable=True,
                                            placeholder="Select tag(s)",
                                            disabled=True,
                                        ),
                                    ],
                                ),
                                html.Button("Leaderboard Tag + Base", id="leaderboard-apply-testing-group-btn"),
                                html.Button("Clear Leaderboard Tag Filter", id="leaderboard-clear-testing-group-btn"),
                                html.Div(
                                    style={"minWidth": "160px", "maxWidth": "180px"},
                                    children=[
                                        html.Label("Policies per page", style={"fontSize": "12px", "marginBottom": "2px"}),
                                        dcc.Dropdown(
                                            id="leaderboard-page-size-dropdown",
                                            options=[
                                                {"label": "15", "value": 15},
                                                {"label": "50", "value": 50},
                                            ],
                                            value=15,
                                            clearable=False,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="leaderboard-tag-filter-status",
                            children="Leaderboard tag filter: all",
                            style={"fontSize": "12px", "color": "#666", "marginBottom": "6px"},
                        ),
                        dash_table.DataTable(
                            id="leaderboard-table",
                            columns=[
                                {"name": "Policy", "id": "policy"},
                                {"name": "Quality Score [%] ↑", "id": "quality_score_pct"},
                                {"name": "Success Rate [%] ↑", "id": "success_rate_pct"},
                                {"name": "Drop in attempt [%] ↓", "id": "dropin_ratio_pct"},
                            ],
                            data=[],
                            page_size=15,
                            sort_action="native",
                            sort_mode="single",
                            sort_by=[{"column_id": "quality_score_pct", "direction": "desc"}],
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "fontFamily": "sans-serif", "fontSize": 13},
                            style_data_conditional=[],
                        ),
                        html.Div(
                            "↑ higher is better, ↓ lower is better.",
                            style={"fontSize": "12px", "color": "#666", "marginTop": "6px"},
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
                                                    value=[],
                                                    multi=True,
                                                    clearable=True,
                                                    placeholder="Select tag(s)",
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
                        html.Div(
                            id="final-violin-wrapper",
                            children=[
                                dcc.Graph(id="cld-violin-graph"),
                                dcc.Graph(id="base-vs-quality-violin-graph"),
                                dcc.Graph(id="base-vs-dropin-violin-graph"),
                            ],
                        ),
                        dash_table.DataTable(
                            id="cld-table",
                            columns=[],
                            data=[],
                            page_size=12,
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "fontFamily": "sans-serif", "fontSize": 13},
                            style_data_conditional=[],
                        ),
                        html.H4("Base-vs-policy significant metric summary"),
                        dash_table.DataTable(
                            id="leaderboard-significant-pairs-table",
                            columns=[],
                            data=[],
                            page_size=12,
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "fontFamily": "sans-serif", "fontSize": 13},
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
                        html.Hr(),
                        dcc.Checklist(
                            id="show-summary-table-toggle",
                            options=[{"label": "Show per-policy interval details", "value": "show"}],
                            value=[],
                            inline=True,
                            style={"marginTop": "8px"},
                        ),
                        html.Div(
                            id="summary-table-wrapper",
                            style={"display": "none", "marginTop": "8px"},
                            children=[
                                html.H4("Per-policy intervals (detailed)"),
                                dash_table.DataTable(
                                    id="summary-table",
                                    columns=[],
                                    data=[],
                                    page_size=12,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"textAlign": "left", "fontFamily": "sans-serif", "fontSize": 13},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            id="failure-page",
            style={"display": "none"},
            children=[
                html.H2("Failure Mode Analysis"),
                html.P(
                    "Dedicated page for detailed rollout-level diagnostics. "
                    "Load per-policy detail links, then inspect the aggregate condition grid across selected completed policies."
                ),
                html.H4("Failure mode highlights"),
                html.Div(
                    id="failure-main-highlights",
                    style={
                        "background": "#fafafa",
                        "border": "1px solid #e0e0e0",
                        "borderRadius": "6px",
                        "padding": "10px 12px",
                    },
                ),
                html.Div(
                    style={"display": "grid", "gap": "8px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "gap": "8px", "alignItems": "flex-end", "flexWrap": "wrap"},
                            children=[
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
                                html.Div(
                                    style={"minWidth": "250px"},
                                    children=[
                                        html.Label("Condition order", style={"fontSize": "12px", "marginBottom": "2px"}),
                                        dcc.Dropdown(
                                            id="failure-condition-order-dropdown",
                                            options=FAILURE_CONDITION_ORDER_OPTIONS,
                                            value="original",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={"minWidth": "240px"},
                                    children=[
                                        html.Label("Policy A", style={"fontSize": "12px", "marginBottom": "2px"}),
                                        dcc.Dropdown(
                                            id="failure-policy-a-dropdown",
                                            options=[],
                                            value=None,
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={"minWidth": "240px"},
                                    children=[
                                        html.Label("Policy B", style={"fontSize": "12px", "marginBottom": "2px"}),
                                        dcc.Dropdown(
                                            id="failure-policy-b-dropdown",
                                            options=[],
                                            value=None,
                                            clearable=False,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
                            children=[
                                html.Button("Select All", id="failure-select-all-btn"),
                                html.Button("Deselect All", id="failure-deselect-all-btn"),
                                html.Div(
                                    style={"minWidth": "260px", "maxWidth": "360px"},
                                    children=[
                                        html.Label("Testing Group", style={"fontSize": "12px", "marginBottom": "2px"}),
                                        dcc.Dropdown(
                                            id="failure-testing-group-dropdown",
                                            options=[],
                                            value=[],
                                            multi=True,
                                            clearable=True,
                                            placeholder="Select tag(s)",
                                            disabled=True,
                                        ),
                                    ],
                                ),
                                html.Button("Plot Tag + Base", id="failure-apply-testing-group-btn"),
                                html.Button("Clear Tag Filter", id="failure-clear-testing-group-btn"),
                            ],
                        ),
                        html.Div(
                            children=[
                                html.Label("Policies to include in aggregation", style={"fontSize": "13px", "fontWeight": "600"}),
                                dcc.Checklist(
                                    id="failure-policy-checklist",
                                    options=[],
                                    value=None,
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
                    ],
                ),
                html.Div(id="failure-load-status", style={"marginTop": "8px", "fontSize": "13px"}),
                dcc.Graph(id="failure-aggregate-graph"),
                dcc.Graph(id="failure-stack-aggregate-graph"),
                dcc.Graph(id="failure-robot-aggregate-graph"),
                html.H4("Selected policy comparison heatmaps"),
                dcc.Graph(id="failure-policy-aggregate-graph"),
                dcc.Graph(id="failure-policy-stack-graph"),
                dcc.Graph(id="failure-policy-robot-graph"),
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
    Output("ab-page", "style"),
    Output("leaderboard-page", "style"),
    Output("failure-page", "style"),
    Input("page-tabs", "value"),
)
def toggle_pages(active_tab: str | None):
    if active_tab == "failure":
        return {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "block"}
    if active_tab == "leaderboard":
        return {"display": "block"}, {"display": "none"}, {"display": "block"}, {"display": "none"}
    return {"display": "block"}, {"display": "block"}, {"display": "none"}, {"display": "none"}


@app.callback(
    Output("raw-table-wrapper", "style"),
    Input("show-raw-table-toggle", "value"),
)
def toggle_raw_table_wrapper(show_toggle: list[str] | None):
    if "show" in (show_toggle or []):
        return {"display": "block", "marginTop": "8px"}
    return {"display": "none", "marginTop": "8px"}


@app.callback(
    Output("summary-table-wrapper", "style"),
    Input("show-summary-table-toggle", "value"),
)
def toggle_summary_table_wrapper(show_toggle: list[str] | None):
    if "show" in (show_toggle or []):
        return {"display": "block", "marginTop": "8px"}
    return {"display": "none", "marginTop": "8px"}


@app.callback(
    Output("failure-detail-store", "data"),
    Output("failure-load-status", "children"),
    Input("uploaded-file-store", "data"),
    State("raw-table", "data"),
    prevent_initial_call=True,
)
def load_failure_detail_data(
    _uploaded_state: dict | None,
    rows: list[dict] | None,
):
    empty_store = {
        "records": [],
        "condition_columns": [],
        "default_x": None,
        "default_y": None,
        "policy_meta": [],
    }

    if not rows:
        return empty_store, "Load the main policy table first."

    all_policy_links = _collect_policy_detail_links(rows)
    clean_df = _raw_to_clean_df(rows)
    policy_meta_map: dict[str, dict[str, object]] = {}
    if not clean_df.empty:
        meta_df = clean_df.copy()
        if "source_order" in meta_df.columns:
            meta_df = meta_df.sort_values("source_order", kind="stable")
        for _, meta_row in meta_df.iterrows():
            policy_name = str(meta_row.get("model_name", "")).strip()
            if not policy_name:
                continue
            raw_tags = meta_row.get("testing_group_tags", [])
            tags = raw_tags if isinstance(raw_tags, list) else _split_testing_group_tags(raw_tags)
            is_base = bool(meta_row.get("is_base_group", False)) or any(_is_base_group_tag(tag) for tag in tags)
            policy_meta_map[policy_name] = {
                "policy_name": policy_name,
                "testing_group_tags": tags,
                "is_base_group": is_base,
            }

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
    policy_meta: list[dict[str, object]] = []
    for index, policy_name in enumerate(unique_policies):
        meta_entry = policy_meta_map.get(policy_name, {})
        raw_tags = meta_entry.get("testing_group_tags", [])
        tags = raw_tags if isinstance(raw_tags, list) else _split_testing_group_tags(raw_tags)
        is_base = bool(meta_entry.get("is_base_group", False)) or any(_is_base_group_tag(tag) for tag in tags)
        policy_meta.append(
            {
                "policy_name": policy_name,
                "testing_group_tags": tags,
                "is_base_group": is_base,
                "source_order": index,
            }
        )

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
        "policy_meta": policy_meta,
    }

    return (
        store_data,
        status,
    )


@app.callback(
    Output("failure-policy-a-dropdown", "options"),
    Output("failure-policy-a-dropdown", "value"),
    Output("failure-policy-b-dropdown", "options"),
    Output("failure-policy-b-dropdown", "value"),
    Output("failure-policy-checklist", "options"),
    Output("failure-policy-checklist", "value"),
    Output("failure-testing-group-dropdown", "options"),
    Output("failure-testing-group-dropdown", "value"),
    Output("failure-testing-group-dropdown", "disabled"),
    Output("failure-active-testing-group-store", "data"),
    Input("failure-detail-store", "data"),
    Input("failure-select-all-btn", "n_clicks"),
    Input("failure-deselect-all-btn", "n_clicks"),
    Input("failure-apply-testing-group-btn", "n_clicks"),
    Input("failure-clear-testing-group-btn", "n_clicks"),
    State("failure-policy-a-dropdown", "value"),
    State("failure-policy-b-dropdown", "value"),
    State("failure-policy-checklist", "value"),
    State("failure-testing-group-dropdown", "value"),
    State("failure-active-testing-group-store", "data"),
)
def sync_failure_policy_selectors(
    failure_store: dict | None,
    _select_all_clicks: int,
    _deselect_all_clicks: int,
    _apply_testing_group_clicks: int,
    _clear_testing_group_clicks: int,
    current_policy_a: str | None,
    current_policy_b: str | None,
    current_checked: list[str] | None,
    selected_testing_group: list[str] | str | None,
    current_active_group: list[str] | str | None,
):
    """Keep failure-analysis selector widgets in sync with loaded rollout-detail data."""
    empty_result = ([], None, [], None, [], [], [], [], True, [])
    if not failure_store or not failure_store.get("records"):
        return empty_result

    detail_df = pd.DataFrame(failure_store.get("records") or [])
    if detail_df.empty or "policy_name" not in detail_df.columns:
        return empty_result

    policies = list(dict.fromkeys(detail_df["policy_name"].astype(str).tolist()))
    if not policies:
        return empty_result

    display_map, _prefix = _make_display_names(policies)
    options = [{"label": display_map.get(policy, policy), "value": policy} for policy in policies]
    policy_meta_entries = failure_store.get("policy_meta") or []
    policy_meta_map: dict[str, dict[str, object]] = {}
    for entry in policy_meta_entries:
        name = str(entry.get("policy_name", "")).strip()
        if not name:
            continue
        policy_meta_map[name] = entry

    group_entries: list[tuple[str, list[str], bool]] = []
    for policy_name in policies:
        entry = policy_meta_map.get(policy_name, {})
        raw_tags = entry.get("testing_group_tags", [])
        tags = raw_tags if isinstance(raw_tags, list) else _split_testing_group_tags(raw_tags)
        row_is_base = bool(entry.get("is_base_group", False)) or any(_is_base_group_tag(tag) for tag in tags)
        group_entries.append((policy_name, tags, row_is_base))

    tag_order, group_to_models, base_models = _build_testing_group_index(group_entries)

    tag_options = [{"label": tag, "value": tag} for tag in tag_order]
    tag_set = set(tag_order)
    if not base_models and current_policy_a in policies:
        base_models = [str(current_policy_a)]

    trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else "failure-detail-store"
    selected_groups = [group for group in _normalize_group_selection(selected_testing_group) if group in tag_set]
    active_groups = [group for group in _normalize_group_selection(current_active_group) if group in tag_set]
    current_checked = current_checked or []

    if trigger == "failure-select-all-btn":
        checked = policies
        active_groups = []
    elif trigger == "failure-deselect-all-btn":
        checked = []
        active_groups = []
    elif trigger == "failure-apply-testing-group-btn":
        if selected_groups:
            checked = _select_models_for_group_tags(selected_groups, base_models, group_to_models, policies)
            active_groups = selected_groups
        else:
            checked = [model for model in current_checked if model in policies]
            if not checked:
                checked = policies
            active_groups = []
    elif trigger == "failure-clear-testing-group-btn":
        checked = policies
        active_groups = []
    elif trigger == "failure-detail-store" and active_groups:
        checked = _select_models_for_group_tags(active_groups, base_models, group_to_models, policies)
    else:
        checked = [model for model in current_checked if model in policies]
        if not checked:
            checked = policies

    dropdown_value, dropdown_disabled, active_groups = _resolve_group_dropdown_state(
        trigger,
        selected_groups,
        active_groups,
        tag_options,
        clear_trigger_id="failure-clear-testing-group-btn",
    )

    policy_pool = checked if checked else policies
    if current_policy_a in policy_pool:
        policy_a = current_policy_a
    else:
        policy_a = policy_pool[0] if policy_pool else None

    remaining = [policy for policy in policy_pool if policy != policy_a]
    if current_policy_b in policy_pool and current_policy_b != policy_a:
        policy_b = current_policy_b
    elif remaining:
        policy_b = remaining[0]
    else:
        policy_b = policy_a

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
        active_groups,
    )


@app.callback(
    Output("failure-aggregate-graph", "figure"),
    Output("failure-stack-aggregate-graph", "figure"),
    Output("failure-robot-aggregate-graph", "figure"),
    Output("failure-policy-aggregate-graph", "figure"),
    Output("failure-policy-stack-graph", "figure"),
    Output("failure-policy-robot-graph", "figure"),
    Output("failure-top-conditions-table", "data"),
    Output("failure-top-conditions-table", "columns"),
    Output("failure-easiest-conditions-table", "data"),
    Output("failure-easiest-conditions-table", "columns"),
    Output("failure-main-highlights", "children"),
    Input("failure-detail-store", "data"),
    Input("failure-metric-dropdown", "value"),
    Input("failure-condition-order-dropdown", "value"),
    Input("failure-policy-checklist", "value"),
    Input("failure-policy-a-dropdown", "value"),
    Input("failure-policy-b-dropdown", "value"),
)
def update_failure_views(
    failure_store: dict | None,
    metric_mode: str | None,
    condition_order_mode: str | None,
    selected_failure_policies: list[str] | None,
    selected_policy_a: str | None,
    selected_policy_b: str | None,
):
    """Build aggregate and pairwise failure-analysis views from rollout-level detail rows."""
    empty_message = "Load or refresh a spreadsheet from A/B Testing page to see failure-analysis highlights."
    if not failure_store or not failure_store.get("records"):
        return _empty_failure_view_result(empty_message)

    detail_df = pd.DataFrame(failure_store.get("records") or [])
    if detail_df.empty:
        return _empty_failure_view_result("No detail rollout rows available")

    condition_columns = {
        str(entry.get("key", "")): str(entry.get("label", ""))
        for entry in (failure_store.get("condition_columns") or [])
        if str(entry.get("key", "")).strip()
    }
    x_key, y_key = _resolve_failure_axes(
        detail_df,
        condition_columns,
        default_x=str(failure_store.get("default_x") or ""),
        default_y=str(failure_store.get("default_y") or ""),
    )
    if x_key is None or y_key is None:
        return _empty_failure_view_result("Condition axes are not available yet. Reload detail sheets and try again.")

    detail_df = detail_df.copy()
    detail_df["policy_name"] = detail_df["policy_name"].astype(str)
    detail_df["task_success"] = pd.to_numeric(detail_df["task_success"], errors="coerce")
    detail_df = detail_df[detail_df["task_success"].notna()].copy()
    if detail_df.empty:
        return _empty_failure_view_result("No valid Task Success values detected in loaded detail sheets.")

    detail_df[x_key] = detail_df[x_key].fillna("NA").astype(str)
    detail_df[y_key] = detail_df[y_key].fillna("NA").astype(str)
    detail_df["quality_score_pct"] = pd.to_numeric(detail_df.get("quality_score_pct"), errors="coerce")

    detail_df, policy_order, selection_error = _filter_failure_policy_selection(
        detail_df,
        selected_failure_policies,
    )
    if selection_error is not None:
        return _empty_failure_view_result(selection_error)

    grouped, aggregate, stack_aggregate, robot_aggregate = _aggregate_failure_frames(detail_df, x_key=x_key, y_key=y_key)

    metric_key, metric_label, colorscale, zmin, zmax, metric_error = _resolve_failure_metric_settings(
        metric_mode,
        grouped,
    )
    if metric_error is not None:
        return _empty_failure_view_result(metric_error)

    condition_order_mode, x_values, y_values, stack_values, robot_values = _resolve_failure_condition_axis_values(
        detail_df,
        stack_aggregate,
        robot_aggregate,
        x_key=x_key,
        y_key=y_key,
        condition_order_mode=condition_order_mode,
    )

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

    stack_aggregate_fig = _build_failure_axis_aggregate_figure(
        stack_aggregate,
        axis_key=y_key,
        axis_values=stack_values,
        metric_key=metric_key,
        metric_label=metric_label,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        axis_label=y_label,
        row_label="All robot conditions",
        title=f"Aggregated by {y_label} ({metric_label})",
    )
    robot_aggregate_fig = _build_failure_axis_aggregate_figure(
        robot_aggregate,
        axis_key=x_key,
        axis_values=robot_values,
        metric_key=metric_key,
        metric_label=metric_label,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        axis_label=x_label,
        row_label="All stack conditions",
        title=f"Aggregated by {x_label} ({metric_label})",
    )

    selected_pair = _select_failure_policy_pair(policy_order, selected_policy_a, selected_policy_b)

    policy_display_map, _policy_prefix = _make_display_names(policy_order)

    if len(selected_pair) >= 2:
        policy_aggregate_fig = _build_failure_policy_grid_figure(
            grouped,
            policy_names=selected_pair,
            x_values=x_values,
            y_values=y_values,
            x_key=x_key,
            y_key=y_key,
            metric_key=metric_key,
            metric_label=metric_label,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            display_names=policy_display_map,
            n_cols=2,
            title=f"Selected policy comparison — aggregate condition heatmap ({metric_label})",
            height=340,
            share_yaxes=True,
        )

        policy_stack_aggregate = _with_failure_rate_columns(
            grouped.groupby(["policy_name", y_key], as_index=False)
            .agg(
                n=("n", "sum"),
                success_rate=("success_rate", "mean"),
                quality_score_pct=("quality_score_pct", "mean"),
            )
        )

        policy_robot_aggregate = _with_failure_rate_columns(
            grouped.groupby(["policy_name", x_key], as_index=False)
            .agg(
                n=("n", "sum"),
                success_rate=("success_rate", "mean"),
                quality_score_pct=("quality_score_pct", "mean"),
            )
        )

        policy_stack_fig = _build_failure_policy_axis_pair_figure(
            policy_stack_aggregate,
            policy_names=selected_pair,
            axis_key=y_key,
            axis_values=stack_values,
            metric_key=metric_key,
            metric_label=metric_label,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            axis_label=y_label,
            title=f"Selected policy comparison — aggregated by {y_label} ({metric_label})",
            row_label="All robot conditions",
            display_names=policy_display_map,
            height=240,
        )

        policy_robot_fig = _build_failure_policy_axis_pair_figure(
            policy_robot_aggregate,
            policy_names=selected_pair,
            axis_key=x_key,
            axis_values=robot_values,
            metric_key=metric_key,
            metric_label=metric_label,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            axis_label=x_label,
            title=f"Selected policy comparison — aggregated by {x_label} ({metric_label})",
            row_label="All stack conditions",
            display_names=policy_display_map,
            height=240,
        )
    else:
        policy_aggregate_fig = _failure_empty_figure("Select two policies to compare aggregate condition heatmaps")
        policy_stack_fig = _failure_empty_figure("Select two policies to compare stack-condition heatmaps")
        policy_robot_fig = _failure_empty_figure("Select two policies to compare robot-condition heatmaps")

    hardest_data, hardest_columns, hardest_rows = _build_failure_condition_table(
        aggregate,
        x_key=x_key,
        y_key=y_key,
        x_label=x_label,
        y_label=y_label,
        ascending_failure=False,
    )
    easiest_data, easiest_columns, easiest_rows = _build_failure_condition_table(
        aggregate,
        x_key=x_key,
        y_key=y_key,
        x_label=x_label,
        y_label=y_label,
        ascending_failure=True,
    )

    main_highlights = _build_failure_main_highlights(
        detail_df,
        policy_order,
        y_values,
        x_values,
        y_label,
        x_label,
        hardest_rows,
        easiest_rows,
        condition_order_mode,
        x_key=x_key,
        y_key=y_key,
    )

    return (
        aggregate_fig,
        stack_aggregate_fig,
        robot_aggregate_fig,
        policy_aggregate_fig,
        policy_stack_fig,
        policy_robot_fig,
        hardest_data,
        hardest_columns,
        easiest_data,
        easiest_columns,
        main_highlights,
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

    upload_state_with_refresh = dict(upload_state)
    upload_state_with_refresh["last_loaded_sheet"] = sheet_name
    upload_state_with_refresh["failure_refresh_token"] = time.time_ns()

    return normalized.to_dict("records"), columns, status, upload_state_with_refresh, options, sheet_name, disabled


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
    selected_testing_group: list[str] | str | None,
    current_active_group: list[str] | str | None,
):
    """Synchronize A/B and multi-policy selectors after table edits or group actions."""
    clean_df = _raw_to_clean_df(rows)
    if "has_success_rate_input" in clean_df.columns and not clean_df.empty:
        eligible_models = clean_df.loc[clean_df["has_success_rate_input"].fillna(True), "model_name"].astype(str).tolist()
    else:
        eligible_models = clean_df["model_name"].astype(str).tolist() if not clean_df.empty else []

    models = eligible_models
    display_map, _pfx = _make_display_names(models)
    options = [{"label": display_map.get(model, model), "value": model} for model in models]

    tag_order, group_to_models, base_models = _build_testing_group_index_for_models(clean_df, models)

    tag_options = [{"label": tag, "value": tag} for tag in tag_order]
    tag_set = set(tag_order)
    if not base_models and current_a in models:
        base_models = [str(current_a)]

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
    active_groups = [group for group in _normalize_group_selection(current_active_group) if group in tag_set]

    selected_groups = [group for group in _normalize_group_selection(selected_testing_group) if group in tag_set]

    if trigger == "select-all-btn":
        checked = eligible_models
        active_groups = []
    elif trigger == "deselect-all-btn":
        checked = []
        active_groups = []
    elif trigger == "apply-testing-group-btn":
        if selected_groups:
            checked = _select_models_for_group_tags(selected_groups, base_models, group_to_models, models)
            active_groups = selected_groups
        else:
            checked = [m for m in current_checked if m in models]
            if not checked:
                checked = eligible_models
            active_groups = []
    elif trigger == "clear-testing-group-btn":
        checked = eligible_models
        active_groups = []
    elif trigger == "raw-table" and active_groups:
        checked = _select_models_for_group_tags(active_groups, base_models, group_to_models, models)
    else:
        checked = [m for m in current_checked if m in models]
        if not checked:
            checked = eligible_models

    dropdown_value, dropdown_disabled, active_groups = _resolve_group_dropdown_state(
        trigger,
        selected_groups,
        active_groups,
        tag_options,
        clear_trigger_id="clear-testing-group-btn",
    )

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
        active_groups,
    )


@app.callback(
    Output("leaderboard-testing-group-dropdown", "options"),
    Output("leaderboard-testing-group-dropdown", "value"),
    Output("leaderboard-testing-group-dropdown", "disabled"),
    Output("leaderboard-active-testing-group-store", "data"),
    Input("raw-table", "data"),
    Input("leaderboard-apply-testing-group-btn", "n_clicks"),
    Input("leaderboard-clear-testing-group-btn", "n_clicks"),
    State("leaderboard-testing-group-dropdown", "value"),
    State("leaderboard-active-testing-group-store", "data"),
)
def sync_leaderboard_testing_group_selector(
    rows: list[dict] | None,
    _apply_clicks: int,
    _clear_clicks: int,
    selected_testing_group: list[str] | str | None,
    current_active_group: list[str] | str | None,
):
    clean_df = _raw_to_clean_df(rows)
    if "has_success_rate_input" in clean_df.columns and not clean_df.empty:
        leaderboard_models = clean_df.loc[
            clean_df["has_success_rate_input"].fillna(True), "model_name"
        ].astype(str).tolist()
    else:
        leaderboard_models = clean_df["model_name"].astype(str).tolist() if not clean_df.empty else []

    tag_order, _group_to_models, _base_models = _build_testing_group_index_for_models(clean_df, leaderboard_models)
    tag_options = [{"label": tag, "value": tag} for tag in tag_order]
    tag_set = set(tag_order)

    trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else "raw-table"
    selected_groups = [group for group in _normalize_group_selection(selected_testing_group) if group in tag_set]
    active_groups = [group for group in _normalize_group_selection(current_active_group) if group in tag_set]

    if trigger == "leaderboard-apply-testing-group-btn":
        active_groups = selected_groups
    elif trigger == "leaderboard-clear-testing-group-btn":
        active_groups = []

    dropdown_value, dropdown_disabled, active_groups = _resolve_group_dropdown_state(
        trigger,
        selected_groups,
        active_groups,
        tag_options,
        clear_trigger_id="leaderboard-clear-testing-group-btn",
    )

    return tag_options, dropdown_value, dropdown_disabled, active_groups


@app.callback(
    Output("leaderboard-table", "page_size"),
    Input("leaderboard-page-size-dropdown", "value"),
)
def update_leaderboard_page_size(page_size_value: int | str | None):
    return 50 if str(page_size_value) == "50" else 15


@app.callback(
    Output("leaderboard-table", "data"),
    Output("leaderboard-table", "style_data_conditional"),
    Output("leaderboard-tag-filter-status", "children"),
    Output("base-vs-quality-violin-graph", "figure"),
    Output("base-vs-dropin-violin-graph", "figure"),
    Output("leaderboard-significant-pairs-table", "data"),
    Output("leaderboard-significant-pairs-table", "columns"),
    Input("raw-table", "data"),
    Input("confidence-level", "value"),
    Input("policy-a-dropdown", "value"),
    Input("policy-checklist", "value"),
    Input("active-testing-group-store", "data"),
    Input("leaderboard-active-testing-group-store", "data"),
    Input("sort-mode-store", "data"),
)
def update_leaderboard_content(
    rows: list[dict] | None,
    confidence_level: float,
    base_policy_selected: str | None,
    selected_policies: list[str] | None,
    active_testing_group: list[str] | str | None,
    active_leaderboard_testing_group: list[str] | str | None,
    sort_mode: str | None,
):
    summary_columns = [
        {"name": "Base", "id": "base_policy"},
        {"name": "Policy", "id": "policy"},
        {"name": "Significant metrics", "id": "significant_metrics"},
        {"name": "Δ SR (pp)", "id": "sr_delta_pp"},
        {"name": "Δ Quality (pp)", "id": "quality_delta_pp"},
        {"name": "Δ Drop-in (pp, lower is better)", "id": "dropin_delta_pp"},
    ]

    clean_df = _raw_to_clean_df(rows)
    if "has_success_rate_input" in clean_df.columns and not clean_df.empty:
        clean_df = clean_df[clean_df["has_success_rate_input"].fillna(True)].copy()

    if clean_df.empty:
        return (
            [],
            [],
            "Leaderboard tag filter: all",
            _empty_figure("No quality-score data for selected base-vs-policy pairs"),
            _empty_figure("No drop-in data for selected base-vs-policy pairs"),
            [],
            summary_columns,
        )

    confidence_level = float(confidence_level or 0.95)
    alpha = 1.0 - confidence_level
    metrics = prepare_policy_metrics(clean_df, confidence_level)
    metrics = _apply_sort_mode(metrics, sort_mode, pin_first=base_policy_selected)
    all_names = metrics["model_name"].astype(str).tolist()
    display_map, prefix = _make_display_names(all_names)
    active_group_values = _normalize_group_selection(active_testing_group)
    active_leaderboard_group_values = _normalize_group_selection(active_leaderboard_testing_group)

    _suffix_parts: list[str] = []
    if active_group_values:
        noun = "testing groups" if len(active_group_values) > 1 else "testing group"
        _suffix_parts.append(f"{noun}: {', '.join(active_group_values)} + Base")
    if prefix:
        _suffix_parts.append(f"common prefix: {prefix}")
    subtitle_suffix = f"<br><sup>{' | '.join(_suffix_parts)}</sup>" if _suffix_parts else ""

    policy_colors = _policy_color_map(all_names)

    leaderboard_df = metrics.copy()
    leaderboard_df["success_rate_pct"] = pd.to_numeric(leaderboard_df["success_rate"], errors="coerce") * 100.0
    leaderboard_df["quality_score_pct"] = pd.to_numeric(leaderboard_df.get("quality_score_pct"), errors="coerce")
    leaderboard_df["dropin_ratio_pct"] = pd.to_numeric(leaderboard_df.get("dropin_ratio_pct"), errors="coerce")
    applied_leaderboard_groups: list[str] = []

    if active_leaderboard_group_values:
        tag_order, group_to_models, base_models = _build_testing_group_index_for_models(clean_df, all_names)
        tag_set = set(tag_order)
        selected_leaderboard_groups = [group for group in active_leaderboard_group_values if group in tag_set]
        if selected_leaderboard_groups:
            applied_leaderboard_groups = selected_leaderboard_groups
            if not base_models and base_policy_selected in all_names:
                base_models = [str(base_policy_selected)]
            selected_models = _select_models_for_group_tags(
                selected_leaderboard_groups,
                base_models,
                group_to_models,
                all_names,
            )
            leaderboard_df = leaderboard_df[leaderboard_df["model_name"].astype(str).isin(selected_models)].copy()

    leaderboard_filter_status = (
        f"Leaderboard tag filter: {', '.join(applied_leaderboard_groups)} + Base"
        if applied_leaderboard_groups
        else "Leaderboard tag filter: all"
    )

    if leaderboard_df["quality_score_pct"].notna().any():
        leaderboard_df = leaderboard_df.sort_values(
            ["quality_score_pct", "success_rate_pct", "source_order"],
            ascending=[False, False, True],
            na_position="last",
            kind="stable",
        )
    else:
        leaderboard_df = leaderboard_df.sort_values(
            ["success_rate_pct", "source_order"],
            ascending=[False, True],
            kind="stable",
        )

    leaderboard_rows: list[dict[str, object]] = []
    for _, row in leaderboard_df.iterrows():
        policy_name = str(row["model_name"])
        leaderboard_rows.append(
            {
                "policy": display_map.get(policy_name, policy_name),
                "quality_score_pct": round(float(row["quality_score_pct"]), 2)
                if pd.notna(row.get("quality_score_pct"))
                else None,
                "success_rate_pct": round(float(row["success_rate_pct"]), 2)
                if pd.notna(row.get("success_rate_pct"))
                else None,
                "dropin_ratio_pct": round(float(row["dropin_ratio_pct"]), 2)
                if pd.notna(row.get("dropin_ratio_pct"))
                else None,
            }
        )

    leaderboard_styles: list[dict[str, object]] = []
    if base_policy_selected and base_policy_selected in set(metrics["model_name"].astype(str)):
        base_policy_display = display_map.get(str(base_policy_selected), str(base_policy_selected))
        safe_base = str(base_policy_display).replace('"', '\\"')
        leaderboard_styles.append(
            {
                "if": {"filter_query": f'{{policy}} = "{safe_base}"'},
                "fontWeight": "700",
                "backgroundColor": "#f5f5f5",
            }
        )

    selected_policies = [
        str(name)
        for name in (selected_policies or [])
        if str(name) in set(metrics["model_name"].astype(str))
    ]
    plot_df = metrics[metrics["model_name"].isin(selected_policies)].copy()

    if plot_df.empty:
        return (
            leaderboard_rows,
            leaderboard_styles,
            leaderboard_filter_status,
            _empty_figure("No quality-score data for selected base-vs-policy pairs"),
            _empty_figure("No drop-in data for selected base-vs-policy pairs"),
            [],
            summary_columns,
        )

    if base_policy_selected and base_policy_selected in set(metrics["model_name"].astype(str)):
        base_policy_for_pairs = base_policy_selected
    else:
        base_policy_for_pairs = str(plot_df["model_name"].iloc[0])

    pair_df_for_pairs = plot_df.copy()
    if base_policy_for_pairs not in set(pair_df_for_pairs["model_name"]) and base_policy_for_pairs in set(metrics["model_name"]):
        base_row = metrics.loc[metrics["model_name"] == base_policy_for_pairs]
        pair_df_for_pairs = pd.concat([base_row, pair_df_for_pairs], ignore_index=True)
    pair_df_for_pairs = pair_df_for_pairs.drop_duplicates(subset=["model_name"], keep="first").reset_index(drop=True)

    sr_pair_letters_df = pd.DataFrame()
    if len(pair_df_for_pairs) >= 2:
        sr_pair_letters_df = base_vs_policy_letter_pairs(
            pair_df_for_pairs,
            base_policy=base_policy_for_pairs,
            alpha=alpha,
            p_adjust_method=None,
        )

    quality_pair_rows: list[dict[str, object]] = []
    has_quality_std = (
        "quality_score_std_pct" in pair_df_for_pairs.columns
        and pd.to_numeric(pair_df_for_pairs.get("quality_score_std_pct"), errors="coerce").notna().any()
    )
    if has_quality_std and base_policy_for_pairs in set(pair_df_for_pairs["model_name"].astype(str)):
        base_quality_row = pair_df_for_pairs.loc[pair_df_for_pairs["model_name"] == base_policy_for_pairs].iloc[0]
        base_quality_mean = _coerce_scalar_float(base_quality_row.get("quality_score_pct"))
        base_quality_std = _coerce_scalar_float(base_quality_row.get("quality_score_std_pct"))
        base_quality_n = int(base_quality_row.get("trials", 0)) if pd.notna(base_quality_row.get("trials")) else 0

        if (
            base_quality_mean is not None
            and base_quality_std is not None
            and base_quality_std >= 0
            and base_quality_n > 1
        ):
            for _, policy_row in pair_df_for_pairs.iterrows():
                policy_name = str(policy_row["model_name"])
                if policy_name == base_policy_for_pairs:
                    continue

                policy_quality_mean = _coerce_scalar_float(policy_row.get("quality_score_pct"))
                policy_quality_std = _coerce_scalar_float(policy_row.get("quality_score_std_pct"))
                policy_quality_n = int(policy_row.get("trials", 0)) if pd.notna(policy_row.get("trials")) else 0

                if (
                    policy_quality_mean is None
                    or policy_quality_std is None
                    or policy_quality_std < 0
                    or policy_quality_n <= 1
                ):
                    continue

                _t, p_value, _dof, ci_low, ci_high = welch_t_test(
                    float(base_quality_mean),
                    float(base_quality_std),
                    int(base_quality_n),
                    float(policy_quality_mean),
                    float(policy_quality_std),
                    int(policy_quality_n),
                    confidence_level,
                )
                if not math.isfinite(p_value):
                    continue

                delta_pp = float(policy_quality_mean - base_quality_mean)
                is_significant = ci_low > 0 or ci_high < 0
                quality_pair_rows.append(
                    {
                        "base_policy": base_policy_for_pairs,
                        "policy": policy_name,
                        "delta_pp": delta_pp,
                        "pair_letters": "a-b" if is_significant else "a-a",
                        "is_significant": is_significant,
                    }
                )
    quality_pair_letters_df = pd.DataFrame(quality_pair_rows)

    dropin_pair_rows: list[dict[str, object]] = []
    has_dropin = (
        "dropin_count" in pair_df_for_pairs.columns
        and pd.to_numeric(pair_df_for_pairs.get("dropin_count"), errors="coerce").notna().any()
    )
    if has_dropin and base_policy_for_pairs in set(pair_df_for_pairs["model_name"].astype(str)):
        base_dropin_row = pair_df_for_pairs.loc[pair_df_for_pairs["model_name"] == base_policy_for_pairs].iloc[0]
        base_trials = int(base_dropin_row.get("trials", 0)) if pd.notna(base_dropin_row.get("trials")) else 0
        base_dropin_count = int(base_dropin_row.get("dropin_count", 0)) if pd.notna(base_dropin_row.get("dropin_count")) else 0
        if base_trials > 0:
            base_dropin_count = max(0, min(base_dropin_count, base_trials))
            for _, policy_row in pair_df_for_pairs.iterrows():
                policy_name = str(policy_row["model_name"])
                if policy_name == base_policy_for_pairs:
                    continue

                policy_trials = int(policy_row.get("trials", 0)) if pd.notna(policy_row.get("trials")) else 0
                policy_dropin_count = int(policy_row.get("dropin_count", 0)) if pd.notna(policy_row.get("dropin_count")) else 0
                if policy_trials <= 0:
                    continue
                policy_dropin_count = max(0, min(policy_dropin_count, policy_trials))

                ci_low, ci_high = delta_ci_newcombe_wilson(
                    base_dropin_count,
                    base_trials,
                    policy_dropin_count,
                    policy_trials,
                    confidence_level,
                )
                delta_pp = ((policy_dropin_count / policy_trials) - (base_dropin_count / base_trials)) * 100.0
                is_significant = ci_low > 0 or ci_high < 0
                dropin_pair_rows.append(
                    {
                        "base_policy": base_policy_for_pairs,
                        "policy": policy_name,
                        "delta_pp": delta_pp,
                        "pair_letters": "a-b" if is_significant else "a-a",
                        "is_significant": is_significant,
                    }
                )
    dropin_pair_letters_df = pd.DataFrame(dropin_pair_rows)

    quality_violin_fig = _build_base_vs_quality_pairs_violin(
        pair_df_for_pairs,
        base_policy=base_policy_for_pairs,
        quality_pair_letters_df=quality_pair_letters_df,
        policy_colors=policy_colors,
        display_names=display_map,
        title_suffix=subtitle_suffix,
    )
    dropin_violin_fig = _build_base_vs_dropin_pairs_violin(
        pair_df_for_pairs,
        base_policy=base_policy_for_pairs,
        dropin_pair_letters_df=dropin_pair_letters_df,
        policy_colors=policy_colors,
        display_names=display_map,
        title_suffix=subtitle_suffix,
    )

    sr_stats: dict[str, dict[str, object]] = {}
    for _, row in sr_pair_letters_df.iterrows():
        policy_name = str(row["policy"])
        sr_stats[policy_name] = {
            "delta_pp": float(row["delta"]) * 100.0,
            "is_significant": str(row.get("pair_letters", "")) == "a-b",
        }

    quality_stats: dict[str, dict[str, object]] = {}
    for _, row in quality_pair_letters_df.iterrows():
        policy_name = str(row["policy"])
        quality_stats[policy_name] = {
            "delta_pp": float(row.get("delta_pp", math.nan)),
            "is_significant": bool(row.get("is_significant", False)),
        }

    dropin_stats: dict[str, dict[str, object]] = {}
    for _, row in dropin_pair_letters_df.iterrows():
        policy_name = str(row["policy"])
        dropin_stats[policy_name] = {
            "delta_pp": float(row.get("delta_pp", math.nan)),
            "is_significant": bool(row.get("is_significant", False)),
        }

    significant_rows: list[dict[str, object]] = []
    pair_policies = [
        str(name)
        for name in pair_df_for_pairs["model_name"].astype(str).tolist()
        if str(name) != base_policy_for_pairs
    ]
    for policy_name in pair_policies:
        metric_notes: list[str] = []
        sr_delta = quality_delta = dropin_delta = None

        sr_entry = sr_stats.get(policy_name)
        if sr_entry and bool(sr_entry.get("is_significant")):
            sr_delta = round(float(sr_entry["delta_pp"]), 2)
            metric_notes.append("Success Rate ↑" if float(sr_entry["delta_pp"]) > 0 else "Success Rate ↓")

        quality_entry = quality_stats.get(policy_name)
        if quality_entry and bool(quality_entry.get("is_significant")):
            quality_delta = round(float(quality_entry["delta_pp"]), 2)
            metric_notes.append("Quality Score ↑" if float(quality_entry["delta_pp"]) > 0 else "Quality Score ↓")

        dropin_entry = dropin_stats.get(policy_name)
        if dropin_entry and bool(dropin_entry.get("is_significant")):
            dropin_delta = round(float(dropin_entry["delta_pp"]), 2)
            if float(dropin_entry["delta_pp"]) < 0:
                metric_notes.append("Drop-in ↓ (better)")
            else:
                metric_notes.append("Drop-in ↑ (worse)")

        if metric_notes:
            significant_rows.append(
                {
                    "base_policy": display_map.get(base_policy_for_pairs, base_policy_for_pairs),
                    "policy": display_map.get(policy_name, policy_name),
                    "significant_metrics": ", ".join(metric_notes),
                    "sr_delta_pp": sr_delta,
                    "quality_delta_pp": quality_delta,
                    "dropin_delta_pp": dropin_delta,
                    "_metric_count": len(metric_notes),
                }
            )

    significant_rows = sorted(
        significant_rows,
        key=lambda row: (-int(row.get("_metric_count", 0)), str(row.get("policy", ""))),
    )
    for row in significant_rows:
        row.pop("_metric_count", None)

    return (
        leaderboard_rows,
        leaderboard_styles,
        leaderboard_filter_status,
        quality_violin_fig,
        dropin_violin_fig,
        significant_rows,
        summary_columns,
    )


@app.callback(
    Output("summary-table", "data"),
    Output("summary-table", "columns"),
    Output("ab-output", "children"),
    Output("ab-common-legend", "children"),
    Output("ab-comparison-graph", "figure"),
    Output("ab-quality-graph", "figure"),
    Output("ab-dropin-graph", "figure"),
    Output("ab-failure-heatmap-graph", "figure"),
    Output("ab-violin-graph", "figure"),
    Output("ab-quality-violin-graph", "figure"),
    Output("ab-dropin-violin-graph", "figure"),
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
    Input("failure-detail-store", "data"),
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
    active_testing_group: list[str] | str | None,
    failure_store: dict | None,
    show_final_violin_toggle: list[str] | None,
    sort_mode: str | None,
    show_allvsall_violin_toggle: list[str] | None,
):
    clean_df = _raw_to_clean_df(rows)
    show_final_violin = "show" in (show_final_violin_toggle or [])
    show_allvsall_violin = "show" in (show_allvsall_violin_toggle or [])
    active_group_values = _normalize_group_selection(active_testing_group)
    active_group_label = ", ".join(active_group_values) if active_group_values else None
    final_violin_style = {"display": "block"} if show_final_violin else {"display": "none"}
    allvsall_violin_style = {"display": "block"} if show_allvsall_violin else {"display": "none"}
    ab_common_legend_default = html.Div(
        "Select Policy A and Policy B to show shared color legend.",
        style={"fontSize": "12px", "color": "#666"},
    )
    ab_metric_card_style = {
        "background": "#fafafa",
        "border": "1px solid #e0e0e0",
        "borderRadius": "6px",
        "padding": "8px 10px",
    }
    ab_overall_card_style = {
        "background": "#f5f5f5",
        "border": "1px solid #bdbdbd",
        "borderRadius": "6px",
        "padding": "8px 10px",
    }

    if clean_df.empty:
        return (
            [],
            [],
            "Load policy rows to start analysis.",
            ab_common_legend_default,
            _empty_figure("Pick two policies for A/B plot"),
            _empty_figure("No quality score data for selected A/B policies"),
            _empty_figure("No drop-in attempt data for selected A/B policies"),
            _failure_empty_figure("Load detailed rollout sheets to compare A/B condition heatmaps"),
            _empty_figure("Pick two policies for posterior uncertainty view"),
            _empty_figure("Pick two policies for A/B quality posterior view"),
            _empty_figure("Pick two policies for A/B drop-in posterior view"),
            _empty_figure("No policy data"),
            _empty_figure("No quality score data for selected policies"),
            _empty_figure("No drop-in ratio data for selected policies (lower is better)"),
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
            _format_sort_status(sort_mode, active_group=active_group_values),
        )

    analysis_df = clean_df.copy()
    if "has_success_rate_input" in analysis_df.columns:
        analysis_df = analysis_df[analysis_df["has_success_rate_input"].fillna(True)].copy()

    if analysis_df.empty:
        return (
            [],
            [],
            "No concluded policies available: success rate is empty for all rows.",
            ab_common_legend_default,
            _empty_figure("Pick two concluded policies for A/B plot"),
            _empty_figure("No quality score data for selected A/B policies"),
            _empty_figure("No drop-in attempt data for selected A/B policies"),
            _failure_empty_figure("Load detailed rollout sheets to compare A/B condition heatmaps"),
            _empty_figure("Pick two concluded policies for posterior uncertainty view"),
            _empty_figure("Pick two concluded policies for quality posterior view"),
            _empty_figure("Pick two concluded policies for drop-in posterior view"),
            _empty_figure("No concluded policies selected"),
            _empty_figure("No quality score data for selected policies"),
            _empty_figure("No drop-in ratio data for selected policies (lower is better)"),
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
            _format_sort_status(sort_mode, active_group=active_group_values),
        )

    metrics = prepare_policy_metrics(analysis_df, confidence_level)
    metrics = _apply_sort_mode(metrics, sort_mode, pin_first=policy_a)
    all_names = metrics["model_name"].astype(str).tolist()
    display_map, prefix = _make_display_names(all_names)
    _multi_sub_parts: list[str] = []
    if active_group_label:
        if len(active_group_values) > 1:
            _multi_sub_parts.append(f"testing groups: {active_group_label} + Base")
        else:
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
        {"name": "Policy", "id": "model_name"},
        {"name": "Successes", "id": "successes"},
        {"name": "Trials", "id": "trials"},
        {"name": "SR [%]", "id": "success_rate"},
        {"name": "SR CI Low [%]", "id": "wilson_low"},
        {"name": "SR CI High [%]", "id": "wilson_high"},
    ]
    if "quality_score_pct" in summary.columns:
        summary_columns.append({"name": "Quality [%]", "id": "quality_score_pct"})
    if has_quality_std:
        summary_columns.append({"name": "Quality STD", "id": "quality_score_std_pct"})
        summary_columns.append({"name": "Q CI Low [%]", "id": "quality_ci_low"})
        summary_columns.append({"name": "Q CI High [%]", "id": "quality_ci_high"})
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
        summary_columns.append({"name": "Drop-in [%] ↓", "id": "dropin_ratio_pct"})
        summary_columns.append({"name": "Drop-in CI Low [%]", "id": "dropin_wilson_low"})
        summary_columns.append({"name": "Drop-in CI High [%]", "id": "dropin_wilson_high"})

    ab_output = "Pick two policies to compare."
    ab_common_legend = ab_common_legend_default
    ab_fig = _empty_figure("Pick two policies for A/B plot")
    ab_quality_fig = _empty_figure("No quality score data for selected A/B policies")
    ab_dropin_fig = _empty_figure("No drop-in attempt data for selected A/B policies")
    ab_failure_heatmap_fig = _failure_empty_figure("Load detailed rollout sheets to compare A/B condition heatmaps")
    ab_violin_fig = _empty_figure("Pick two policies for A/B posterior view")
    ab_quality_violin_fig = _empty_figure("Pick two policies for A/B quality posterior view")
    ab_dropin_violin_fig = _empty_figure("Pick two policies for A/B drop-in posterior view")

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
                html.Div("SR [%] ↑", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "2px"}),
                html.Div(
                    f"\u0394 (B \u2212 A): {delta * 100:.2f} pp | "
                    f"CI: [{delta_low * 100:.2f}, {delta_high * 100:.2f}] pp",
                    style={"fontSize": "12px"},
                ),
                html.Div(
                    decision,
                    style={"fontWeight": "600", "marginTop": "2px", "color": decision_color, "fontSize": "12px"},
                ),
            ],
            style=ab_metric_card_style,
        )

        ab_policies = [policy_a, policy_b]
        ab_df = metrics.set_index("model_name").loc[ab_policies].reset_index()
        _ab_short_names = [display_map.get(n, n) for n in ab_df["model_name"]]
        ab_quality_letters = {policy_a: "a", policy_b: "a"}
        ab_dropin_letters = {policy_a: "a", policy_b: "a"}
        ab_quality_violin_fig = _empty_figure("No quality-score uncertainty data for selected A/B policies")
        ab_dropin_violin_fig = _empty_figure("No drop-in uncertainty data for selected A/B policies")
        _legend_entries = []
        for label, policy_name, role in [
            ("Policy A", policy_a, "base"),
            ("Policy B", policy_b, "experimental"),
        ]:
            _legend_entries.append(
                html.Div(
                    [
                        html.Span("●", style={"color": policy_colors.get(policy_name, "#1f77b4"), "fontSize": "16px"}),
                        html.Span(
                            f" {label}: {display_map.get(policy_name, policy_name)} ({role})",
                            style={"fontSize": "13px"},
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center", "gap": "4px"},
                )
            )

        _legend_notes = [
            html.Span("Success/Quality ↑ | Drop-in ↓", style={"fontSize": "12px", "color": "#666"})
        ]
        if prefix:
            _legend_notes.append(
                html.Span(f"Common prefix stripped: {prefix}", style={"fontSize": "12px", "color": "#666"})
            )

        ab_common_legend = html.Div(
            [
                html.Div(
                    _legend_entries,
                    style={"display": "flex", "flexWrap": "wrap", "gap": "14px", "alignItems": "center"},
                ),
                html.Div(
                    _legend_notes,
                    style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "marginTop": "4px"},
                ),
            ]
        )

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
            yaxis_title="Success Rate [%] ↑",
            xaxis_title="",
            title=None,
            yaxis_range=[0, min(105, max(5, math.ceil((ab_df["wilson_high"].max() * 100) / 5) * 5 + 5))],
            bargap=0.0,
            bargroupgap=0.0,
            height=320,
            margin={"l": 45, "r": 20, "t": 20, "b": 30},
        )
        ab_fig.update_xaxes(showticklabels=False)

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
                yaxis_title="Quality Score [%] ↑",
                xaxis_title="",
                title=None,
                yaxis_range=[0, min(105, max(5, math.ceil(q_max_y / 5) * 5 + 5))],
                bargap=0.0,
                bargroupgap=0.0,
                height=320,
                margin={"l": 45, "r": 20, "t": 20, "b": 30},
            )
            ab_quality_fig.update_xaxes(showticklabels=False)

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
                        ab_quality_letters[policy_b] = "b"
                    elif _dch < 0:
                        _qdec = "B has significantly lower quality."
                        _qcolor = "#c62828"
                        q_better = False
                        q_worse = True
                        ab_quality_letters[policy_b] = "b"
                    else:
                        _qdec = "Inconclusive quality difference."
                        _qcolor = "#9e9e9e"
                        q_better = False
                        q_worse = False

                    quality_card = html.Div(
                        [
                            html.Div("Quality [%] ↑", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "2px"}),
                            html.Div(
                                f"\u0394 (B \u2212 A): {_dq:.2f} pp, "
                                f"CI: [{_dcl:.2f}, {_dch:.2f}] pp",
                                style={"fontSize": "12px"},
                            ),
                            html.Div(_qdec, style={"fontWeight": "600", "marginTop": "2px", "color": _qcolor, "fontSize": "12px"}),
                        ],
                        style=ab_metric_card_style,
                    )

                ab_quality_violin_fig = _build_quality_posterior_violin(
                    ab_df,
                    ab_quality_letters,
                    policy_colors,
                    "",
                    display_names=display_map,
                )
                ab_quality_violin_fig.update_layout(
                    height=320,
                    margin={"l": 45, "r": 20, "t": 20, "b": 30},
                    title=None,
                    violingap=0.0,
                    violingroupgap=0.0,
                )
                ab_quality_violin_fig.update_traces(width=0.9, selector={"type": "violin"})
                ab_quality_violin_fig.update_xaxes(categoryorder="array", categoryarray=_ab_short_names)
                ab_quality_violin_fig.update_xaxes(showticklabels=False, title_text="")
                ab_quality_violin_fig.update_yaxes(title_text="Quality Score [%] ↑")

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
                yaxis_title="Drop-in Attempt [%] ↓",
                xaxis_title="",
                title=None,
                yaxis_range=[0, min(105, max(5, math.ceil(_di_max_y_ab / 5) * 5 + 5))],
                bargap=0.0,
                bargroupgap=0.0,
                height=320,
                margin={"l": 45, "r": 20, "t": 20, "b": 30},
            )
            ab_dropin_fig.update_xaxes(showticklabels=False)

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
                ab_dropin_letters[policy_b] = "b"
            elif _di_dlow > 0:
                _didec = "B has significantly higher drop-in ratio (worse)."
                _dicolor = "#c62828"
                di_worse = True
                ab_dropin_letters[policy_b] = "b"
            else:
                _didec = "Inconclusive drop-in ratio difference."
                _dicolor = "#9e9e9e"

            ab_dropin_violin_fig = _build_dropin_posterior_violin(
                ab_df,
                ab_dropin_letters,
                policy_colors,
                "",
                display_names=display_map,
            )
            ab_dropin_violin_fig.update_layout(
                height=320,
                margin={"l": 45, "r": 20, "t": 20, "b": 30},
                title=None,
                violingap=0.0,
                violingroupgap=0.0,
            )
            ab_dropin_violin_fig.update_traces(width=0.9, selector={"type": "violin"})
            ab_dropin_violin_fig.update_xaxes(categoryorder="array", categoryarray=_ab_short_names)
            ab_dropin_violin_fig.update_xaxes(showticklabels=False, title_text="")
            ab_dropin_violin_fig.update_yaxes(title_text="Drop-in Attempt [%] ↓")

            dropin_card = html.Div(
                [
                    html.Div("Drop-in [%] ↓", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "2px"}),
                    html.Div(
                        f"\u0394 (B \u2212 A): {_di_delta * 100:.2f} pp, "
                        f"CI: [{_di_dlow * 100:.2f}, {_di_dhigh * 100:.2f}] pp",
                        style={"fontSize": "12px"},
                    ),
                    html.Div(_didec, style={"fontWeight": "600", "marginTop": "2px", "color": _dicolor, "fontSize": "12px"}),
                ],
                style=ab_metric_card_style,
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
                html.Div(_ov, style={"fontWeight": "600", "color": _oc, "fontSize": "12px"}),
                style=ab_overall_card_style,
            )
            _cards = [ab_output]
            if quality_card is not None:
                _cards.append(quality_card)
            if dropin_card is not None:
                _cards.append(dropin_card)
            _cards.append(overall_card)
            ab_output = html.Div(_cards, style={"display": "grid", "gap": "6px"})

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
            "",
            display_names=display_map,
        )
        ab_violin_fig.update_layout(
            height=320,
            margin={"l": 45, "r": 20, "t": 20, "b": 30},
            title=None,
            violingap=0.0,
            violingroupgap=0.0,
        )
        ab_violin_fig.update_traces(width=0.9, selector={"type": "violin"})
        ab_violin_fig.update_xaxes(categoryorder="array", categoryarray=_ab_short_names)
        ab_violin_fig.update_xaxes(showticklabels=False, title_text="")
        ab_violin_fig.update_yaxes(title_text="Success Rate [%] ↑")

    if policy_a and policy_b and failure_store and failure_store.get("records"):
        ab_failure_df = pd.DataFrame(failure_store.get("records") or [])
        if not ab_failure_df.empty and {"policy_name", "task_success"}.issubset(set(ab_failure_df.columns)):
            condition_columns = {
                str(entry.get("key", "")): str(entry.get("label", ""))
                for entry in (failure_store.get("condition_columns") or [])
                if str(entry.get("key", "")).strip()
            }
            x_key = str(failure_store.get("default_x") or "")
            y_key = str(failure_store.get("default_y") or "")
            available_keys = [key for key in condition_columns if key in ab_failure_df.columns]
            if (not x_key or x_key not in ab_failure_df.columns) and available_keys:
                x_key = available_keys[0]
            if (
                (not y_key or y_key not in ab_failure_df.columns or y_key == x_key)
                and len(available_keys) > 1
            ):
                y_key = next((key for key in available_keys if key != x_key), "")

            if x_key and y_key and x_key in ab_failure_df.columns and y_key in ab_failure_df.columns:
                ab_failure_df = ab_failure_df.copy()
                ab_failure_df["policy_name"] = ab_failure_df["policy_name"].astype(str)
                ab_failure_df["task_success"] = pd.to_numeric(ab_failure_df["task_success"], errors="coerce")
                ab_failure_df = ab_failure_df[ab_failure_df["task_success"].notna()].copy()
                ab_failure_df = ab_failure_df[ab_failure_df["policy_name"].isin([policy_a, policy_b])].copy()
                if not ab_failure_df.empty:
                    ab_failure_df[x_key] = ab_failure_df[x_key].fillna("NA").astype(str)
                    ab_failure_df[y_key] = ab_failure_df[y_key].fillna("NA").astype(str)
                    if "quality_score_pct" not in ab_failure_df.columns:
                        ab_failure_df["quality_score_pct"] = pd.NA
                    ab_grouped = (
                        ab_failure_df.groupby(["policy_name", x_key, y_key], as_index=False)
                        .agg(
                            n=("task_success", "size"),
                            success_rate=("task_success", "mean"),
                            quality_score_pct=("quality_score_pct", "mean"),
                        )
                    )
                    ab_grouped["failure_rate"] = 1.0 - ab_grouped["success_rate"]
                    ab_grouped["success_rate_pct"] = ab_grouped["success_rate"] * 100.0
                    ab_grouped["failure_rate_pct"] = ab_grouped["failure_rate"] * 100.0

                    x_values = ab_failure_df[x_key].astype(str).drop_duplicates().tolist()
                    y_values = ab_failure_df[y_key].astype(str).drop_duplicates().tolist()
                    pair_policies = [name for name in [policy_a, policy_b] if name in set(ab_grouped["policy_name"].astype(str))]
                    if len(pair_policies) == 2:
                        ab_failure_heatmap_fig = _build_failure_policy_grid_figure(
                            ab_grouped,
                            policy_names=pair_policies,
                            x_values=x_values,
                            y_values=y_values,
                            x_key=x_key,
                            y_key=y_key,
                            metric_key="success_rate_pct",
                            metric_label="Success rate (%)",
                            colorscale="RdYlGn",
                            zmin=0.0,
                            zmax=100.0,
                            display_names=display_map,
                            n_cols=2,
                            title="A/B condition heatmap comparison (Success rate %)",
                            height=340,
                            share_yaxes=True,
                        )
                    else:
                        ab_failure_heatmap_fig = _failure_empty_figure(
                            "Selected A/B policies do not both have rollout-detail heatmap rows"
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
        dropin_fig = _empty_figure("No drop-in ratio data for selected policies (lower is better)")
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
                yaxis_title="Attempt Drop-in Ratio (%) (lower is better)",
                xaxis_title="Policy",
                title=f"Selected policies attempt drop-in (lower is better) with {int(confidence_level * 100)}% Wilson CIs{_multi_sub}",
                yaxis_range=[0, min(105, max(5, math.ceil(_mp_di_max / 5) * 5 + 5))],
            )
        else:
            dropin_fig = _empty_figure("No drop-in ratio data for selected policies (lower is better)")

        # SR vs Quality scatter plot
        _q_vals_scatter = pd.to_numeric(plot_df.get("quality_score_pct"), errors="coerce")
        if _q_vals_scatter.notna().any():
            sr_vs_q_fig = go.Figure()
            for _, _srow in plot_df.iterrows():
                _sname = str(_srow["model_name"])
                _sshort = display_map.get(_sname, _sname)
                _sr = float(_srow["success_rate"]) * 100
                _qv = float(_srow.get("quality_score_pct", math.nan)) if pd.notna(_srow.get("quality_score_pct")) else math.nan
                if not math.isfinite(_qv):
                    continue
                _sc = policy_colors.get(_sname, "#1f77b4")
                _trace_kw: dict = dict(
                    x=[_sr],
                    y=[_qv],
                    mode="markers+text",
                    marker=dict(color=_sc, size=12),
                    text=[_sshort],
                    textposition="top center",
                    name=_sshort,
                    showlegend=False,
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

    sort_status = _format_sort_status(sort_mode, prefix=prefix, active_group=active_group_values)

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

    summary_column_ids = [
        str(column.get("id"))
        for column in summary_columns
        if str(column.get("id", "")).strip()
    ]
    summary_table_df = summary[[column_id for column_id in summary_column_ids if column_id in summary.columns]].copy()
    summary_table_data = summary_table_df.to_dict("records")

    return (
        summary_table_data,
        summary_columns,
        ab_output,
        ab_common_legend,
        ab_fig,
        ab_quality_fig,
        ab_dropin_fig,
        ab_failure_heatmap_fig,
        ab_violin_fig,
        ab_quality_violin_fig,
        ab_dropin_violin_fig,
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
