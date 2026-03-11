from __future__ import annotations

import hashlib
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
from dash import dash_table

from data_utils import (
    CSV_SINGLE_SHEET_NAME,
    DEFAULT_TRIALS,
    list_local_spreadsheet_sheets,
    load_local_spreadsheet,
    normalize_policy_dataframe,
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
    "success_rate": "Sorted by success rate",
    "quality_score": "Sorted by Quality Score [%]",
}

QUALITY_SCORE_COLUMN_CANDIDATES = ["Quality Score [%]", "Quality Score", "QualityScore", "Quality"]
QUALITY_SCORE_STD_COLUMN_CANDIDATES = [
    "Quality Score STD [%]",
    "Quality Score STD",
    "QualityScoreSTD",
    "Quality STD",
]
SUCCESS_RATE_INPUT_CANDIDATES = ["Success Rate [%]", "Success Rate", "Success_Rate", "Rate", "Accuracy"]
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
                "has_success_rate_input",
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

    success_rate_input_col = _find_column(df, SUCCESS_RATE_INPUT_CANDIDATES)
    if success_rate_input_col is not None:
        df["has_success_rate_input"] = _percent_like_to_numeric(df[success_rate_input_col]).notna()
    else:
        df["has_success_rate_input"] = True

    grouped = (
        df.groupby("model_name", as_index=False)
        .agg(
            {
                "successes": "sum",
                "trials": "sum",
                "source_order": "min",
                "quality_score_pct": "mean",
                "quality_score_std_pct": "first",
                "has_success_rate_input": "any",
            }
        )
        .sort_values("source_order", kind="stable")
        .reset_index(drop=True)
    )
    return grouped


def _apply_sort_mode(metrics: pd.DataFrame, sort_mode: str | None) -> pd.DataFrame:
    mode = sort_mode or "original"
    if mode == "success_rate":
        return metrics.sort_values(["success_rate", "source_order"], ascending=[False, True], kind="stable").reset_index(drop=True)

    if mode == "quality_score" and "quality_score_pct" in metrics.columns:
        return metrics.sort_values(["quality_score_pct", "source_order"], ascending=[False, True], na_position="last", kind="stable").reset_index(drop=True)

    if "source_order" in metrics.columns:
        return metrics.sort_values("source_order", ascending=True, kind="stable").reset_index(drop=True)

    return metrics.reset_index(drop=True)


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
        title="Base-vs-policy posterior uncertainty with pair letters",
        yaxis_range=[0, 105],
        annotations=annotations,
    )
    return fig


app = Dash(__name__)
app.title = "Robot Policy Evaluation Dashboard"

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px"},
    children=[
        dcc.Store(id="uploaded-file-store", data=None),
        dcc.Store(id="sort-mode-store", data="original"),
        html.H2("Robot Policy Evaluation Dashboard"),
        html.P("Upload a local CSV/XLSX, edit/log rollout results, and compare policies with Wilson confidence intervals."),
        html.Hr(),
        html.Div(
            style={"display": "flex", "gap": "10px", "alignItems": "center"},
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
        dcc.Graph(id="ab-comparison-graph"),
        dcc.Graph(id="ab-quality-graph"),
        dcc.Graph(id="ab-violin-graph"),
        html.Hr(),
        html.H4("Multi-policy comparison + compact letter display"),
        html.Div(
            style={"display": "flex", "gap": "10px", "alignItems": "center"},
            children=[
                html.Button("Select All", id="select-all-btn"),
                html.Button("Deselect All", id="deselect-all-btn"),
                html.Button("Original Order", id="sort-original-btn"),
                html.Button("Sort by Success Rate", id="sort-success-btn"),
                html.Button("Sort by Quality Score [%]", id="sort-quality-btn"),
            ],
        ),
        html.Div(id="sort-status", style={"marginTop": "6px", "fontSize": "13px"}),
        dcc.Checklist(id="policy-checklist", options=[], value=[], inline=True, style={"marginTop": "8px"}),
        dcc.Checklist(
            id="show-final-violin-toggle",
            options=[{"label": "Show final base-vs-policy violin panel", "value": "show"}],
            value=["show"],
            inline=True,
            style={"marginTop": "8px"},
        ),
        dcc.Graph(id="performance-graph"),
        dcc.Graph(id="quality-score-graph"),
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
    ],
)


@app.callback(
    Output("sort-mode-store", "data"),
    Input("sort-original-btn", "n_clicks"),
    Input("sort-success-btn", "n_clicks"),
    Input("sort-quality-btn", "n_clicks"),
    State("sort-mode-store", "data"),
    prevent_initial_call=True,
)
def update_sort_mode(
    _sort_original_clicks: int,
    _sort_success_clicks: int,
    _sort_quality_clicks: int,
    current_mode: str | None,
):
    trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
    if trigger == "sort-success-btn":
        return "success_rate"
    if trigger == "sort-quality-btn":
        return "quality_score"
    if trigger == "sort-original-btn":
        return "original"
    return current_mode or "original"


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
    State("local-file-upload", "filename"),
    State("uploaded-file-store", "data"),
    prevent_initial_call=True,
)
def load_file_to_table(
    upload_contents: str | None,
    selected_sheet_name: str | None,
    filename: str | None,
    stored_upload: dict | None,
):
    trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None

    if trigger == "local-file-upload":
        if not upload_contents:
            return no_update, no_update, "Upload a CSV/XLSX file first.", no_update, no_update, no_update, no_update

        try:
            sheets = list_local_spreadsheet_sheets(upload_contents, filename)
        except Exception as exc:
            return no_update, no_update, f"Error: {exc}", no_update, no_update, no_update, no_update

        default_sheet = sheets[0] if sheets else CSV_SINGLE_SHEET_NAME
        upload_state = {
            "contents": upload_contents,
            "filename": filename,
            "sheets": sheets,
        }
    else:
        if not stored_upload or "contents" not in stored_upload:
            return no_update, no_update, "Upload a CSV/XLSX file first.", no_update, no_update, no_update, no_update
        upload_state = stored_upload
        sheets = upload_state.get("sheets") or [CSV_SINGLE_SHEET_NAME]
        default_sheet = selected_sheet_name or sheets[0]

    if selected_sheet_name and selected_sheet_name in sheets:
        sheet_name = selected_sheet_name
    else:
        sheet_name = default_sheet

    options = [{"label": name, "value": name} for name in sheets]
    disabled = len(sheets) <= 1

    try:
        df = load_local_spreadsheet(upload_state["contents"], upload_state.get("filename"), sheet_name=sheet_name)
        normalized = normalize_policy_dataframe(df)
    except Exception as exc:
        return no_update, no_update, f"Error: {exc}", upload_state, options, sheet_name, disabled

    columns = [{"name": col, "id": col, "editable": col not in {"success_rate", "wilson_low", "wilson_high"}} for col in normalized.columns]
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
    Input("raw-table", "data"),
    Input("select-all-btn", "n_clicks"),
    Input("deselect-all-btn", "n_clicks"),
    State("policy-a-dropdown", "value"),
    State("policy-b-dropdown", "value"),
    State("policy-checklist", "value"),
)
def sync_policy_selectors(
    rows: list[dict] | None,
    _select_all_clicks: int,
    _deselect_all_clicks: int,
    current_a: str | None,
    current_b: str | None,
    current_checked: list[str] | None,
):
    clean_df = _raw_to_clean_df(rows)
    if "has_success_rate_input" in clean_df.columns and not clean_df.empty:
        eligible_models = clean_df.loc[clean_df["has_success_rate_input"].fillna(True), "model_name"].astype(str).tolist()
    else:
        eligible_models = clean_df["model_name"].astype(str).tolist() if not clean_df.empty else []

    models = eligible_models
    display_map, _pfx = _make_display_names(models)
    options = [{"label": display_map.get(model, model), "value": model} for model in models]

    trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else "raw-table"

    if current_a in models:
        policy_a = current_a
    else:
        policy_a = models[0] if models else None

    remaining_for_b = [m for m in models if m != policy_a]
    if current_b in models and current_b != policy_a:
        policy_b = current_b
    else:
        policy_b = remaining_for_b[0] if remaining_for_b else (models[0] if models else None)

    current_checked = current_checked or []
    if trigger == "select-all-btn":
        checked = eligible_models
    elif trigger == "deselect-all-btn":
        checked = []
    else:
        checked = [m for m in current_checked if m in models]
        if not checked:
            checked = eligible_models

    return options, policy_a, options, policy_b, options, checked


@app.callback(
    Output("summary-table", "data"),
    Output("summary-table", "columns"),
    Output("ab-output", "children"),
    Output("ab-comparison-graph", "figure"),
    Output("ab-quality-graph", "figure"),
    Output("ab-violin-graph", "figure"),
    Output("performance-graph", "figure"),
    Output("quality-score-graph", "figure"),
    Output("sr-vs-quality-graph", "figure"),
    Output("final-violin-wrapper", "style"),
    Output("cld-violin-graph", "figure"),
    Output("cld-table", "data"),
    Output("cld-table", "columns"),
    Output("cld-table", "style_data_conditional"),
    Output("sort-status", "children"),
    Input("raw-table", "data"),
    Input("confidence-level", "value"),
    Input("policy-a-dropdown", "value"),
    Input("policy-b-dropdown", "value"),
    Input("policy-checklist", "value"),
    Input("show-final-violin-toggle", "value"),
    Input("sort-mode-store", "data"),
)
def update_analysis(
    rows: list[dict] | None,
    confidence_level: float,
    policy_a: str | None,
    policy_b: str | None,
    selected_policies: list[str] | None,
    show_final_violin_toggle: list[str] | None,
    sort_mode: str | None,
):
    clean_df = _raw_to_clean_df(rows)
    show_final_violin = "show" in (show_final_violin_toggle or [])
    final_violin_style = {"display": "block"} if show_final_violin else {"display": "none"}

    if clean_df.empty:
        return (
            [],
            [],
            "Add policy rows to start analysis.",
            _empty_figure("Pick two policies for A/B plot"),
            _empty_figure("No quality score data for selected A/B policies"),
            _empty_figure("Pick two policies for posterior uncertainty view"),
            _empty_figure("No policy data"),
            _empty_figure("No quality score data for selected policies"),
            _empty_figure("No quality data for scatter view"),
            final_violin_style,
            _empty_figure("No policy data"),
            [],
            [],
            [],
            f"Order mode: {SORT_MODE_LABELS.get(sort_mode or 'original', 'Original sheet order')}",
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
            _empty_figure("Pick two concluded policies for posterior uncertainty view"),
            _empty_figure("No concluded policies selected"),
            _empty_figure("No quality score data for selected policies"),
            _empty_figure("No quality data for scatter view"),
            final_violin_style,
            _empty_figure("No concluded policies selected"),
            [],
            [],
            [],
            f"Order mode: {SORT_MODE_LABELS.get(sort_mode or 'original', 'Original sheet order')}",
        )

    metrics = prepare_policy_metrics(analysis_df, confidence_level)
    metrics = _apply_sort_mode(metrics, sort_mode)
    all_names = metrics["model_name"].astype(str).tolist()
    display_map, prefix = _make_display_names(all_names)
    _prefix_sub = f"<br><sup>common prefix: {prefix}</sup>" if prefix else ""
    policy_colors = _policy_color_map(all_names)
    alpha = 1.0 - confidence_level
    has_quality_std = (
        "quality_score_std_pct" in metrics.columns
        and pd.to_numeric(metrics.get("quality_score_std_pct"), errors="coerce").notna().any()
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

    ab_output = "Pick two policies to compare."
    ab_fig = _empty_figure("Pick two policies for A/B plot")
    ab_quality_fig = _empty_figure("No quality score data for selected A/B policies")
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

                    # Combined verdict
                    if sr_better and q_better:
                        _ov = "Both metrics favor B over A."
                        _oc = "#2e7d32"
                    elif sr_worse and q_worse:
                        _ov = "Both metrics favor A over B."
                        _oc = "#c62828"
                    elif sr_better and q_worse:
                        _ov = "Trade-off: B has higher success rate but lower quality."
                        _oc = "#e65100"
                    elif sr_worse and q_better:
                        _ov = "Trade-off: B has lower success rate but higher quality."
                        _oc = "#e65100"
                    elif sr_better:
                        _ov = "B has higher success rate; quality is inconclusive."
                        _oc = "#2e7d32"
                    elif sr_worse:
                        _ov = "B has lower success rate; quality is inconclusive."
                        _oc = "#c62828"
                    elif q_better:
                        _ov = "Success rate is inconclusive; B has higher quality."
                        _oc = "#2e7d32"
                    elif q_worse:
                        _ov = "Success rate is inconclusive; B has lower quality."
                        _oc = "#c62828"
                    else:
                        _ov = "No significant difference on either metric."
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
                    ab_output = html.Div([ab_output, quality_card, overall_card])

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
            title=f"Selected policies with {int(confidence_level * 100)}% Wilson CIs{_prefix_sub}",
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
                title="Selected policies quality score" + (f" with {int(confidence_level * 100)}% CIs" if mp_has_q_std else "") + _prefix_sub,
                yaxis_range=[0, min(105, max(5, math.ceil(mp_q_max / 5) * 5 + 5))],
            )
        else:
            quality_fig = _empty_figure("No quality score data for selected policies")

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
                title=f"Success Rate vs Quality Score{_prefix_sub}",
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
            )
        else:
            violin_fig = go.Figure()

    sort_status = f"Order mode: {SORT_MODE_LABELS.get(sort_mode or 'original', 'Original sheet order')}"
    if prefix:
        sort_status += f" | common prefix: {prefix}"

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

    return (
        summary.to_dict("records"),
        summary_columns,
        ab_output,
        ab_fig,
        ab_quality_fig,
        ab_violin_fig,
        fig,
        quality_fig,
        sr_vs_q_fig,
        final_violin_style,
        violin_fig,
        cld_data,
        cld_columns,
        cld_row_styles,
        sort_status,
    )


if __name__ == "__main__":
    app.run(debug=True)
