from __future__ import annotations

import base64
import io
import re
from typing import Iterable

import pandas as pd


DEFAULT_TRIALS = 44
CSV_SINGLE_SHEET_NAME = "CSV"
MODEL_COLUMN_CANDIDATES = ["Model Name", "Model", "Policy", "Policy Name"]
SUCCESSES_COLUMN_CANDIDATES = ["Successes", "Success Count", "Num Successes", "Wins"]
TRIALS_COLUMN_CANDIDATES = ["Trials", "Rollouts", "Attempts", "N", "Num Trials"]
SUCCESS_RATE_COLUMN_CANDIDATES = ["Success Rate [%]", "Success Rate", "Success_Rate", "Rate", "Accuracy"]
HEADER_GROUPS = {
    "model": MODEL_COLUMN_CANDIDATES,
    "successes": SUCCESSES_COLUMN_CANDIDATES,
    "trials": TRIALS_COLUMN_CANDIDATES,
    "success_rate": SUCCESS_RATE_COLUMN_CANDIDATES,
}


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(col).lower()): str(col) for col in df.columns
    }
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    return None


def _normalize_header_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _make_unique_columns(columns: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique: list[str] = []

    for idx, col in enumerate(columns):
        base = str(col).strip()
        if not base:
            base = f"col_{idx + 1}"

        counter = counts.get(base, 0)
        counts[base] = counter + 1

        if counter == 0:
            unique.append(base)
        else:
            unique.append(f"{base}_{counter + 1}")

    return unique


def _promote_header_row(df: pd.DataFrame, header_row: int) -> pd.DataFrame:
    header_values = df.iloc[header_row].tolist()
    fallback_columns = [str(col) for col in df.columns]

    candidate_columns: list[str] = []
    for idx, value in enumerate(header_values):
        label = "" if pd.isna(value) else str(value).strip()
        if not label or _normalize_header_token(label).startswith("unnamed"):
            fallback = fallback_columns[idx] if idx < len(fallback_columns) else ""
            fallback = fallback.strip()
            if fallback and not _normalize_header_token(fallback).startswith("unnamed"):
                label = fallback
            else:
                label = f"col_{idx + 1}"
        candidate_columns.append(label)

    promoted = df.iloc[header_row + 1 :].copy()
    promoted.columns = _make_unique_columns(candidate_columns)
    promoted = promoted.dropna(how="all").reset_index(drop=True)
    return promoted


def _detect_header_row(df: pd.DataFrame, max_scan_rows: int = 35) -> int | None:
    if df.empty:
        return None

    normalized_groups = {
        group: {_normalize_header_token(value) for value in values}
        for group, values in HEADER_GROUPS.items()
    }

    best_row: int | None = None
    best_rank: tuple[int, int, int] = (-1, -1, -1)

    scan_rows = min(max_scan_rows, len(df))
    for row_index in range(scan_rows):
        row_values = df.iloc[row_index].tolist()
        tokens = [_normalize_header_token(value) for value in row_values if _normalize_header_token(value)]

        matched_groups: set[str] = set()
        for token in tokens:
            for group, aliases in normalized_groups.items():
                if token in aliases:
                    matched_groups.add(group)

        has_model = "model" in matched_groups
        has_metric = any(metric in matched_groups for metric in ["successes", "trials", "success_rate"])
        non_empty = len(tokens)
        rank = (1 if (has_model and has_metric) else 0, len(matched_groups), non_empty)

        if rank > best_rank:
            best_rank = rank
            best_row = row_index

    if best_row is None:
        return None
    if best_rank[0] == 0:
        return None
    return best_row


def _promote_header_from_any_row_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    header_row = _detect_header_row(df)
    if header_row is None:
        return df
    return _promote_header_row(df, header_row)


def _decode_upload_contents(upload_contents: str) -> bytes:
    if not upload_contents:
        raise ValueError("No file content received. Upload a CSV/XLSX file.")

    if "," not in upload_contents:
        raise ValueError("Invalid upload payload.")

    _prefix, encoded = upload_contents.split(",", 1)
    return base64.b64decode(encoded)


def list_local_spreadsheet_sheets(upload_contents: str, filename: str | None) -> list[str]:
    raw_bytes = _decode_upload_contents(upload_contents)
    filename = (filename or "").lower()

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        excel = pd.ExcelFile(io.BytesIO(raw_bytes))
        return excel.sheet_names

    return [CSV_SINGLE_SHEET_NAME]


def load_local_spreadsheet(
    upload_contents: str,
    filename: str | None,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    raw_bytes = _decode_upload_contents(upload_contents)
    filename = (filename or "").lower()

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(io.StringIO(raw_bytes.decode("utf-8-sig")))
        except UnicodeDecodeError:
            return pd.read_csv(io.StringIO(raw_bytes.decode("latin-1")))

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        requested_sheet = None if sheet_name in {None, "", CSV_SINGLE_SHEET_NAME} else sheet_name
        excel = pd.ExcelFile(io.BytesIO(raw_bytes))
        if requested_sheet is not None and requested_sheet not in excel.sheet_names:
            raise ValueError(
                f"Sheet '{requested_sheet}' not found. Available sheets: {', '.join(excel.sheet_names)}"
            )
        return pd.read_excel(io.BytesIO(raw_bytes), sheet_name=requested_sheet if requested_sheet is not None else 0)

    try:
        return pd.read_csv(io.StringIO(raw_bytes.decode("utf-8-sig")))
    except Exception as csv_exc:
        try:
            return pd.read_excel(io.BytesIO(raw_bytes))
        except Exception as xlsx_exc:
            raise ValueError(
                "Unsupported file type. Upload CSV or XLSX exported from Google Sheets. "
                f"CSV parse error: {csv_exc}. XLSX parse error: {xlsx_exc}."
            ) from xlsx_exc


def normalize_policy_dataframe(df: pd.DataFrame, default_trials: int = DEFAULT_TRIALS) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["model_name", "successes", "trials"])

    working_df = df.copy()

    model_col = _find_column(working_df, MODEL_COLUMN_CANDIDATES)
    successes_col = _find_column(working_df, SUCCESSES_COLUMN_CANDIDATES)
    trials_col = _find_column(working_df, TRIALS_COLUMN_CANDIDATES)
    success_rate_col = _find_column(working_df, SUCCESS_RATE_COLUMN_CANDIDATES)

    if model_col is None or (successes_col is None and success_rate_col is None):
        promoted_df = _promote_header_from_any_row_if_needed(working_df)
        if not promoted_df.equals(working_df):
            working_df = promoted_df
            model_col = _find_column(working_df, MODEL_COLUMN_CANDIDATES)
            successes_col = _find_column(working_df, SUCCESSES_COLUMN_CANDIDATES)
            trials_col = _find_column(working_df, TRIALS_COLUMN_CANDIDATES)
            success_rate_col = _find_column(working_df, SUCCESS_RATE_COLUMN_CANDIDATES)

    if model_col is None:
        raise ValueError(
            "Could not find a model column. Add one of: Model Name / Model / Policy / Policy Name. "
            "If your header is not on the first row, ensure the header row contains these names."
        )

    out = working_df.copy()
    out["model_name"] = out[model_col].astype(str).str.strip()
    out["model_name"] = out["model_name"].replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    out["model_name"] = out["model_name"].ffill()

    if trials_col is not None:
        trials = pd.to_numeric(out[trials_col], errors="coerce")
    else:
        trials = pd.Series(default_trials, index=out.index, dtype=float)
    trials = trials.fillna(default_trials).clip(lower=1)
    out["trials"] = trials.round().astype(int)

    if successes_col is not None:
        successes = pd.to_numeric(out[successes_col], errors="coerce")
    elif success_rate_col is not None:
        rate = pd.to_numeric(out[success_rate_col], errors="coerce")
        rate = rate.where(rate <= 1.0, rate / 100.0)
        successes = (rate * out["trials"]).round()
    else:
        successes = pd.Series(pd.NA, index=out.index)

    out["successes"] = pd.to_numeric(successes, errors="coerce")
    out["successes"] = out["successes"].fillna(0).clip(lower=0)
    out["successes"] = out[["successes", "trials"]].min(axis=1)
    out["successes"] = out["successes"].round().astype(int)

    out = out[out["model_name"].notna()].copy()
    out = out[out["model_name"].astype(str).str.strip().astype(bool)].copy()

    required_first = ["model_name", "successes", "trials"]
    remaining = [col for col in out.columns if col not in required_first]
    return out[required_first + remaining]
