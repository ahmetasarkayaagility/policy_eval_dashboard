from __future__ import annotations

import base64
import functools
import io
import os
import re
import shutil
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

import pandas as pd

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency fallback
    def load_dotenv(*_args, **_kwargs):
        return False


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


DEFAULT_TRIALS = 44
CSV_SINGLE_SHEET_NAME = "CSV"
FALLBACK_DEFAULT_GOOGLE_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/1yGNy2hHN5kyMltyjmYUhrmjvzzEMYweGe9MeYex_Lls/"
    "edit?gid=245836030#gid=245836030"
)
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

GOOGLE_SHEETS_READONLY_SCOPE = "https://www.googleapis.com/auth/spreadsheets.readonly"
DEFAULT_GOOGLE_SHEET_URL = os.getenv("DEFAULT_GOOGLE_SHEET_URL", FALLBACK_DEFAULT_GOOGLE_SHEET_URL).strip()
DEFAULT_GOOGLE_OAUTH_CLIENT_SECRET_FILE = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET_FILE", "").strip()
DEFAULT_GOOGLE_OAUTH_TOKEN_CACHE = os.getenv("GOOGLE_OAUTH_TOKEN_CACHE", "").strip()
EVAL_DETAILS_COLUMN_CANDIDATES = [
    "Eval Details",
    "Evaluation Details",
    "Eval Detail",
    "Rollout Details",
    "Details",
    "Detail Link",
    "Details Link",
]
EVAL_DETAILS_URL_COLUMN_CANDIDATES = [
    "eval_details_url",
    "Eval Details URL",
    "Evaluation Details URL",
    "Detail URL",
    "Details URL",
    "Rollout Details URL",
    "EvalDetailsURL",
    "DetailURL",
]


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


def _percent_like_to_numeric(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
    text = text.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    return pd.to_numeric(text, errors="coerce")


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


def _import_google_dependencies():
    try:
        import google.auth
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except ImportError as exc:
        raise ValueError(
            "Google Sheets dependencies are missing. Install requirements.txt and retry."
        ) from exc

    return google.auth, Request, Credentials, InstalledAppFlow, build, HttpError


def _parse_google_spreadsheet_url(spreadsheet_url: str) -> tuple[str, str | None]:
    clean_url = str(spreadsheet_url or "").strip()
    if not clean_url:
        raise ValueError("Google Sheets URL is empty.")

    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", clean_url)
    if not match:
        raise ValueError(
            "Invalid Google Sheets URL. Expected format: "
            "https://docs.google.com/spreadsheets/d/<SPREADSHEET_ID>/edit"
        )

    spreadsheet_id = match.group(1)

    parsed = urlparse(clean_url)
    query_gid = parse_qs(parsed.query).get("gid", [None])[0]

    fragment_gid = None
    if parsed.fragment:
        fragment_map = parse_qs(parsed.fragment)
        fragment_gid = fragment_map.get("gid", [None])[0]
        if fragment_gid is None and parsed.fragment.startswith("gid="):
            fragment_gid = parsed.fragment.split("=", 1)[1]

    gid = str(query_gid or fragment_gid) if (query_gid or fragment_gid) is not None else None
    return spreadsheet_id, gid


def _extract_url_from_value(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None

    formula_match = re.search(
        r'(?is)=\s*HYPERLINK\(\s*(?:"([^"]+)"|\'([^\']+)\')',
        text,
    )
    if formula_match:
        url = formula_match.group(1) or formula_match.group(2)
        if url:
            return url.strip()

    url_match = re.search(r"https?://[^\s\)\]\}\"']+", text)
    if url_match:
        return url_match.group(0).rstrip(".,;)")

    return None


def extract_url_from_cell_value(value: object) -> str | None:
    return _extract_url_from_value(value)


def _google_token_cache_path() -> Path:
    if DEFAULT_GOOGLE_OAUTH_TOKEN_CACHE:
        return Path(DEFAULT_GOOGLE_OAUTH_TOKEN_CACHE).expanduser()
    return Path.home() / ".config" / "policy_eval_dashboard" / "google_oauth_token.json"


def _gcloud_adc_credentials_path() -> Path:
    return Path.home() / ".config" / "gcloud" / "application_default_credentials.json"


def _google_client_secret_path() -> Path:
    if DEFAULT_GOOGLE_OAUTH_CLIENT_SECRET_FILE:
        return Path(DEFAULT_GOOGLE_OAUTH_CLIENT_SECRET_FILE).expanduser()

    repo_candidate = Path(__file__).resolve().parent / "google_oauth_client_secret.json"
    cwd_candidate = Path.cwd() / "google_oauth_client_secret.json"

    if cwd_candidate.exists():
        return cwd_candidate
    if repo_candidate.exists():
        return repo_candidate

    return cwd_candidate


def _google_error_message(exc: Exception) -> str:
    status_code = getattr(getattr(exc, "resp", None), "status", None)
    if status_code in {401, 403}:
        return (
            "Google Sheets access denied. Sign in with your company Google account and ensure the sheet is "
            "shared with your company domain (or your account)."
        )
    if status_code == 404:
        return "Google Sheet not found. Check that the URL is correct and the sheet still exists."
    if status_code is not None:
        return f"Google Sheets API error ({status_code}): {exc}"
    return f"Google Sheets error: {exc}"


def _adc_credentials_with_scope(scopes: list[str]):
    google_auth, Request, Credentials, _InstalledAppFlow, _build, _HttpError = _import_google_dependencies()

    explicit_credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if explicit_credentials_file:
        explicit_path = Path(explicit_credentials_file).expanduser()
        if explicit_path.exists():
            try:
                creds, _project = google_auth.load_credentials_from_file(str(explicit_path), scopes=scopes)
                if creds is not None and not getattr(creds, "valid", False):
                    creds.refresh(Request())
                if creds is not None and getattr(creds, "valid", False):
                    return creds
            except Exception:
                pass

    adc_path = _gcloud_adc_credentials_path()
    if adc_path.exists():
        try:
            adc_credentials = Credentials.from_authorized_user_file(str(adc_path), scopes=scopes)
            if adc_credentials is not None and not adc_credentials.valid and adc_credentials.refresh_token:
                adc_credentials.refresh(Request())
            if adc_credentials is not None and adc_credentials.valid:
                return adc_credentials
        except Exception:
            pass

    # Last resort: only attempt default chain when explicitly allowed, because
    # it can be slow on local machines due to metadata-probe backoff.
    if str(os.getenv("POLICY_EVAL_ALLOW_GOOGLE_DEFAULT", "")).strip().lower() not in {"1", "true", "yes"}:
        return None

    try:
        adc_credentials, _project = google_auth.default(scopes=scopes)
        if adc_credentials is None:
            return None
        if getattr(adc_credentials, "requires_scopes", False):
            adc_credentials = adc_credentials.with_scopes(scopes)
        if not getattr(adc_credentials, "valid", False):
            adc_credentials.refresh(Request())
        if getattr(adc_credentials, "valid", False):
            return adc_credentials
    except Exception:
        return None
    return None


def _cached_oauth_credentials(scopes: list[str], persist_refresh: bool = True):
    _google_auth, Request, Credentials, _InstalledAppFlow, _build, _HttpError = _import_google_dependencies()
    token_path = _google_token_cache_path()
    if not token_path.exists():
        return None

    try:
        user_credentials = Credentials.from_authorized_user_file(str(token_path), scopes=scopes)
        if user_credentials is not None and not user_credentials.valid and user_credentials.refresh_token:
            user_credentials.refresh(Request())
            if persist_refresh:
                token_path.parent.mkdir(parents=True, exist_ok=True)
                token_path.write_text(user_credentials.to_json(), encoding="utf-8")
        if user_credentials is not None and user_credentials.valid:
            return user_credentials
    except Exception:
        return None

    return None


def _get_google_credentials(scopes: list[str]):
    _google_auth, _Request, _Credentials, InstalledAppFlow, _build, _HttpError = _import_google_dependencies()

    adc_credentials = _adc_credentials_with_scope(scopes)
    if adc_credentials is not None:
        return adc_credentials

    token_credentials = _cached_oauth_credentials(scopes, persist_refresh=True)
    if token_credentials is not None:
        return token_credentials

    client_secret_file = _google_client_secret_path()
    if not client_secret_file.exists():
        if shutil.which("gcloud"):
            adc_hint = (
                "Recommended: run `gcloud auth application-default login "
                "--scopes=https://www.googleapis.com/auth/spreadsheets.readonly` "
                "and retry."
            )
        else:
            adc_hint = (
                "Recommended: install Google Cloud SDK (`gcloud`) and run "
                "`gcloud auth application-default login "
                "--scopes=https://www.googleapis.com/auth/spreadsheets.readonly` "
                "for one-time company sign-in."
            )

        raise ValueError(
            "Google authentication is required, but no credentials were found. "
            f"{adc_hint} "
            "Alternatively, configure OAuth Desktop client JSON with "
            "GOOGLE_OAUTH_CLIENT_SECRET_FILE (or place `google_oauth_client_secret.json` in project root). "
            "Create a Desktop OAuth client in Google Cloud Console and set "
            "GOOGLE_OAUTH_CLIENT_SECRET_FILE to that JSON path."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_file), scopes=scopes)
    try:
        user_credentials = flow.run_local_server(
            host="localhost",
            port=0,
            open_browser=True,
            prompt="consent",
            authorization_prompt_message="Opening a browser window for Google sign-in...",
            success_message="Google authentication complete. You can close this tab.",
        )
    except Exception:
        user_credentials = flow.run_console()
    if user_credentials is None or not user_credentials.valid:
        raise ValueError("Google OAuth authentication did not complete.")

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(user_credentials.to_json(), encoding="utf-8")
    return user_credentials


def get_google_auth_status() -> dict[str, str | bool]:
    try:
        _import_google_dependencies()
    except Exception as exc:
        return {
            "ready": False,
            "state": "error",
            "message": str(exc),
        }

    scopes = [GOOGLE_SHEETS_READONLY_SCOPE]

    if _adc_credentials_with_scope(scopes) is not None:
        return {
            "ready": True,
            "state": "ready",
            "message": "ready via Application Default Credentials",
        }

    if _cached_oauth_credentials(scopes, persist_refresh=False) is not None:
        return {
            "ready": True,
            "state": "ready",
            "message": "ready via cached OAuth token",
        }

    client_secret_file = _google_client_secret_path()
    if client_secret_file.exists():
        return {
            "ready": False,
            "state": "signin",
            "message": "OAuth client found; click Load/Refresh Google Sheet to sign in once",
        }

    if shutil.which("gcloud"):
        return {
            "ready": False,
            "state": "setup",
            "message": "run gcloud auth application-default login once",
        }

    return {
        "ready": False,
        "state": "setup",
        "message": "install gcloud or set GOOGLE_OAUTH_CLIENT_SECRET_FILE",
    }


@functools.lru_cache(maxsize=1)
def _google_sheets_service():
    _google_auth, _request, _credentials, _flow, build, _http_error = _import_google_dependencies()
    credentials = _get_google_credentials([GOOGLE_SHEETS_READONLY_SCOPE])
    return build("sheets", "v4", credentials=credentials, cache_discovery=False)


def _sheet_values_to_dataframe(rows: list[list[object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    width = max((len(row) for row in rows), default=0)
    if width <= 0:
        return pd.DataFrame()

    normalized_rows = [list(row) + [pd.NA] * (width - len(row)) for row in rows]
    columns = [f"col_{idx + 1}" for idx in range(width)]
    out = pd.DataFrame(normalized_rows, columns=columns)
    out = out.replace({"": pd.NA}).dropna(how="all").reset_index(drop=True)
    return out


def _overlay_hyperlink_formulas(
    base_rows: list[list[object]],
    formula_rows: list[list[object]],
) -> list[list[object]]:
    out_rows: list[list[object]] = []
    max_rows = max(len(base_rows), len(formula_rows))
    for row_index in range(max_rows):
        base_row = list(base_rows[row_index]) if row_index < len(base_rows) else []
        formula_row = formula_rows[row_index] if row_index < len(formula_rows) else []
        max_cols = max(len(base_row), len(formula_row))
        if len(base_row) < max_cols:
            base_row.extend([pd.NA] * (max_cols - len(base_row)))

        for col_index in range(max_cols):
            formula_value = formula_row[col_index] if col_index < len(formula_row) else None
            if isinstance(formula_value, str) and re.search(r"(?is)^\s*=\s*HYPERLINK\(", formula_value):
                base_row[col_index] = formula_value

        out_rows.append(base_row)

    return out_rows


def _sheet_hyperlink_map(service, spreadsheet_id: str, range_name: str) -> dict[tuple[int, int], str]:
    response = service.spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        ranges=[range_name],
        includeGridData=True,
        fields=(
            "sheets(data(rowData(values(hyperlink,formattedValue,textFormatRuns(startIndex,format(link))))))"
        ),
    ).execute()

    out: dict[tuple[int, int], str] = {}
    sheets = response.get("sheets") or []
    for sheet in sheets:
        data_entries = sheet.get("data") or []
        for data_entry in data_entries:
            row_data = data_entry.get("rowData") or []
            for row_index, row in enumerate(row_data):
                values = row.get("values") or []
                for col_index, cell in enumerate(values):
                    url = str(cell.get("hyperlink") or "").strip()
                    if not url:
                        for run in cell.get("textFormatRuns") or []:
                            link = ((run.get("format") or {}).get("link") or {}).get("uri")
                            if link:
                                url = str(link).strip()
                                break
                    if url:
                        out[(row_index, col_index)] = url
    return out


def _overlay_hyperlink_map(
    rows: list[list[object]],
    hyperlink_map: dict[tuple[int, int], str],
) -> list[list[object]]:
    if not hyperlink_map:
        return rows

    out_rows: list[list[object]] = [list(row) for row in rows]
    max_col_by_row: dict[int, int] = {}
    for (row_index, col_index), _url in hyperlink_map.items():
        max_col_by_row[row_index] = max(max_col_by_row.get(row_index, -1), col_index)

    for row_index, max_col in max_col_by_row.items():
        while len(out_rows) <= row_index:
            out_rows.append([])
        if len(out_rows[row_index]) <= max_col:
            out_rows[row_index].extend([pd.NA] * (max_col + 1 - len(out_rows[row_index])))

    for (row_index, col_index), url in hyperlink_map.items():
        out_rows[row_index][col_index] = url

    return out_rows


def _rows_contain_any_url(rows: list[list[object]]) -> bool:
    for row in rows:
        for value in row:
            if _extract_url_from_value(value):
                return True
    return False


@functools.lru_cache(maxsize=64)
def _google_spreadsheet_metadata(spreadsheet_id: str) -> dict[str, object]:
    _google_auth, _request, _credentials, _flow, _build, HttpError = _import_google_dependencies()
    service = _google_sheets_service()

    try:
        return service.spreadsheets().get(
            spreadsheetId=spreadsheet_id,
            fields="properties(title),sheets(properties(title,sheetId))",
        ).execute()
    except HttpError as exc:
        raise ValueError(_google_error_message(exc)) from exc


def list_google_spreadsheet_sheets(spreadsheet_url: str) -> dict[str, object]:
    spreadsheet_id, preferred_gid = _parse_google_spreadsheet_url(spreadsheet_url)
    metadata = _google_spreadsheet_metadata(spreadsheet_id)

    sheets: list[dict[str, str]] = []
    default_sheet = None
    for sheet in metadata.get("sheets", []):
        props = sheet.get("properties", {})
        title = str(props.get("title", "")).strip()
        if not title:
            continue
        gid = str(props.get("sheetId")) if props.get("sheetId") is not None else ""
        sheets.append({"name": title, "gid": gid})
        if preferred_gid is not None and gid == preferred_gid:
            default_sheet = title

    if not sheets:
        raise ValueError("No sheets/tabs found in this Google Spreadsheet.")

    if default_sheet is None:
        default_sheet = sheets[0]["name"]

    title = str(metadata.get("properties", {}).get("title", "Google Sheet"))
    return {
        "spreadsheet_id": spreadsheet_id,
        "title": title,
        "sheets": sheets,
        "default_sheet": default_sheet,
    }


def load_google_spreadsheet(
    spreadsheet_url: str,
    sheet_name: str | None = None,
    enrich_hyperlinks: bool = True,
) -> pd.DataFrame:
    spreadsheet_id, _preferred_gid = _parse_google_spreadsheet_url(spreadsheet_url)
    _google_auth, _request, _credentials, _flow, _build, HttpError = _import_google_dependencies()
    service = _google_sheets_service()

    selected_sheet_name = str(sheet_name).strip() if sheet_name else ""
    if not selected_sheet_name:
        metadata = list_google_spreadsheet_sheets(spreadsheet_url)
        selected_sheet_name = str(metadata.get("default_sheet") or "")

    if not selected_sheet_name:
        raise ValueError("Could not determine which Google Sheet tab to load.")

    escaped_sheet_name = selected_sheet_name.replace("'", "''")
    range_name = f"'{escaped_sheet_name}'"

    try:
        values_result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueRenderOption="UNFORMATTED_VALUE",
            dateTimeRenderOption="FORMATTED_STRING",
        ).execute()
    except HttpError as exc:
        raise ValueError(_google_error_message(exc)) from exc

    rows = values_result.get("values", [])

    if enrich_hyperlinks:
        try:
            formula_result = service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueRenderOption="FORMULA",
                dateTimeRenderOption="FORMATTED_STRING",
            ).execute()
            formula_rows = formula_result.get("values", [])
            if formula_rows:
                rows = _overlay_hyperlink_formulas(rows, formula_rows)
        except Exception:
            pass

        if not _rows_contain_any_url(rows):
            try:
                hyperlink_map = _sheet_hyperlink_map(service, spreadsheet_id, range_name)
                rows = _overlay_hyperlink_map(rows, hyperlink_map)
            except Exception:
                pass

    return _sheet_values_to_dataframe(rows)


def promote_header_row_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return _promote_header_from_any_row_if_needed(df.copy())


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
        rate = _percent_like_to_numeric(out[success_rate_col])
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

    eval_url_col = _find_column(out, EVAL_DETAILS_URL_COLUMN_CANDIDATES)
    eval_details_col = _find_column(out, EVAL_DETAILS_COLUMN_CANDIDATES)
    eval_details_urls = pd.Series(pd.NA, index=out.index, dtype="object")
    if eval_url_col is not None:
        eval_details_urls = out[eval_url_col].map(_extract_url_from_value)
    if eval_details_col is not None:
        parsed_from_details = out[eval_details_col].map(_extract_url_from_value)
        eval_details_urls = eval_details_urls.where(eval_details_urls.notna(), parsed_from_details)
    out["eval_details_url"] = eval_details_urls

    required_first = ["model_name", "successes", "trials"]
    remaining = [col for col in out.columns if col not in required_first]
    return out[required_first + remaining]
