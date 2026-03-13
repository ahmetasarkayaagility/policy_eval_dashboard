#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SCOPE="https://www.googleapis.com/auth/spreadsheets.readonly"
DEFAULT_URL="https://docs.google.com/spreadsheets/d/1yGNy2hHN5kyMltyjmYUhrmjvzzEMYweGe9MeYex_Lls/edit?gid=245836030#gid=245836030"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/setup_google_auth_linux.sh [--sheet-url URL] [--client-secret /path/to/client_secret.json]

What it does:
  1) Creates/updates .env with DEFAULT_GOOGLE_SHEET_URL
  2) Optionally stores GOOGLE_OAUTH_CLIENT_SECRET_FILE in .env
  3) Uses gcloud ADC login if available (recommended)
  4) Verifies sheet access via Python loader
EOF
}

SHEET_URL=""
CLIENT_SECRET_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sheet-url)
      SHEET_URL="${2:-}"
      shift 2
      ;;
    --client-secret)
      CLIENT_SECRET_PATH="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Python was not found. Install Python 3.11+ and retry."
    exit 1
  fi
fi

ENV_FILE="$ROOT_DIR/.env"
if [[ ! -f "$ENV_FILE" && -f "$ROOT_DIR/.env.example" ]]; then
  cp "$ROOT_DIR/.env.example" "$ENV_FILE"
fi

upsert_env() {
  local key="$1"
  local value="$2"
  "$PYTHON_BIN" - "$ENV_FILE" "$key" "$value" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]

lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
out = []
replaced = False
for line in lines:
    if line.startswith(f"{key}="):
        out.append(f"{key}={value}")
        replaced = True
    else:
        out.append(line)

if not replaced:
    if out and out[-1].strip():
        out.append("")
    out.append(f"{key}={value}")

path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
PY
}

if [[ -z "$SHEET_URL" ]]; then
  SHEET_URL="$DEFAULT_URL"
fi
upsert_env "DEFAULT_GOOGLE_SHEET_URL" "$SHEET_URL"

if [[ -z "$CLIENT_SECRET_PATH" ]]; then
  if [[ -n "${GOOGLE_OAUTH_CLIENT_SECRET_FILE:-}" ]]; then
    CLIENT_SECRET_PATH="$GOOGLE_OAUTH_CLIENT_SECRET_FILE"
  elif [[ -f "$ROOT_DIR/google_oauth_client_secret.json" ]]; then
    CLIENT_SECRET_PATH="$ROOT_DIR/google_oauth_client_secret.json"
  fi
fi

if [[ -n "$CLIENT_SECRET_PATH" ]]; then
  if [[ ! -f "$CLIENT_SECRET_PATH" ]]; then
    echo "Provided client-secret file not found: $CLIENT_SECRET_PATH"
    exit 1
  fi
  upsert_env "GOOGLE_OAUTH_CLIENT_SECRET_FILE" "$CLIENT_SECRET_PATH"
fi

echo "[1/3] Ensuring Python deps are installed..."
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PYTHON_BIN" -r requirements.txt >/dev/null
else
  echo "uv not found; skipping dependency sync (install from requirements.txt manually if needed)."
fi

echo "[2/3] Bootstrapping credentials..."
if command -v gcloud >/dev/null 2>&1; then
  echo "Using gcloud ADC login (recommended for company users)."
  gcloud auth application-default login --scopes="$SCOPE"
else
  echo "gcloud was not found on PATH."
  if [[ -z "$CLIENT_SECRET_PATH" ]]; then
    cat <<'EOF'
No OAuth client configured yet.

Fast options:
  A) Install gcloud and rerun this script (recommended)
  B) Create/download OAuth desktop client JSON and rerun with:
     ./scripts/setup_google_auth_linux.sh --client-secret /path/to/client_secret.json

OAuth console links:
  https://console.cloud.google.com/apis/credentials
  https://console.cloud.google.com/apis/library/sheets.googleapis.com
EOF
  fi
fi

echo "[3/3] Verifying Google Sheets access..."
"$PYTHON_BIN" - <<'PY'
from data_utils import DEFAULT_GOOGLE_SHEET_URL, list_google_spreadsheet_sheets

if not DEFAULT_GOOGLE_SHEET_URL:
    raise SystemExit("DEFAULT_GOOGLE_SHEET_URL is empty. Set it in .env and rerun.")

try:
    meta = list_google_spreadsheet_sheets(DEFAULT_GOOGLE_SHEET_URL)
except Exception as exc:
    raise SystemExit(
        "Google auth/bootstrap is not complete yet. "
        f"Current error: {exc}"
    )

sheet_names = [str(s.get("name", "")) for s in meta.get("sheets", [])]
print(
    "OK: "
    f"title='{meta.get('title')}', "
    f"default_tab='{meta.get('default_sheet')}', "
    f"tabs={len(sheet_names)}"
)
PY

echo "Completed. You can now run: uv run --python .venv/bin/python python app.py"
