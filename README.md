# Robot Policy Evaluation Dashboard (Dash)

Minimal interactive dashboard for tracking robot policy rollout experiments from local files or Google Sheets links.

## Features

- Upload a local CSV/XLSX file (e.g., downloaded from Google Sheets).
- Paste a Google Sheets URL and load/refresh directly in-app (no manual XLSX export needed).
- Shows a small in-app Google auth status hint (`ready` / `needs sign-in`) near the Google URL field.
- Supports hidden Google hyperlink cells in `Eval Details` (including `HYPERLINK(...)` formulas) and extracts them into `eval_details_url` for downstream rollout-detail analysis.
- Choose which sheet/tab to load when an XLSX file contains multiple sheets.
- Reuse a default Google Sheets URL via `DEFAULT_GOOGLE_SHEET_URL`.
- Edit/log rollout rows in-app.
- Compute per-policy Wilson confidence intervals for success rates.
- Compute per-policy quality-score confidence intervals (t-distribution) when a `Quality Score STD [%]` column is present.
- Compute per-policy attempt-drop-in Wilson confidence intervals when an `Attempt to drop in Ratio [%]` column is present (binary proportion, lower is better).
- **Common prefix stripping**: automatically detects and removes the longest shared prefix (at `_` or `-` boundaries) from policy names in all charts, violins, and CLD table for cleaner visuals. The stripped prefix is shown in chart subtitles and the sort-status bar.
- Compare two policies (A/B) with:
  - Compact delta + CI summary (one line per metric)
  - Color-coded verdict per metric: **green** (better), **red** (worse), **gray** (inconclusive)
  - **Combined overall verdict** synthesizing success-rate, quality, and drop-in signals (trade-offs, agreements, mixed signals)
  - Bar chart with Wilson CI error bars
  - Quality-score bar chart with t-distribution CI error bars (when STD is present)
  - Attempt drop-in ratio bar chart with Wilson CI error bars (when column is present; lower is better)
  - Posterior violin plot (Bayesian uncertainty)
- Compare multiple policies with base-vs-policy pair letters.
- **Testing Group tag filtering**: when a `Testing Group` column is present, use `Plot Tag + Base` to instantly select policies in the chosen tag plus the row tagged as `Base`/`Default`/`Baseline`/`Control`.
- **Success Rate vs Quality Score scatter plot**: each policy is a point at (SR%, Quality%) with Wilson CI horizontal bars and quality CI vertical bars, enabling quick Pareto-style comparison.
- Visualize posterior uncertainty using Bayesian violins in a final optional panel that compares base-vs-policy pairs (`base vs candidate`) with pair letters (`a-a` / `a-b`).
- Plot selected policy success rates with Wilson CI error bars.
- Plot selected policies' quality score (%) with t-distribution CI error bars (when STD is present) and quality pair-letter annotations.
- Plot selected policies' attempt drop-in ratio (%) with Wilson CI error bars (when column is present).
- Select all / deselect all policies in the plot.
- Cleaner multi-policy controls: selection, tag filtering, and sorting controls are grouped in wrapped rows; policy checkboxes are shown in a compact scrollable panel.
- Keep original sheet order by default, with optional sorting by success rate, `Quality Score [%]`, or `Attempt Drop-in Ratio [%]`.
- **Dedicated Failure Mode Analysis page/tab** with rollout-level diagnostics sourced from per-policy detail sheet links.
- **Single aggregate failure heatmap** for fast condition scanning: one grayscale heatmap averaged across completed policies with metric switch (`failure`, `success`, `quality`, `n`) and per-cell value labels.
- **Main-page failure highlights**: compact hardest/easiest condition highlights from aggregate failure analysis.
- **Axis-aggregated condition heatmaps**: separate grayscale heatmaps for stack conditions (aggregated over robot conditions) and robot conditions (aggregated over stack conditions), with per-cell value labels.
- **Top hardest + easiest conditions** tables for quick best/worst condition lookup.
- Auto-deselect policies that have empty values in the source success-rate column.
- Use consistent per-policy color mapping across all plots.
- Exclude policies with empty source success-rate values from analysis.
- Decluttered CLD comparison table: p-value columns removed (significance is conveyed by pair letters and row highlighting).
- Conditional row highlighting in the comparison table: green for both-significant, amber for one-significant.

Final violin panel behavior:

- The panel can be hidden with `Show final base-vs-policy violin panel` to reduce UI clutter.
- It uses the selected base policy from A/B section (`Policy A`) as the reference.
- For each selected policy, it shows a posterior violin pair (`base` and `other`) on success-rate (%) axis.
- The companion table in this section also reports base-vs-policy rows (`delta`, `p-value`, letters) instead of all-vs-all CLD groups.

## Setup

```bash
uv venv .venv
uv pip install -r requirements.txt
uv run --python .venv/bin/python python app.py
```

Open: `http://127.0.0.1:8050`

### Google Sheets authentication

Quickest Linux onboarding (recommended):

```bash
cp .env.example .env
./scripts/setup_google_auth_linux.sh
```

This script:

- updates `.env` with a default Google Sheet URL,
- uses `gcloud` ADC login if available,
- falls back to OAuth client flow if configured,
- verifies sheet access immediately.

The app uses this order for auth, to keep sign-in as frictionless as possible:

1. **Application Default Credentials (ADC)** (best for company-managed setups).
2. **Cached OAuth token** from a previous sign-in.
3. **Interactive OAuth browser popup** (first-time fallback).

#### Option A — Company users via ADC (least friction)

Install Google Cloud SDK (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get update && sudo apt-get install -y google-cloud-cli
```

Then authenticate once:

```bash
gcloud auth application-default login --scopes=https://www.googleapis.com/auth/spreadsheets.readonly
```

#### Option B — OAuth Desktop client (fallback when ADC is unavailable)

For team onboarding, one admin can create the OAuth client once and share the downloaded JSON securely.

Create/download OAuth client JSON:

1. Open `https://console.cloud.google.com/apis/library/sheets.googleapis.com` and enable **Google Sheets API**.
2. Open `https://console.cloud.google.com/apis/credentials`.
3. Configure OAuth consent screen as **Internal** (company workspace) if applicable.
4. Click **Create Credentials → OAuth client ID → Desktop app**.
5. Download the JSON file.

Use it in this project:

```bash
cp /path/from/downloads/client_secret_*.json ./google_oauth_client_secret.json
./scripts/setup_google_auth_linux.sh --client-secret ./google_oauth_client_secret.json
```

Or set env var directly:

```bash
export GOOGLE_OAUTH_CLIENT_SECRET_FILE=/path/to/client_secret.json
```

On first Google Sheet load (or bootstrap verification), a browser sign-in/consent tab opens automatically.

Optional env vars:

- `DEFAULT_GOOGLE_SHEET_URL` — pre-filled URL in the app input field.
  - Current fallback default is preconfigured to your primary sheet URL and can be overridden with this env var.
- `GOOGLE_OAUTH_CLIENT_SECRET_FILE` — OAuth desktop client JSON path.
- `GOOGLE_OAUTH_TOKEN_CACHE` — token cache path (default: `~/.config/policy_eval_dashboard/google_oauth_token.json`).

## Input workflow

1. Choose one of two paths:
  - **Local file**: export `CSV`/`XLSX` and click `Upload CSV/XLSX`.
  - **Google link**: paste the spreadsheet URL and click `Load/Refresh Google Sheet`.
2. If the source has multiple tabs, use the `Sheet` dropdown to select the tab you want to analyze.
3. Review/edit rows in the table and continue with analysis.

Failure analysis workflow:

1. Open the `Failure Mode Analysis` tab.
2. Click `Load/Refresh detailed rollout sheets`.
3. The app reads per-policy detail URLs from `eval_details_url` / `Eval Details` and loads rollout sheets.
  - Rows without a specified success-rate value are treated as planning rows and skipped.
4. Choose a metric and inspect:
  - the full aggregate grayscale condition heatmap,
  - stack-condition aggregate heatmap,
  - robot-condition aggregate heatmap,
  - plus hardest/easiest condition tables.

For Google links, authentication is required (handled automatically as above).

Note: CSV files have a single table only, so there is no sheet selection for CSV.

The loader is flexible about header placement:

- It scans the top part of the file to detect the row that looks like the header (e.g., `Model Name`, `Successes`, `Trials`).
- If your file has title/metadata rows before the table, they are skipped automatically.
- Blank model cells caused by merged-cell style formatting are forward-filled so rows stay associated with the right model.

Under multi-policy comparison:

- Use `Original Order`, `Sort by Success Rate`, `Sort by Quality Score [%]`, and `Sort by Attempt Drop-in` to control policy order.
- If your sheet contains `Testing Group`, choose a tag and click `Plot Tag + Base` to plot that tag plus the base-tagged row.
- Use `Clear Tag Filter` to go back to normal manual policy selection.
- When your source sheet has a success-rate column and some rows are empty, those policies are auto-deselected from initial plotting.
- Quality score values in `[0, 1]` are automatically converted to percentage points `[0, 100]` for display and sorting.

## Expected columns

The app expects at least a model name column plus either:

- `successes` + `trials`, or
- `success rate` (+ optional `trials`; defaults to 44)

Optional columns:

- `Quality Score [%]` — mean quality score per policy (values in `[0, 1]` are auto-scaled to `[0, 100]`).
- `Quality Score STD [%]` — standard deviation of per-rollout quality scores. When present, the app computes t-distribution CIs for quality and runs Welch t-tests for A/B and multi-policy comparisons.
- `Attempt to drop in Ratio [%]` — proportion of attempts where a drop-in occurred (binary Yes/No outcome). Lower is better. Values in `[0, 1]` are auto-scaled to `[0, 100]`. The count of drop-in events is back-computed as `round(ratio × trials)` and used for Wilson CIs and Newcombe-Wilson delta CIs.
- `Testing Group` — tag/category for each policy row (e.g., `Training Hyperparams`, `Model Arch.`, `Data Mixture & Augmentation`). Values such as `Base`, `Default`, `Baseline`, or `Control` are treated as base rows for tag-group plotting.
- `Eval Details` (or `Eval Details URL`) — link to a per-policy rollout-detail sheet. Hidden Google hyperlink labels like `Link` are supported and extracted to `eval_details_url` automatically when possible.

Expected columns in rollout-detail sheets (Failure Mode Analysis tab):

- `Task Success` (required) — per-rollout binary outcome (`1/0`, `true/false`, `yes/no`, etc. are accepted).
- `Score based on rubric` (optional) — per-rollout quality score; values in `[0, 1]` are auto-scaled to `%`.
- Condition columns (recommended): `Relative Stance Offset`, `Tote on Pallet Offset`.
  - The failure page auto-detects condition columns and auto-selects the primary X/Y pair for aggregate heatmap rendering.

Accepted model-name aliases include: `Model Name`, `Model`, `Policy`, `Policy Name`.

## Notes

- Confidence level controls Wilson intervals, quality-score t-distribution CIs, and hypothesis thresholds (`alpha = 1 - confidence`).
- Success-rate comparisons use two-proportion z-tests; quality-score comparisons use Welch's t-test. Statistical test details (p-values, t-statistics) are computed internally but hidden from the UI to reduce clutter — significance is communicated through color-coded verdicts and pair letters.
- Base-vs-policy pair letters are unadjusted (matching single-comparison A/B logic).
- Both A/B verdicts and the multi-policy comparison table use color-coding / row highlighting to surface significant differences at a glance.
- The A/B combined verdict provides a single-glance recommendation when multiple metrics are available (e.g., "B is significantly better on success rate, quality", "Trade-off: B is better on success rate but worse on drop-in ratio").
