# Robot Policy Evaluation Dashboard (Dash)

Minimal interactive dashboard for tracking robot policy rollout experiments from locally uploaded spreadsheets.

## Features

- Upload a local CSV/XLSX file (e.g., downloaded from Google Sheets).
- Choose which sheet/tab to load when an XLSX file contains multiple sheets.
- Edit/log rollout rows in-app.
- Compute per-policy Wilson confidence intervals for success rates.
- Compute per-policy quality-score confidence intervals (t-distribution) when a `Quality Score STD [%]` column is present.
- **Common prefix stripping**: automatically detects and removes the longest shared prefix (at `_` or `-` boundaries) from policy names in all charts, violins, and CLD table for cleaner visuals. The stripped prefix is shown in chart subtitles and the sort-status bar.
- Compare two policies (A/B) with:
  - Compact delta + CI summary (one line per metric)
  - Color-coded verdict per metric: **green** (better), **red** (worse), **gray** (inconclusive)
  - **Combined overall verdict** synthesizing both success-rate and quality signals (trade-offs, agreements, mixed signals)
  - Bar chart with Wilson CI error bars
  - Quality-score bar chart with t-distribution CI error bars (when STD is present)
  - Posterior violin plot (Bayesian uncertainty)
- Compare multiple policies with base-vs-policy pair letters.
- **Success Rate vs Quality Score scatter plot**: each policy is a point at (SR%, Quality%) with Wilson CI horizontal bars and quality CI vertical bars, enabling quick Pareto-style comparison.
- Visualize posterior uncertainty using Bayesian violins in a final optional panel that compares base-vs-policy pairs (`base vs candidate`) with pair letters (`a-a` / `a-b`).
- Plot selected policy success rates with Wilson CI error bars.
- Plot selected policies' quality score (%) with t-distribution CI error bars (when STD is present) and quality pair-letter annotations.
- Select all / deselect all policies in the plot.
- Keep original sheet order by default, with optional sorting by success rate or `Quality Score [%]`.
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

## Input workflow

1. In Google Sheets, click `File -> Download` and export as `CSV` or `XLSX`.
2. In the app, click `Upload CSV/XLSX` and pick the downloaded file.
3. If the file has multiple sheets (XLSX), use the `Sheet` dropdown to select the tab you want to analyze.
4. Review/edit rows in the table and continue with analysis.

No API keys or Google authentication are required.

Note: CSV files have a single table only, so there is no sheet selection for CSV.

The loader is flexible about header placement:

- It scans the top part of the file to detect the row that looks like the header (e.g., `Model Name`, `Successes`, `Trials`).
- If your file has title/metadata rows before the table, they are skipped automatically.
- Blank model cells caused by merged-cell style formatting are forward-filled so rows stay associated with the right model.

Under multi-policy comparison:

- Use `Original Order`, `Sort by Success Rate`, and `Sort by Quality Score [%]` to control policy order.
- When your source sheet has a success-rate column and some rows are empty, those policies are auto-deselected from initial plotting.
- Quality score values in `[0, 1]` are automatically converted to percentage points `[0, 100]` for display and sorting.

## Expected columns

The app expects at least a model name column plus either:

- `successes` + `trials`, or
- `success rate` (+ optional `trials`; defaults to 44)

Optional columns:

- `Quality Score [%]` — mean quality score per policy (values in `[0, 1]` are auto-scaled to `[0, 100]`).
- `Quality Score STD [%]` — standard deviation of per-rollout quality scores. When present, the app computes t-distribution CIs for quality and runs Welch t-tests for A/B and multi-policy comparisons.

Accepted model-name aliases include: `Model Name`, `Model`, `Policy`, `Policy Name`.

## Notes

- Confidence level controls Wilson intervals, quality-score t-distribution CIs, and hypothesis thresholds (`alpha = 1 - confidence`).
- Success-rate comparisons use two-proportion z-tests; quality-score comparisons use Welch's t-test. Statistical test details (p-values, t-statistics) are computed internally but hidden from the UI to reduce clutter — significance is communicated through color-coded verdicts and pair letters.
- Base-vs-policy pair letters are unadjusted (matching single-comparison A/B logic).
- Both A/B verdicts and the multi-policy comparison table use color-coding / row highlighting to surface significant differences at a glance.
- The A/B combined verdict provides a single-glance recommendation when both SR and quality data are available (e.g., "Both metrics favor B", "Trade-off: higher SR but lower quality").
