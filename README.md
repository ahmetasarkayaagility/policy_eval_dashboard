# Robot Policy Evaluation Dashboard (Dash)

Minimal interactive dashboard for tracking robot policy rollout experiments from locally uploaded spreadsheets.

## Features

- Upload a local CSV/XLSX file (e.g., downloaded from Google Sheets).
- Choose which sheet/tab to load when an XLSX file contains multiple sheets.
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

Accepted model-name aliases include: `Model Name`, `Model`, `Policy`, `Policy Name`.

## Notes

- Confidence level controls Wilson intervals, quality-score t-distribution CIs, and hypothesis thresholds (`alpha = 1 - confidence`).
- Success-rate comparisons use two-proportion z-tests; quality-score comparisons use Welch's t-test. Statistical test details (p-values, t-statistics) are computed internally but hidden from the UI to reduce clutter — significance is communicated through color-coded verdicts and pair letters.
- Base-vs-policy pair letters are unadjusted (matching single-comparison A/B logic).
- Both A/B verdicts and the multi-policy comparison table use color-coding / row highlighting to surface significant differences at a glance.
- The A/B combined verdict provides a single-glance recommendation when multiple metrics are available (e.g., "B is significantly better on success rate, quality", "Trade-off: B is better on success rate but worse on drop-in ratio").
