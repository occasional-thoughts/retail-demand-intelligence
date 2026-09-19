# Phase 5 — Two dashboards, two audiences

## What is generated, and what you still do by hand

`make dashboards` writes both workbooks from the pipeline's extracts:

```
tableau/operations_command_centre.twb   5 sheets + 1 dashboard
tableau/executive_review.twb            6 sheets + 1 dashboard
```

They are generated rather than clicked because a `.twb` is XML, and generating
it means the dashboards are rebuilt from the pipeline output by running a
script — they cannot silently drift from the data, and the design is reviewable
in a diff like everything else in the repo. `scripts/build_tableau_workbooks.py`
is the definition; `scripts/tableau/_twb.py` is the writer.

**What the generator produces:** data connections, field typing, every sheet's
rows/columns/marks/encodings, filters, and the dashboard layout.

**What you should still do in the GUI:** visual polish that is genuinely
faster and better by hand — sort orders, custom colour palettes, reference
lines, number formatting, tooltips. Save from Tableau when you are done; your
saved version replaces the generated one. Regenerating overwrites it, so
regenerate first and polish second.

---

## The requirement, restated

The brief's hard constraint: these must not be the same charts twice. They are
built from **different extracts, at different grains, over different time
horizons, answering different questions**. If a chart would make sense on both,
it belongs on neither.

This is enforced, not just intended: `build_tableau_workbooks.py` asserts that
the operational workbook contains no `exec_` extract and the executive workbook
no `ops_` extract, and `tests/test_tableau.py` re-checks it.

| | Operations: *Command Centre* | Executive: *Retail Performance & Customer Intelligence* |
|---|---|---|
| Reader | Inventory / warehouse manager | Commercial leadership, strategy |
| Opened | Every morning | Monthly review |
| Time direction | **Forward** 8 weeks | **Backward** 23 months |
| Grain | Category, supplier lane | Company, segment, region |
| Question | "What do I order today?" | "Where is value moving, and what threatens it?" |
| Success | A decision made in 90 seconds | A conversation framed correctly |
| Canvas | 1600 × 900 landscape (wall monitor) | 1200 × 1600 portrait (exports to a PDF pack) |
| Extracts | `ops_*.csv` | `exec_*.csv` |

---

## Install and open

**Tableau Public Desktop** — free, macOS-native. Download from
public.tableau.com/app/discover, then:

```bash
open tableau/operations_command_centre.twb
```

The workbooks carry **absolute paths** to `data/08_reporting/*.csv`, generated
at build time. If you move the project, re-run `make dashboards`.

> Tableau Public saves workbooks to the public web. Keep the two separate and
> title them clearly. Nothing in these extracts is personal data — they are
> aggregates of a public dataset.

---

## Workbook 1 — Inventory Operations Command Centre

| Sheet | Shows | Why it is on an operational dashboard |
|---|---|---|
| **Action Queue** | Recommended order units per flagged category, coloured by status | The to-do list. Ranked so the top bar is the first thing to do this morning. Everything else on the dashboard exists to explain this one sheet. |
| **Days To Stockout** | Days of cover remaining per flagged category | The urgency behind the queue. A category with 17 days of cover and a 20-day lead time is already late. |
| **Cover vs Lead Time** | Scatter: weeks of cover against mean lead time, sized by forecast demand | Anything below the diagonal cannot be replenished before it runs out. |
| **Projected Shortfall** | Forecast demand minus stock, ranked | The size of the gap, so a limited budget goes to the biggest hole. |
| **Supplier Lane Risk** | Category × seller state, coloured by lead-time coefficient of variation | Risk is unpredictability, not slowness. Colouring by CV rather than mean is the whole point — see the memo, §2. |

**Polish worth adding by hand**

1. **Action Queue** — sort descending by Recommended Order Units. Set the
   `Stock Status` palette explicitly: STOCKOUT `#8B0000`, CRITICAL `#D32F2F`,
   REORDER `#F9A825`.
2. **Cover vs Lead Time** — add a reference line at
   `Weeks Of Cover = Lead Time Mean Days / 7`. This is the sheet's actual
   insight and the generator cannot express it.
3. **Filter action** — clicking a category in the Action Queue should filter
   every other sheet. Dashboard → Actions → Add → Filter.
4. Keep the simulated-stock caption at the bottom. It is generated in, and
   `tests/test_tableau.py` fails if it goes missing.

---

## Workbook 2 — Retail Performance & Customer Intelligence

| Sheet | Shows | Why it is on an executive dashboard |
|---|---|---|
| **Revenue Trajectory** | Monthly revenue, 23 complete months | One chart, no filters, readable standing up. |
| **Average Order Value** | Monthly AOV | Separates volume growth from value growth — different strategic responses. |
| **Where The Value Is** | Scatter: revenue share against customer share, sized by revenue, coloured by relative churn risk | Distance from the 45° line is the argument for where retention budget goes. *At Risk High-Value* sits at 29.6% of revenue on 15.6% of customers. |
| **Churn Exposure By State** | Revenue at risk per state, coloured by late-delivery rate | Regional investment decisions. The colour is the hypothesis: exposure tracks fulfilment. |
| **Sentiment vs Late Delivery** | Scatter of monthly sentiment against late-delivery rate, coloured by month | If these move together, sentiment is a logistics problem, not a product problem — a different remediation budget. Measured correlation: **r = −0.42**. |
| **Category Sentiment Risk** | Mean sentiment per category | Where the complaints concentrate. |

**Polish worth adding by hand**

1. **Where The Value Is** — add a 45° reference line (`y = x`). Without it the
   scatter is just points; with it, it is an argument.
2. **Churn Exposure By State** — sort descending, then optionally convert to a
   map (set `State` → Geographic Role → State/Province, data source country
   Brazil). Keep the sorted bar alongside: the map shows the pattern, the bar
   gives the number someone quotes in a meeting.
3. **Turn it into a Story**, not just a dashboard, with three captioned points:
   *"Revenue is growing but concentrating"* → *"Value sits in segments we are
   losing"* → *"Sentiment tracks delivery, not product"*.

---

## The check before you call it done

Open both side by side and ask: **could I swap any chart between them without
it looking out of place?** If yes, that chart is not designed for its audience
yet.

The operations workbook should look almost unreadable to an executive — too
dense, too granular, too many SKUs. The executive workbook should look useless
to a warehouse manager — no SKUs, no actions, nothing about this week.

That mutual unreadability is the deliverable.
