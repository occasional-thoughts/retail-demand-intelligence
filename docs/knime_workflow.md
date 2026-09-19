# Phase 2 — KNIME data-quality workflow

## Why this exists alongside the Kedro pipeline

It would be trivial to compute these statistics in pandas. That is not the
point, and saying so plainly is part of the deliverable.

The Kedro pipeline and this workflow serve **different people at different
moments**:

| | Kedro pipeline | KNIME workflow |
|---|---|---|
| Audience | Data engineer | Business analyst / data steward |
| When it runs | Every scheduled refresh, in production | Once, up front, and again when a source changes |
| Question it answers | "Did the load succeed and are the numbers right?" | "What is wrong with this file, and what must the pipeline handle?" |
| Changing it requires | A pull request, a review, a deploy | Dragging a node and hitting execute |
| Output | Validated tables and a machine-readable report | A human-readable scorecard |

The analyst who owns supplier data quality is not going to open a Python
repository to find out that 610 products have no category. They need to
profile a file the morning it arrives and hand the engineer a list. That is
what a low-code tool is genuinely for, and why this workflow is **deliberately
not** wired into the Python pipeline. Wiring it in would make it a redundant
copy of work the pipeline already does; keeping it separate makes it a
different capability.

**The workflow's findings are what justify specific nodes in the Kedro
pipeline.** That link is the deliverable — see "Closing the loop" at the end.

---

## Install

**Already installed on this machine** at `~/knime/KNIME 5.12.0.app` (Apple
Silicon build). Launch it, pick `~/knime-workspace` as the workspace, then
`File → New → KNIME Workflow` and name it `olist_data_quality_profile`.

```bash
open ~/knime/KNIME\ 5.12.0.app
```

### Why this workflow is built in the GUI and not shipped as a .knwf file

Everything else in this project is generated from code — including the Tableau
workbooks, which are XML and therefore scriptable. The KNIME workflow is the
one deliverable that is not, and the reason is worth stating rather than
hiding:

1. **No headless executor.** KNIME's batch executor
   (`org.knime.product.KNIME_BATCH_APPLICATION`) is not distributed on the
   public update site for 5.12 — `p2.director -list` against
   `update.knime.com/analytics-platform/5.12` returns no matching feature. So a
   generated workflow could not be *executed* to prove it works.
2. **The entry node is not hand-authorable.** In 5.12 the CSV Reader persists
   its settings through `CSVTableReaderNodeParameters` with a custom
   `ConfigAndSourceSerializer`, rather than the classic flat `NodeSettings`
   key/value format that the bundled 3.7-era metanode templates use.

Generating a `.knwf` under those two constraints would mean shipping an
unverifiable file whose reader node needs reconfiguring anyway. The node
inventory below was verified against the actual install instead, so the recipe
is known-correct for this version — which is the part that was genuinely worth
checking.

### Node inventory, verified against KNIME 5.12.0

Every node this recipe uses was confirmed present in the installed plugins:

| Node in the recipe | Factory class | Plugin |
|---|---|---|
| CSV Reader | `org.knime.base.node.io.filehandling.csv.reader.CSVTableReaderNodeFactory` | `org.knime.base` |
| Row Sampling | `org.knime.base.node.preproc.sample.SamplingNodeFactory` | `org.knime.base` |
| Statistics | `org.knime.base.node.viz.statistics2.Statistics3NodeFactory` | `org.knime.base` |
| Missing Value | `org.knime.base.node.preproc.missingval.MissingValueHandling3NodeFactory` | `org.knime.base` |
| Numeric Outliers | `org.knime.base.node.stats.outlier.handler.NumericOutliersNodeFactory` | `org.knime.base` |
| Duplicate Row Filter | `org.knime.base.node.preproc.duplicates.DuplicateRowFilterNodeFactory` | `org.knime.base` |
| Rule Engine | `org.knime.base.node.rules.RuleEngineNodeFactory` | `org.knime.base` |
| GroupBy | `org.knime.base.node.preproc.groupby.GroupByNodeFactory` | `org.knime.base` |
| Math Formula | `org.knime.ext.jep.JEPNodeFactory` | `org.knime.ext.jep` |
| CSV Writer | `org.knime.base.node.io.filehandling.csv.writer.CSVWriter2NodeFactory` | `org.knime.base` |

Note the Rule Engine sits in `org.knime.base.node.rules`, **not**
`...node.rules.engine` — a detail worth knowing if you ever script against it.
No extra extensions need installing; Math Formula's JEP plugin ships with the
base product.

---

## Build the workflow

Three branches, one per file. Every node below was confirmed present in the
KNIME 5.12 install. Find a node by typing its name in the node search on the
left, drag it onto the canvas, and connect nodes by dragging from the output
triangle on one node to the input triangle on the next. Open a node's settings
with a double-click. Run everything with **Execute all** (F7).

If a setting is named slightly differently from what is written here, the
node's description panel explains every option. Look it up there instead of
guessing.

### Branch A: order items (the sampled profile the brief asks for)

**A1. CSV Reader**: file `data/raw/olist_order_items_dataset.csv`.
Execute it and check the output shows **112,650 rows**.

**A2. Row Sampling**: relative, **10%**, random sampling with a **fixed seed
of 42**. A fixed seed means you get the same sample on every run.

**A3. Rule Engine**: flag each row. Output column name: `quality_flag`.

```
MISSING $price$                 => "FAIL: missing price"
$price$ <= 0                    => "FAIL: non-positive price"
$freight_value$ < 0             => "FAIL: negative freight"
MISSING $freight_value$         => "WARN: missing freight"
$price$ > 277.4                 => "WARN: price outlier"
TRUE                            => "OK"
```

Rules run top to bottom and the first match wins, which is why the `MISSING`
checks come before the comparisons. `277.4` is the price's upper IQR fence
(Q3 134.9 + 1.5 × IQR 95.0). The lower fence is negative, so low outliers
cannot occur.

**A4. GroupBy**: group column `quality_flag`. In the aggregation tab, add
`quality_flag` with method **Count**. The result is one row per flag with its
count.

**A5. CSV Writer**: `knime/output/order_items_scorecard.csv`.

**Expected result:** roughly 92.5% `OK` and about 7.5% `WARN: price outlier`,
with no `FAIL` rows. Because this branch is a 10% sample, the counts will be
about 11,265 rows in total and the percentages will be close to these, not
exact.

### Branch B: products (full file)

**B1. CSV Reader**: `data/raw/olist_products_dataset.csv`, 32,951 rows.

**B2. GroupBy** with **no group column**, which aggregates the whole table
into a single row. Aggregations:
- `product_id`: **Count**
- `product_category_name`: **Missing value count**

**B3. CSV Writer**: `knime/output/products_scorecard.csv`.

**Expected result:** 32,951 products, of which **610** have a missing
category.

### Branch C: reviews (full file)

**C1. CSV Reader**: `data/raw/olist_order_reviews_dataset.csv`, 99,224 rows.

**C2. GroupBy** with **no group column**. Aggregations:
- `review_id`: **Count**
- `review_id`: **Unique count**
- `review_comment_message`: **Missing value count**
- `review_comment_title`: **Missing value count**

**C3. CSV Writer**: `knime/output/reviews_scorecard.csv`.

**Expected result:** 99,224 reviews, **98,410** unique IDs (so **814**
duplicate rows), **58,247** with no message and **87,656** with no title.

### Finish

1. Add a **workflow annotation** above each branch that says what it checks.
   Right-click the canvas and choose *New workflow annotation*. The analyst
   reading this canvas later needs those labels.
2. **Export** the workflow: *File → Export KNIME Workflow* and save it to
   `knime/olist_data_quality_profile.knwf`.
3. Compare your numbers with the expected figures above and with
   `knime/expected_figures.json`. If a figure is far off, the most likely
   cause is a column read with the wrong type. Check the CSV Reader's
   transformation tab.

---

## Important: profile more than one file

`order_items` on its own is **clean**. Verified over all 112,650 rows:

| Check | Failures |
|---|---|
| Primary key uniqueness (`order_id` + `order_item_id`) | 0 |
| Non-positive price | 0 |
| Negative freight | 0 |
| Missing price / freight / shipping limit | 0 |
| Price IQR outliers (1.5×) | 8,427 — **7.48%** |

So a profile limited to `order_items` produces a quality score of 92.5/100 and
one finding. That is a real result and worth reporting — "the transactional
core is sound" is useful to a steward — but it makes for a thin scorecard, and
more importantly it is not where this dataset's defects live.

**Add a second CSV Reader branch for `olist_products_dataset.csv` and a third
for `olist_order_reviews_dataset.csv`.** That is where the profile earns its
keep. Verified figures to check your workflow against:

| File | Finding | Figure |
|---|---|---|
| products | `product_category_name` missing | **610** of 32,951 (1.85%) |
| products | categories with no English translation | **2** — `pc_gamer`, `portateis_cozinha_e_preparadores_de_alimentos` |
| reviews | `review_id` not unique | **814** duplicates (99,224 rows, 98,410 distinct) |
| reviews | no review body | **58,247** — 58.7% |
| reviews | no review title | **87,656** — 88.3% |
| orders | never delivered | **2,965** |

Run `python scripts/knime_reference_figures.py` for the `order_items` numbers
in machine-readable form (`knime/expected_figures.json`). It is a verification
fixture only — deliberately outside the Kedro pipeline — so that a
mis-configured Rule Engine produces a scorecard you can catch rather than a
confident page of wrong numbers.

---

## Closing the loop — the part that matters

The workflow is only a deliverable if its findings changed the pipeline. Every
row below is a real defect in the source data and a real decision in the code:

| KNIME finding | What the Kedro pipeline does about it | Where |
|---|---|---|
| 610 products have no category | `COALESCE(..., 'unknown')` rather than dropping the rows | `conf/base/catalog.yml`, `pg_order_items` |
| 2 categories absent from the translation table | No FK declared on that column — an FK there would reject the load; the gap is handled in the silver layer | `scripts/sql/01_schema.sql` |
| `review_id` is not unique (814 duplicates) | Surrogate `review_key` primary key, so duplicates are preserved rather than rejected | `scripts/sql/01_schema.sql` |
| 58.7% of reviews have no body, 88.3% no title | Review text lives in MongoDB, not in a relational column of mostly nulls | `scripts/generate_events.py` |
| 7.48% of prices are IQR outliers | WAPE rather than MAPE as the forecast metric, so a handful of extremes cannot dominate | `pipelines/forecasting/nodes.py` |
| 2,965 orders never delivered | Lead times computed only from delivered orders; the rest are excluded explicitly rather than silently producing null intervals | `pipelines/scale/nodes.py` |

"Profiling found X, so the pipeline does Y" is the sentence an interviewer is
listening for. This table is that sentence, six times.
