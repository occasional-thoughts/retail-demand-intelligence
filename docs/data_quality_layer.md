# The data-quality layer

**Live:** https://claude.ai/artifact/VjWJmUKzZWsns7wRktwBqJ

A profiling layer over the four raw source files, aimed at a data steward
rather than an engineer. It is deliberately **not** part of the Kedro pipeline.

```bash
make quality      # profile the raw files -> quality/source_profile.json
make dashboards   # build the explorer page from that profile
```

---

## Why a separate layer at all

It would be trivial to compute these statistics inside the pipeline. That is
not the point, and saying so plainly is part of the deliverable.

The pipeline and this layer serve **different people at different moments**:

| | Kedro pipeline | Quality explorer |
|---|---|---|
| Audience | Data engineer | Data steward / business analyst |
| When | Every scheduled refresh, in production | When a file arrives, or a rule is questioned |
| Question | "Did the load succeed and are the numbers right?" | "What is wrong with this file, and where should the line be?" |
| Changing a rule needs | A pull request, a review, a deploy | Moving a slider |
| Output | Validated tables, a machine-readable report | A score, a distribution, and a remediation mapping |

The analyst who owns supplier data quality is not going to open a repository to
find out that 610 products have no category. That asymmetry is the entire
argument for a second layer, and it is why this one is kept out of the DAG:
wiring it in would make it a duplicate of work the pipeline already does.

---

## What makes it low-code rather than just a report

The checks come in two kinds.

**Fixed checks** — nulls, duplicates, key uniqueness. There is nothing to tune;
a null is a null. These are computed once by `scripts/profile_sources.py`.

**Tunable checks** — outlier rules, whose answer depends entirely on where you
draw the line. Rather than hard-coding a threshold, the profiler exports the
full value distribution (value/count pairs plus quartiles), and the page
recomputes the count at whatever fence the steward picks.

So the steward can change:

- the **outlier fence** (IQR × 1.0 to 3.0)
- the **penalty weight** for failing and warning rows
- the **pass mark**
- whether **expected-sparsity checks** count toward the score

…and every figure on the page — score, verdict, per-check counts, the
distribution chart, the remediation table — recomputes live. No Python, no
deploy, no engineer.

The client-side arithmetic is verified against the Python profiler: at IQR ×3.0
both report **13,189** warning rows, to the row.

---

## What the profile found

Score **91.8 / 100** at default settings, over 344,266 rows and 14 checks.

**`order_items` is clean.** Every fixed check passes across all 112,650 rows —
no duplicate keys, no non-positive prices, no negative freight, no missing
values. Worth reporting in its own right: "the transactional core is sound" is
useful to a steward. But it means a profile limited to this one file finds
almost nothing.

The defects live elsewhere:

| File | Finding | Count |
|---|---|---|
| products | missing category | **610** of 32,951 |
| products | no English translation | **2** — `pc_gamer`, `portateis_cozinha_e_preparadores_de_alimentos` |
| reviews | duplicate `review_id` | **814** |
| reviews | no body / no title | 58,247 / 87,656 — *reported, not scored* |
| orders | never delivered | **2,965** |
| order_items | price outliers at IQR ×1.5 | 8,427 — *tunable* |

### One scoring decision worth defending

Counting the 58,247 reviews with no body as a warning drags the score to
**53/100**. That number would be wrong in a way that costs real time: the
sparsity is precisely *why* review text sits in a document store rather than a
relational column, and a steward chasing it would be chasing a non-problem.

Those checks are reported as **INFO** and excluded from the score by default —
with a toggle, because it is an editorial judgement and the steward is entitled
to disagree with it.

---

## Closing the loop

The layer is only a deliverable if its findings changed the pipeline. Every
breaching check on the page carries the decision it justified and the file that
decision lives in:

| Finding | What the pipeline does about it | Where |
|---|---|---|
| 610 products have no category | `COALESCE(..., 'unknown')` rather than dropping the rows | `conf/base/catalog.yml` |
| 2 categories absent from the translation table | No FK on that column — an FK there would reject the load; the gap is handled in the silver layer | `scripts/sql/01_schema.sql` |
| `review_id` is not unique (814 duplicates) | Surrogate `review_key` primary key, so duplicates are preserved rather than rejected | `scripts/sql/01_schema.sql` |
| 58.7% of reviews have no body | Review text lives in MongoDB, not in a relational column of mostly nulls | `scripts/generate_events.py` |
| 7.48% of prices are outliers | WAPE rather than MAPE as the forecast metric, so extremes cannot dominate | `pipelines/forecasting/nodes.py` |
| 2,965 orders never delivered | Lead times computed only from delivered orders, excluded explicitly rather than silently producing null intervals | `pipelines/scale/nodes.py` |

"Profiling found X, so the pipeline does Y" — six times, each one checkable
against the code.
