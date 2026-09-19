# Retail Demand Forecasting & Inventory Intelligence Platform

An end-to-end analytics platform built on the [Olist Brazilian e-commerce
dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce): three
mismatched source systems unified through an orchestrated pipeline, two
validated models, and two dashboards built for two genuinely different
audiences.

**What it demonstrates:** relational + document + distributed-file integration,
enforced cross-system data quality, leakage-free time-series validation, a
controlled feature ablation, and the translation of model output into
operational and strategic decisions.

**Live dashboards** (interactive, built from the pipeline's own output):

| | |
|---|---|
| **Inventory Operations** &mdash; forward 8 weeks, category grain, daily | https://claude.ai/artifact/1jan7w6LEFmvnbiEQCDM1g |
| **Retail Performance Review** &mdash; backward 23 months, company grain, monthly | https://claude.ai/artifact/1oTTzRGbfGZRCCthhtzguu |

---

## Quick start

```bash
git clone <this repo> && cd retail-intel
```

1. **Get the data.** Download the Olist dataset from Kaggle and unzip the nine
   CSVs into `data/raw/`.

2. **Start the source systems.** Either path works and both produce identical
   databases on identical ports:

   ```bash
   docker compose up -d          # preferred, fully reproducible
   ./scripts/local_db.sh start   # no Docker: conda-managed Postgres + MongoDB
   ```

3. **Build the environment.**
   ```bash
   conda create -n retail python=3.11 -y && conda activate retail
   pip install -r requirements.txt && pip install -e . --no-deps
   ```

4. **Run everything.**
   ```bash
   make sources     # load Postgres, generate the MongoDB event store
   make pipeline    # the full Kedro run
   make dashboards  # generate both Tableau workbooks
   make test        # the test suite
   ```

Extracts land in `data/08_reporting/`; workbooks in `tableau/`.

---

## Architecture

Full diagram and tool-by-tool justification: **[docs/architecture.md](docs/architecture.md)**

```
PostgreSQL ─┐
MongoDB    ─┼──▶ Kedro ──▶ bronze ──▶ integrity ──▶ silver ──▶ gold ──┬──▶ demand forecast ──▶ ops_*.csv    ──▶ Tableau (operations)
Parquet    ─┘    (Spark)      │          │                            └──▶ churn risk      ──▶ exec_*.csv   ──▶ Tableau (executive)
                              │          ▼
                              │     quarantine
                              ▼
                        KNIME (separate, no-code profiling)
```

### The three source systems, and why they are different on purpose

| Source | Holds | Why not just use one database |
|---|---|---|
| **PostgreSQL** | orders, order_items, products, sellers, customers, payments, review scores | Fixed schema with hard relationships. PKs and FKs make bad data *impossible to insert*, rather than something a downstream job hopes to notice. |
| **MongoDB** | review text, browsing sessions with nested event arrays | 59% of reviews have no body; most have no title; the moderation block exists on ~8%. Sessions are a variable-length array of differently-shaped events. Relationally this is a wide table of nulls. |
| **Parquet + Spark** | 2.9M carrier scan events | Generated from a 112k-row seed, partitioned by month, aggregated with percentiles — larger than its own source and shuffle-heavy. |

---

## The parts worth reading

### 1. Referential integrity that fails loudly

No foreign key can span two databases. When the order system and the web
platform are maintained separately, references drift — so ~5% of the event
store deliberately points at orders and customers that do not exist.

A naive pipeline inner-joins the two sources, silently loses those rows, and
reports a clean run. This one
([`pipelines/integrity/nodes.py`](src/retail_intel/pipelines/integrity/nodes.py)):

- **separates** matched from unmatched instead of dropping,
- **quarantines** the unmatched rows to `data/quarantine/` with a reason and a
  timestamp, so a person can investigate them,
- **stops the run** when breakage exceeds a ceiling,
- **stops the run when it finds no breakage at all** — a check that never fires
  is indistinguishable from a check that is broken, and a silent no-op test is
  worse than no test because it buys false confidence.

Every injected orphan ID is written to `data/quarantine/expected_orphans.json`,
so the checks are tested against independent ground truth rather than against
themselves.

### 2. Walk-forward validation, and two leakage bugs the tests caught

A random train/test split on a time series lets the model train on July and be
scored on June. The metrics look excellent and mean nothing. This uses
expanding-window walk-forward validation, with one model per horizon step so
that no feature reaches into its own prediction window.

The test suite asserts the absence of leakage rather than assuming it — and it
earned its keep. Four real defects were caught during the build, three of which
produced *better* numbers and no error:

| Defect | Symptom | How it was caught |
|---|---|---|
| **Backtest trained on its own test rows.** Training was selected by origin week `<= cutoff`; the test rows' origin *is* the cutoff, so they sat in both sets. | WAPE **0.102**. True value: **0.402**. | The result looked too good; fold construction was re-read, then pinned by a test asserting train/test disjointness. |
| **Weekly panel silently zeroed.** `to_period("W-MON").start_time` yields *Tuesdays*; `date_range(freq="W-MON")` yields *Mondays*. The densifying join matched nothing. | Every week became 0 units. Right shape, right row count, no error. | `is_forecastable` dropped to 0 of 74 categories. Now guarded by an assertion that densifying cannot destroy the measure. |
| **Rolling windows crossed category boundaries.** `shift(1)` was grouped; the `.rolling()` after it was not. | Each category's opening weeks absorbed the previous category's closing weeks. | `test_first_row_of_each_category_has_no_history` |
| **Hardcoded feature name** `units_roll_mean_12`, which only exists under the default parameters. | `KeyError` whenever `rolling_windows` changed. | `test_lag_features_only_look_backwards` |

The first two are the instructive ones: both made the model look *better*, and
neither raised anything. A pipeline that cannot fail loudly will fail quietly.

Metrics are always reported against naive and seasonal-naive baselines, each
scored only on rows where that baseline is defined. Filling a missing 52-week
lag with zero does not measure the seasonal baseline; it measures a
constant-zero forecast and flatters the model for an unrelated reason.

### 3. The sentiment tooling decision

The brief permits "VADER or a lightweight transformer". **VADER is the wrong
tool for this dataset, and using it would have produced a false finding.**

Olist reviews are written in Brazilian Portuguese. VADER's lexicon is English.
Validated against star rating — independent ground truth the customer supplied
alongside the text:

| scorer | Spearman vs. rating | returns 0.0 for | 1★ → 5★ separation |
|---|---|---|---|
| **Portuguese lexicon** (this repo) | **0.687** | 12.6% | 0.43 |
| VADER (English) | 0.133 | **81.3%** | 0.06 |

VADER cannot read four out of five reviews. Had it been used, the ablation in
the next section would have shown sentiment contributing nothing — a tooling
artefact presented as a finding about customers.

So `src/retail_intel/sentiment.py` implements a Portuguese lexicon scorer
(negation windows, intensifiers, multi-word phrases, accent folding) and keeps
VADER as a documented baseline. The validation runs on every pipeline
execution, so the claim is re-checked rather than asserted once.

### 4. An ablation that asks the sharp question

"Does sentiment help?" is too easy — a text-sentiment score correlates with a
star rating, so a model containing both would credit sentiment for the
rating's work.

The question tested instead: **does reading the review text add anything beyond
the star rating the customer already gave?** The baseline arm keeps
`avg_review_score` and `min_review_score` and removes only the NLP-derived
features. Both arms share one split, one seed, one hyperparameter set.

The AUC difference is then **bootstrapped, paired, on the test set**, because a
gap of a few thousandths between two models is noise, and reporting it as an
improvement is exactly what the ablation exists to prevent. The pipeline ships
whichever feature set the ablation justifies — if the text features do not
clear the bar, the simpler model goes to production.

### 5. Two dashboards that would be useless to each other's audience

Both dashboards are **generated** from the pipeline's extracts rather than
clicked together, so they cannot drift from the data and the design is
reviewable in a diff.

They ship in two forms:

| Form | Built by | Status |
|---|---|---|
| **Published web dashboards** (links above) | `scripts/build_web_dashboards.py` | **Verified** &mdash; rendered and checked in a browser |
| **Tableau workbooks** (`tableau/*.twb`) | `scripts/build_tableau_workbooks.py` | Generated and structurally tested; **never opened in Tableau** |

The `.twb` files are real, generated Tableau XML with 8 tests over their
structure — but Tableau Public needs an administrator install that was not
available on the build machine, so nobody has yet opened them. That is stated
here rather than implied away.

Guides: **[docs/tableau_build_guide.md](docs/tableau_build_guide.md)**

| | Operations | Executive |
|---|---|---|
| Reader | Inventory manager | Commercial leadership |
| Cadence | Daily | Monthly |
| Time direction | Forward 8 weeks | Backward 24 months |
| Grain | Category, supplier lane | Company, segment, region |
| Question | "What do I order today?" | "Where is value moving, and what threatens it?" |

The check applied to both: *could any chart be swapped between them without
looking out of place?* If yes, it is not designed for its audience yet.

Audience separation is **enforced, not just intended**: the build asserts that
the operational workbook references no `exec_` extract and vice versa, and
`tests/test_tableau.py` re-checks that plus the reference graph — a field on a
shelf with no matching `column-instance` renders an empty sheet in Tableau
without raising anything, so it is asserted here instead.

### 6. The no-code layer, and why it is not redundant

Build guide: **[docs/knime_workflow.md](docs/knime_workflow.md)** — verified
node-by-node against KNIME 5.12.0, with every factory class confirmed present
in the install.

The KNIME workflow profiles raw data quality and emits a scorecard. It is
deliberately **not** wired into the Python pipeline. The analyst who owns
supplier data quality is not going to open a repository to discover that 610
products have no category — they need to profile a file the morning it arrives
and hand the engineer a list. The workflow's findings are traced to the
specific pipeline nodes that handle them.

---

## Results

### Demand forecast — beats naive at every horizon

Expanding-window walk-forward, 5 folds, 8-week horizon, 37 forecastable
categories:

| | WAPE | vs. model |
|---|---|---|
| **Model** (pooled-horizon gradient boosting) | **0.328** | — |
| Naive (next week = last week) | 0.350 | model **6.2% better** |
| Seasonal naive (same week last year) | 0.849 | model **61.8% better** |

Per-horizon, model vs. naive: 0.273/0.308, 0.275/0.285, 0.272/0.310,
0.405/0.417, 0.396/0.420, 0.337/0.355, 0.310/0.324, 0.333/0.359 — the model
wins at all eight.

The margin is modest and reported as such. Getting there required the model to
be made *simpler*, not more complex: one pooled model instead of eight
per-horizon models, a trimmed feature set, and heavier regularisation. Earlier
per-horizon configurations lost to the naive baseline outright, and that was
reported before it was fixed.

**Stated caveat:** hyperparameters were selected on the same folds used to
report, so 6.2% is mildly optimistic. An unbiased figure needs a held-out
period. This is recorded in `demand_metrics.json` under `caveat`.

### Sentiment ablation — a clean negative result

| | ROC-AUC | PR-AUC | Lift @ decile |
|---|---|---|---|
| With sentiment | 0.567 | 0.029 | 1.99 |
| **Without sentiment** | **0.577** | 0.029 | **2.05** |

ΔAUC **−0.0096**; paired bootstrap over 1,000 resamples gives a 95% interval of
**[−0.032, +0.013]**, spanning zero. Permutation importance of every sentiment
feature is negative. **Verdict: the NLP features do not earn their place**, and
the pipeline ships the simpler model automatically.

This is the outcome the ablation was built to detect. It is also why the
Portuguese scorer mattered: with VADER — blind to 81% of these reviews — the
same null result would have been produced by the tooling, and been
indistinguishable from a real finding about customers.

Sentiment does survive as a *diagnostic*: monthly sentiment correlates with
late-delivery rate at **r = −0.42**.

### Churn ranking — weak model, useful ordering

Population return rate is **1.14%**. The score still ranks:

| Risk band | Customers | Actual return rate | Lift vs. base |
|---|---|---|---|
| Low | 29,047 | 2.08% | **1.82×** |
| Moderate | 17,427 | 0.30% | 0.27× |
| High | 8,714 | 0.08% | 0.07× |
| Critical | 2,905 | 0.00% | 0.00× |

The top-scoring half contains **91% of all 664 repeat purchasers**. Monotone
across all four bands.

### Scale layer

**2,699,671** carrier scan rows generated from 110,181 order lines, partitioned
by year and month, aggregated to lead-time distributions across **369
category/state lanes**.

### Data quality

| Check | Examined | Quarantined | Rate |
|---|---|---|---|
| `reviews.order_id → orders.order_id` | 99,224 | 4,950 | 4.99% |
| `sessions.customer_id → customers.customer_id` | 110,615 | 12,458 | 11.26% |

Both within their per-check thresholds, nothing lost, every injected orphan
recovered — verified against the ground-truth manifest by
`tests/test_integration_ground_truth.py`.

**The session rate is not worse data.** It is the same 5% document-level
breakage measured after a customer-level aggregation: valid sessions collapse
together, orphans each carry a unique dangling id and collapse into nothing.
The first run failed on this, against a threshold written at the wrong grain.
Thresholds are now per-check, and the reason is written down in
`parameters.yml` — the alternative being to widen one global number until the
check stops complaining.

---

## Engineering standards

| | |
|---|---|
| **Tests** | 74 tests: 59 unit (no database needed — nodes are pure functions, which is why connections live in the catalog) + 7 integration verifying the integrity layer against the orphan ground-truth manifest + 8 validating the generated workbooks |
| **Leakage** | asserted, not assumed: rolling windows, lag alignment, fold ordering, train/test disjointness |
| **Assumptions** | every business number lives in `conf/base/parameters.yml`, reviewable by someone who does not read Python |
| **Secrets** | `conf/local/` is gitignored; no credential is committed |
| **Reproducibility** | fixed seeds throughout; the orphan-injection manifest is written to disk |
| **Failure** | integrity breaches stop the run; quarantined rows are persisted with a reason |
| **Scheduling** | launchd (`scripts/schedule/`) locally, GitHub Actions in CI |

---

## Honest limitations

Stated plainly, because a portfolio project that hides these is less credible,
not more:

1. **Inventory is simulated.** Olist publishes no stock ledger, no supplier
   contracts and no reorder policy. Opening stock is generated from a
   documented deterministic policy (6 weeks of cover, 95% service level). The
   *forecast demand* and *lead times* underneath it are derived from real order
   and delivery timestamps. Every artefact carrying simulated stock is flagged
   `simulated_stock = true`, and the dashboards caption it.

2. **The clickstream is synthetic.** Olist ships no browsing data. Sessions are
   generated deterministically, anchored to real purchase timestamps so the
   derived behavioural features carry structure rather than noise. They are a
   demonstration of document-store modelling, not a finding about real users.

3. **Carrier scan events are a reconstruction.** The *intervals* are real —
   every scan is interpolated between genuine Olist timestamps, so aggregated
   lead times are the dataset's own. The per-scan granularity is synthetic.

4. **Churn is heavily imbalanced.** Olist customers are overwhelmingly one-time
   buyers. The positive class is left small rather than resampled: a retention
   budget is allocated off calibrated probabilities, not off accuracy.

5. **This was never deployed to Microsoft Fabric.** Fabric capacity is not
   available through the author's university tenant and Power BI Desktop has no
   macOS build. `docs/architecture.md` contains a component-by-component
   migration design. It is a design, not a description of something that ran.

---

## Repository layout

```
conf/base/catalog.yml        41 datasets: SQL, Mongo, Spark, Parquet, CSV
conf/base/parameters.yml     every business assumption, in one reviewable file
src/retail_intel/
  datasets/                  custom MongoDB Kedro datasets
  sentiment.py               Portuguese lexicon scorer + VADER baseline
  pipelines/
    ingestion/               three sources -> bronze
    integrity/               cross-system FK checks -> quarantine
    scale/                   Spark: build and aggregate the scan feed
    features/                silver conformance, two gold feature stores
    forecasting/             walk-forward backtest, refit, forecast
    churn/                   ablation, model, scoring
    serving/                 operational and executive extracts
dashboards/                  web dashboard templates + built pages
tableau/                     generated .twb workbooks (operations + executive)
scripts/
  build_web_dashboards.py    payloads + injection for the published pages
  build_tableau_workbooks.py the Tableau dashboard definitions
  tableau/_twb.py            minimal Tableau .twb XML writer
  sql/01_schema.sql          the relational contract
  load_postgres.py           parent-first load with FKs enforced
  generate_events.py         event store + deliberate orphan injection
  local_db.sh                Docker-free database path
tests/                       56 tests, database-free
docs/                        architecture, KNIME guide, Tableau guide, memo
```
