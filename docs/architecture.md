# Architecture

## Data flow

```
 SOURCE SYSTEMS                  PIPELINE (Kedro)                    CONSUMERS
 ─────────────────               ────────────────                    ─────────

┌──────────────────┐
│   PostgreSQL     │  orders, order_items, products,
│                  │  sellers, customers, payments,
│  relational      │  review scores
│  order mgmt      │────┐
└──────────────────┘    │
                        │     ┌─────────────────────┐
┌──────────────────┐    ├────▶│  BRONZE             │
│   MongoDB        │    │     │  land each source   │
│                  │    │     │  unmodified         │
│  review text     │────┤     └──────────┬──────────┘
│  web sessions    │    │                │
│  (nested events) │    │                ▼
└──────────────────┘    │     ┌─────────────────────┐      ┌──────────────┐
                        │     │  INTEGRITY          │─────▶│  QUARANTINE  │
┌──────────────────┐    │     │  cross-system FK    │      │  orphaned    │
│  Parquet @ scale │    │     │  checks, fail loud  │      │  records +   │
│                  │◀───┘     └──────────┬──────────┘      │  JSON report │
│  2.9M carrier    │                     │                 └──────────────┘
│  scan events     │────────┐            ▼
│  (Spark)         │        │  ┌─────────────────────┐
└──────────────────┘        └─▶│  SILVER             │
         ▲                     │  conformed,         │
         │                     │  validated,         │
   built by a Spark            │  sentiment-scored   │
   node from real              └──────────┬──────────┘
   delivery timestamps                    │
                                          ▼
                               ┌─────────────────────┐
                               │  GOLD               │
                               │  weekly demand +    │
                               │  customer features  │
                               └──────────┬──────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
             ┌─────────────────────┐            ┌─────────────────────┐
             │  DEMAND FORECAST    │            │  CHURN RISK         │
             │  walk-forward       │            │  + sentiment        │
             │  validated, 8 wks   │            │  ablation           │
             └──────────┬──────────┘            └──────────┬──────────┘
                        │                                  │
                        ▼                                  ▼
             ┌─────────────────────┐            ┌─────────────────────┐
             │  ops_*.csv          │            │  exec_*.csv         │
             │  inventory position │            │  revenue, segments, │
             │  reorder alerts     │            │  churn exposure,    │
             │  supplier risk      │            │  sentiment trend    │
             └──────────┬──────────┘            └──────────┬──────────┘
                        ▼                                  ▼
             ┌─────────────────────┐            ┌─────────────────────┐
             │  TABLEAU            │            │  TABLEAU            │
             │  Operations         │            │  Executive          │
             │  (daily, forward,   │            │  (monthly, backward,│
             │   SKU-grain)        │            │   company-grain)    │
             └─────────────────────┘            └─────────────────────┘

 ── separate, deliberately not connected ──────────────────────────────────

┌──────────────────┐            ┌─────────────────────┐
│  4 raw CSVs      │───────────▶│  QUALITY EXPLORER   │──▶ live score +
│  as they arrived │            │  tunable rules      │    remediation map
└──────────────────┘            └─────────────────────┘    (for analysts)
```

## Why each tool is here

Every one of these is load-bearing. Swapping any of them out would mean losing
a capability, not just changing a name.

| Tool | Why *this* and not something else |
|---|---|
| **PostgreSQL** | Orders and customers have a fixed schema and hard relationships. Primary and foreign keys make duplicates and dangling references *impossible to insert*, rather than something the pipeline hopes to catch later. |
| **MongoDB** | Review documents have genuinely optional fields (most reviews have no title, 59% have no body at all) and sessions are a nested array of variable-shaped events. Relationally that is either a wide table of nulls or a pile of sparse side-tables. Server-side aggregation also summarises ~2M nested events without pulling them over the wire. |
| **PySpark** | The carrier scan feed is ~2.9M rows generated from a 112k-row seed, partitioned by month, then aggregated with percentiles. Larger than its own source, shuffle-heavy, and partition-aware on write — the smallest honest case for a distributed engine. |
| **Kedro** | 29 nodes across 7 pipelines with an explicit DAG. Connections live in the catalog, so nodes are pure functions of their inputs — which is what lets the entire test suite run in seconds with no database. A failing layer can be re-run alone. |
| **Python / scikit-learn** | Walk-forward backtesting and a paired bootstrap ablation are bespoke evaluation logic. `HistGradientBoosting*` was chosen over LightGBM/XGBoost specifically because it needs no external OpenMP library — one fewer install failure between a reviewer and a working repo. |
| **Quality explorer** | A different *audience*, not a different implementation. Outlier rules are tunable in the page, so a steward changes a threshold without a deploy. See `docs/data_quality_layer.md`. |
| **Tableau** | Two workbooks for two audiences. See `docs/tableau_build_guide.md`. |
| **launchd / GitHub Actions** | Scheduled refresh. The local stand-in for Fabric Data Factory. |

## If Microsoft Fabric access becomes available

This project was built locally because Fabric capacity is not available
through the author's university tenant, and Power BI Desktop has no macOS
build. The architecture was designed so that constraint changes deployment
only, not design — the medallion layering, the partitioned Parquet, and the
separation of serving extracts from marts are all Fabric-shaped already.

| Local implementation | Fabric equivalent |
|---|---|
| `data/01_bronze`, `02_silver`, `03_gold` as partitioned Parquet | OneLake / Lakehouse Delta tables, same three layers |
| `spark.SparkDataset` writing partitioned Parquet | Fabric Spark notebook writing to the Lakehouse |
| `pandas.SQLQueryDataset` against local Postgres | Dataflow Gen2 or Lakehouse shortcut to the source system |
| `kedro run` triggered by launchd | Data Factory pipeline on a schedule |
| `data/08_reporting/*.csv` extracts | Lakehouse SQL analytics endpoint |
| Tableau operational workbook | Power BI semantic model + operations report, DirectLake |
| Tableau executive workbook | Power BI executive report, or Tableau against the SQL endpoint |
| `pytest` in GitHub Actions | Same, plus Fabric deployment pipelines for environment promotion |

**Not claimed:** this was never deployed to Fabric. The table above is a
migration design, not a description of something that ran.
