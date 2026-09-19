# Insight memo — Retail Demand & Customer Intelligence

**To:** Commercial and Operations leadership
**From:** Capabilities & Insights Analytics
**Re:** Three actions from the demand-forecast and churn models
**Date:** 16 September 2026 · **Basis:** 99,441 orders, R$15.7M revenue, 23 complete months

---

## Bottom line

Three findings, each actionable this quarter:

1. **Peak demand is a one-week event, not a season.** Our busiest week ran **3.5× a normal week**, while the three-week Black Friday window averaged only **+18%**. Any buffer sized on the window understates the peak by roughly threefold.
2. **Our stock buffer is paying for supplier unpredictability, not supplier slowness.** **61%** of required safety stock is driven by lead-time *variability*. Bringing our worst lanes to the consistency our best lanes already achieve would cut safety stock **49%** — without a single day of speed improvement.
3. **Retention spend should be concentrated, and the NLP programme should not be funded.** The top-scoring **50% of customers capture 91% of all repeat purchasers**. Separately, review-text sentiment **failed** a controlled test of whether it adds anything beyond the star rating customers already give us.

---

## 1. Size peak buffers to the peak week, not the peak season

**Recommendation.** Set Q4 safety stock against the single-week peak. For the 11 categories currently below reorder point, release the **1,265-unit** replenishment now; **office_furniture** is already inside its lead time and must be ordered today.

**Evidence.** Week of 21 Nov 2017 hit **3,703 units** against a non-peak weekly mean of **1,058** — a **3.5× spike**. The surrounding three-week window averaged only **+18%**, because the spike is concentrated in days, not weeks. Today, **11 of 37 forecastable categories** are below reorder point, carrying a projected **2,155-unit shortfall** over the next 8 weeks — **54% of those categories' own forecast demand**, and **26% of total forecast volume**.

**Expected impact.** Closing the flagged gap addresses the full 2,155-unit projected shortfall. Applying peak-week rather than peak-window sizing raises the Q4 buffer on affected categories roughly threefold against the true spike.

**Confidence.** *High on the demand pattern* — it is directly observed in two years of order data. *Medium on the stock figures* — opening stock is simulated (see Basis).

---

## 2. Attack lead-time variability, not lead-time length

**Recommendation.** Put the **231 high-risk supplier lanes** on a delivery-consistency SLA governed by 95th-percentile transit time, not average transit time. Prioritise the five São Paulo lanes carrying the largest revenue.

**Evidence.** High-risk and low-risk lanes differ by only **1.8 days at the mean** (12.2 vs 10.4) — but by **14.4 days at the 95th percentile** (29.3 vs 14.9). Safety stock responds to the tail, not the mean: decomposing our own formula, **61% of required buffer (2,782 of 4,568 units) comes from the lead-time variance term**, not from demand variance. The 231 high-risk lanes carry **R$14.3M — 91% of mapped revenue**.

This also has a service consequence. Late-delivery rate correlates with monthly order volume at **r = 0.55**: our five busiest months ran **10.5% late against 5.5% in all others**. Fulfilment degrades precisely when volume arrives — which is the same week finding as Recommendation 1.

**Expected impact.** Bringing high-risk lanes to the coefficient of variation our low-risk lanes already achieve (0.27) would reduce required safety stock from **4,568 to 2,336 units — a 49% reduction** in working capital tied up in buffer, at unchanged service level.

**Confidence.** *High.* Lead times are derived from 2.7M actual delivery scan events spanning 110,181 order lines; the variability gap is measured, not assumed.

---

## 3. Concentrate retention on the top half — and do not fund the sentiment programme

**Recommendation.** Target retention at the **29,047 customers in the top-scoring half** of the base. Separately: **do not productionise review-text NLP** as a churn input on current evidence.

**Evidence.** Only **1.14%** of customers return within 180 days, so untargeted retention spend is almost entirely wasted. The model ranks usefully despite that: the top-scoring half returns at **2.08% (1.82× base) and contains 91% of all 664 repeat purchasers**; the bottom-scoring quarter returns at **0.0%**.

On sentiment, we ran a controlled ablation asking the sharp question — does reading the review *text* add anything beyond the star rating the same customer already gave? Two models, identical rows, split, seed and hyperparameters, differing only in the NLP features:

| | ROC-AUC | Lift @ top decile |
|---|---|---|
| With sentiment | 0.567 | 1.99 |
| Without sentiment | **0.577** | **2.05** |

Sentiment made the model **worse** (ΔAUC −0.0096). A paired bootstrap over 1,000 resamples puts the 95% interval at **[−0.032, +0.013]** — spanning zero, so the difference is not distinguishable from noise in either direction. The simpler model ships.

Sentiment still earns its place as a *diagnostic*: monthly sentiment correlates with late-delivery rate at **r = −0.42**, which says our review scores are substantially a logistics outcome, not a product one. That routes the remediation budget to Recommendation 2.

**Expected impact.** Concentrating spend on the top half reaches 91% of realistic returners at 50% of the contact cost. Declining the NLP build avoids an ongoing pipeline dependency with no measured return.

**Confidence.** *High on the negative result* — it is what the ablation was designed to detect, and the bootstrap quantifies it. *Medium on the targeting* — ROC-AUC 0.577 is a weak model in absolute terms; it ranks well enough to prioritise a list, not well enough to make individual decisions.

---

## What we would need to go further

- **A stock ledger.** Every inventory figure here rests on a simulated opening position. Real on-hand data would convert these from directional to operational overnight; nothing else in the pipeline would change.
- **A third year of history.** At 23 months the forecast beats a naive baseline by 6.2% — real, but thin. Most of the remaining error is seasonal structure the model has seen only once.
- **Real clickstream.** Browsing behaviour is currently synthetic, so it contributes no genuine signal to the churn model. Live session data is the most likely source of the lift that review sentiment failed to provide.

## Basis of these numbers

Olist Brazilian e-commerce dataset: 99,441 orders across 74 categories, Sep 2016 – Aug 2018. Demand forecast validated by expanding-window walk-forward backtesting (5 folds, 8-week horizon): **WAPE 0.328 vs 0.350 for a naive baseline and 0.849 for seasonal naive** — the model beats naive at every one of the 8 horizons. Churn measured against a 180-day outcome window with features cut off at the observation date.

**Simulated inputs, stated plainly.** Olist publishes no stock ledger, supplier contracts or browsing data. Opening stock is generated from a documented policy (6 weeks cover, 95% service level) and browsing sessions are synthetic. Both are flagged as simulated wherever they appear. **Forecast demand, lead times, delivery performance, revenue, review scores and sentiment are all derived from real order data.**

**One caveat on the forecast.** Model hyperparameters were selected by comparing candidates on the same walk-forward folds used to report the result, so the 6.2% improvement is mildly optimistic. An unbiased figure needs a held-out period excluded from selection.
