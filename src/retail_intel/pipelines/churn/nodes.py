"""Churn-risk modelling, with a controlled ablation on review sentiment.

What is actually being predicted
--------------------------------
Olist customers are overwhelmingly one-time buyers. Defining churn as
"stopped a regular buying pattern" would apply to almost nobody, so the label
here is the sharper and more useful one: given everything known about a
customer at the observation cut-off, do they place another order inside the
following 180 days? Churn risk is one minus that probability.

The positive class is small. It is left small. Resampling it to look balanced
would improve the confusion matrix and destroy the calibration, and a
retention budget is allocated off calibrated probabilities, not off accuracy.

The ablation
------------
The question is not "does sentiment help?" -- a text-sentiment score is
obviously correlated with a star rating, and a model with both would show
sentiment mattering when it is really the rating doing the work.

The sharper question, and the one tested here, is: *does reading the review
text add anything beyond the star rating the customer already gave?* So the
baseline arm keeps `avg_review_score` and `min_review_score`, and only the
NLP-derived features are removed. Both arms are otherwise identical: same
rows, same split, same seed, same hyperparameters.

The difference in AUC is then bootstrapped on the test set, because a gap of
a few thousandths between two models is noise, and reporting it as an
improvement would be exactly the kind of false claim this ablation exists to
prevent.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

TARGET = "returned"

# NLP-derived features -- the only things removed in the ablation's baseline arm
SENTIMENT_FEATURES = ["avg_sentiment", "min_sentiment", "n_text_reviews"]

# columns that are identifiers, labels, or leak the outcome
EXCLUDED = {
    "customer_unique_id", "returned", "churned", "observation_cutoff",
    "state", "top_category", "segment", "rfm_cell",
}


def _feature_columns(frame: pd.DataFrame, include_sentiment: bool) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if column not in EXCLUDED and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not include_sentiment:
        columns = [column for column in columns if column not in SENTIMENT_FEATURES]
    return columns


def _fit(train_x, train_y, params: dict) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=0.05,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=params["random_state"],
    )
    model.fit(train_x, train_y)
    return model


def _evaluate(model, test_x, test_y) -> dict:
    probability = model.predict_proba(test_x)[:, 1]
    scores = {
        "roc_auc": round(float(roc_auc_score(test_y, probability)), 4),
        "pr_auc": round(float(average_precision_score(test_y, probability)), 4),
        "brier": round(float(brier_score_loss(test_y, probability)), 5),
        "base_rate": round(float(test_y.mean()), 5),
    }
    # lift in the top decile: the operational question is "if we can only
    # contact 10% of customers, how much better than random is our list?"
    cut = np.quantile(probability, 0.9)
    top = test_y[probability >= cut]
    scores["top_decile_rate"] = round(float(top.mean()), 5)
    scores["top_decile_lift"] = round(float(top.mean() / test_y.mean()), 3)
    return scores, probability


def _bootstrap_auc_delta(
    test_y: np.ndarray,
    probability_with: np.ndarray,
    probability_without: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap the paired AUC difference on the test set.

    Paired resampling -- the same resampled rows scored by both models --
    because the two arms are evaluated on identical data and the variance
    that matters is the variance of the *difference*, not of each AUC alone.
    """
    rng = np.random.default_rng(seed)
    n = len(test_y)
    deltas = np.empty(n_boot)

    for index in range(n_boot):
        sample = rng.integers(0, n, n)
        resampled_y = test_y[sample]
        if resampled_y.sum() in (0, len(resampled_y)):
            deltas[index] = np.nan          # degenerate resample, no AUC defined
            continue
        deltas[index] = roc_auc_score(
            resampled_y, probability_with[sample]
        ) - roc_auc_score(resampled_y, probability_without[sample])

    deltas = deltas[~np.isnan(deltas)]
    lower, upper = np.percentile(deltas, [2.5, 97.5])
    return {
        "mean_delta": round(float(deltas.mean()), 5),
        "ci_95": [round(float(lower), 5), round(float(upper), 5)],
        "share_of_resamples_positive": round(float((deltas > 0).mean()), 4),
        "significant_at_95": bool(lower > 0),
        "n_bootstrap": int(len(deltas)),
    }


def run_sentiment_ablation(gold_rfm_segments: pd.DataFrame, params: dict) -> dict:
    """Train both arms and report whether the text features earn their place."""
    frame = gold_rfm_segments.copy()
    target = frame[TARGET].to_numpy()

    with_columns = _feature_columns(frame, include_sentiment=True)
    without_columns = _feature_columns(frame, include_sentiment=False)
    present_sentiment = [c for c in SENTIMENT_FEATURES if c in with_columns]

    if not present_sentiment:
        raise ValueError(
            "no sentiment features present -- the ablation would compare a model "
            "against itself. Check that the sentiment node ran."
        )

    # one split, reused by both arms, so the comparison is paired
    indices = np.arange(len(frame))
    train_index, test_index = train_test_split(
        indices,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=target,
    )

    results = {}
    probabilities = {}
    models = {}

    for label, columns in (("with_sentiment", with_columns), ("without_sentiment", without_columns)):
        features = frame[columns]
        model = _fit(features.iloc[train_index], target[train_index], params)
        scores, probability = _evaluate(
            model, features.iloc[test_index], target[test_index]
        )
        scores["n_features"] = len(columns)
        results[label] = scores
        probabilities[label] = probability
        models[label] = model
        log.info("%-18s ROC-AUC %.4f | PR-AUC %.4f | lift@10%% %.2f",
                 label, scores["roc_auc"], scores["pr_auc"], scores["top_decile_lift"])

    delta = {
        "roc_auc": round(
            results["with_sentiment"]["roc_auc"] - results["without_sentiment"]["roc_auc"], 5
        ),
        "pr_auc": round(
            results["with_sentiment"]["pr_auc"] - results["without_sentiment"]["pr_auc"], 5
        ),
        "brier": round(
            results["with_sentiment"]["brier"] - results["without_sentiment"]["brier"], 6
        ),
    }

    bootstrap = _bootstrap_auc_delta(
        target[test_index],
        probabilities["with_sentiment"],
        probabilities["without_sentiment"],
        seed=params["random_state"],
    )

    # permutation importance of the sentiment features specifically, as a
    # second, independent line of evidence
    features_with = frame[with_columns]
    importance = permutation_importance(
        models["with_sentiment"],
        features_with.iloc[test_index],
        target[test_index],
        n_repeats=10,
        random_state=params["random_state"],
        scoring="roc_auc",
    )
    importance_by_feature = {
        column: round(float(importance.importances_mean[position]), 5)
        for position, column in enumerate(with_columns)
    }
    ranked = sorted(importance_by_feature.items(), key=lambda kv: -kv[1])

    verdict = (
        "sentiment features materially improve the model"
        if bootstrap["significant_at_95"]
        else "sentiment features do NOT show a statistically reliable improvement"
    )
    log.info("ablation verdict: %s (dAUC %.5f, 95%% CI %s)",
             verdict, delta["roc_auc"], bootstrap["ci_95"])

    return {
        "question": (
            "Does NLP sentiment extracted from review text add predictive power "
            "beyond the star rating the same customer already gave?"
        ),
        "design": {
            "arms": ["with_sentiment", "without_sentiment"],
            "features_removed_in_baseline": present_sentiment,
            "features_retained_in_both": ["avg_review_score", "min_review_score"],
            "paired": True,
            "split": f"stratified hold-out, test_size={params['test_size']}",
            "seed": params["random_state"],
        },
        "results": results,
        "delta": delta,
        "bootstrap_auc_delta": bootstrap,
        "verdict": verdict,
        "permutation_importance_top_15": dict(ranked[:15]),
        "permutation_importance_sentiment_features": {
            column: importance_by_feature[column] for column in present_sentiment
        },
    }


def train_churn_model(gold_rfm_segments: pd.DataFrame, churn_ablation: dict, params: dict):
    """Fit the production churn model on the full cohort.

    The ablation decides which feature set ships. If the text features did not
    clear the significance bar, the simpler model is the one that goes to
    production -- carrying an NLP dependency that does not earn its keep is a
    maintenance cost with no return.
    """
    use_sentiment = churn_ablation["bootstrap_auc_delta"]["significant_at_95"]
    columns = _feature_columns(gold_rfm_segments, include_sentiment=use_sentiment)

    log.info(
        "training production churn model %s sentiment features (%d features)",
        "WITH" if use_sentiment else "WITHOUT",
        len(columns),
    )
    model = _fit(gold_rfm_segments[columns], gold_rfm_segments[TARGET], params)
    return {"model": model, "features": columns, "uses_sentiment": use_sentiment}


def score_customers(churn_model: dict, gold_rfm_segments: pd.DataFrame) -> pd.DataFrame:
    """Attach churn risk and a risk band to every customer."""
    frame = gold_rfm_segments.copy()
    probability_return = churn_model["model"].predict_proba(
        frame[churn_model["features"]]
    )[:, 1]

    frame["p_return"] = probability_return
    frame["churn_risk"] = 1 - probability_return

    # Absolute churn risk is ~99% for almost everyone, because only ~1% of
    # customers return at all. Banding on fixed thresholds therefore puts the
    # entire base in one bucket and tells a commercial leader nothing.
    #
    # The decision-useful quantity is *relative* risk: who is unusually likely
    # to churn compared with the population, and who is unusually likely to
    # come back. Quantile bands surface that ordering; the absolute probability
    # is kept alongside so nobody mistakes "Low" for "safe".
    frame["relative_churn_risk"] = frame["churn_risk"] / frame["churn_risk"].mean()
    frame["risk_band"] = pd.qcut(
        frame["churn_risk"].rank(method="first"),
        q=[0, 0.5, 0.8, 0.95, 1.0],
        labels=["Low", "Moderate", "High", "Critical"],
    )

    # Two complementary money figures:
    #   revenue_at_risk        -- exposure if nothing is done
    #   expected_retained      -- what the base is worth on current trajectory
    # The second is what a retention programme is trying to move, and it
    # discriminates between customers where the first does not.
    frame["revenue_at_risk"] = frame["churn_risk"] * frame["monetary"]
    frame["expected_retained_revenue"] = frame["p_return"] * frame["monetary"]

    summary = frame.groupby("risk_band", observed=True).agg(
        customers=("customer_unique_id", "count"),
        revenue_at_risk=("revenue_at_risk", "sum"),
        expected_retained=("expected_retained_revenue", "sum"),
        # the check that the bands mean anything: actual return rate must fall
        # monotonically from Low risk to Critical
        actual_return_rate=("returned", "mean"),
    )
    log.info("churn scoring:\n%s", summary.to_string())

    return frame[
        [
            "customer_unique_id", "state", "segment", "rfm_score",
            "recency_days", "frequency", "monetary", "avg_order_value",
            "avg_review_score", "avg_sentiment", "any_late_delivery",
            "p_return", "churn_risk", "relative_churn_risk", "risk_band",
            "revenue_at_risk", "expected_retained_revenue", "returned",
        ]
    ]
