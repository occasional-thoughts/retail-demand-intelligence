"""Sentiment scorer behaviour.

These test the properties that matter for the downstream feature, not exact
score values -- pinning a float would make the lexicon impossible to tune.
"""
import pytest

from retail_intel.sentiment import PHRASES, normalise, score_text


def test_empty_and_missing_text_score_zero():
    assert score_text(None) == 0.0
    assert score_text("") == 0.0
    assert score_text("    ") == 0.0


def test_accents_are_folded():
    # the same word typed with and without accents must score identically
    assert score_text("otimo produto") == score_text("ótimo produto")
    assert normalise("PÉSSIMO") == "pessimo"


def test_positive_and_negative_reviews_separate():
    positive = score_text("Produto de ótima qualidade, entrega super rápida. Recomendo!")
    negative = score_text("Péssimo, produto quebrado e chegou muito atrasado.")
    assert positive > 0.3
    assert negative < -0.3
    assert positive > negative


def test_negation_flips_polarity():
    # "recomendo" is strongly positive; "não recomendo" must not be
    assert score_text("recomendo") > 0
    assert score_text("nao recomendo") < 0


def test_intensifier_amplifies():
    assert score_text("muito bom") > score_text("bom")


def test_diminisher_damps():
    assert score_text("pouco bom") < score_text("bom")


def test_score_is_bounded():
    # even a review stuffed with superlatives cannot leave [-1, 1]
    extreme = " ".join(["excelente maravilhoso perfeito otimo"] * 50)
    assert -1.0 <= score_text(extreme) <= 1.0
    assert -1.0 <= score_text("pessimo horrivel lixo " * 50) <= 1.0


@pytest.mark.parametrize("phrase", list(PHRASES))
def test_every_phrase_scores_in_its_stated_direction(phrase):
    expected = PHRASES[phrase]
    actual = score_text(phrase)
    if expected > 0:
        assert actual > 0, f"{phrase!r} is listed positive but scored {actual}"
    else:
        assert actual < 0, f"{phrase!r} is listed negative but scored {actual}"
