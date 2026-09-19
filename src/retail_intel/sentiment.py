"""Sentiment scoring for Brazilian-Portuguese review text.

The problem this module solves
------------------------------
The brief allows "VADER or a lightweight transformer". VADER is the obvious
default -- and on this dataset it is the wrong tool, because every review is
written in Brazilian Portuguese and VADER's lexicon is English. Applied to
"Produto pessimo, nunca chegou", VADER scores 0.0: it recognises none of the
words. Running it anyway would produce a feature that is almost entirely
zeros, an ablation that shows no effect, and a conclusion ("sentiment doesn't
help predict churn") that is an artefact of the tooling rather than a finding
about customers.

So this module implements a Portuguese lexicon scorer, and keeps VADER
alongside it as an explicit baseline. Both are validated against star rating
in `validate_against_rating`, which is the part that matters: star rating is
independent ground truth for review sentiment, so the correlation between a
scorer's output and the stars says directly whether that scorer is reading
the text or hallucinating.

Design follows VADER's structure -- lexicon, negation window, intensifiers,
and a bounded compound score -- because that structure is sound; only the
language is different.
"""
from __future__ import annotations

import logging
import math
import re
import unicodedata
from functools import lru_cache

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Lexicon. Polarity is on [-1, 1]. Weights reflect intensity, so "péssimo"
# is stronger than "ruim" and "adorei" is stronger than "gostei".
# Vocabulary is drawn from the terms that actually occur in Olist review
# text: delivery timing, product condition, and seller conduct dominate.
# ---------------------------------------------------------------------
_POSITIVE = {
    "otimo": 0.85, "otima": 0.85, "excelente": 0.90, "excelentes": 0.90,
    "bom": 0.60, "boa": 0.60, "bons": 0.60, "boas": 0.60,
    "perfeito": 0.90, "perfeita": 0.90, "perfeitamente": 0.85,
    "maravilhoso": 0.92, "maravilhosa": 0.92, "maravilha": 0.90,
    "adorei": 0.88, "amei": 0.90, "amo": 0.85, "gostei": 0.65, "gostou": 0.60,
    "recomendo": 0.80, "recomendado": 0.75, "recomendadissimo": 0.90,
    "rapido": 0.65, "rapida": 0.65, "rapidez": 0.70, "rapidamente": 0.65,
    "agil": 0.60, "agilidade": 0.65, "eficiente": 0.65, "eficaz": 0.60,
    "satisfeito": 0.70, "satisfeita": 0.70, "satisfacao": 0.70,
    "qualidade": 0.55, "durabilidade": 0.50, "resistente": 0.45,
    "lindo": 0.70, "linda": 0.70, "bonito": 0.60, "bonita": 0.60,
    "top": 0.70, "show": 0.65, "massa": 0.55, "legal": 0.50,
    "parabens": 0.75, "obrigado": 0.45, "obrigada": 0.45, "agradeco": 0.50,
    "confiavel": 0.65, "seguro": 0.45, "honesto": 0.60,
    "super": 0.35, "melhor": 0.60, "otimos": 0.85,
    "cumpriu": 0.55, "cumprido": 0.55, "conforme": 0.45, "certinho": 0.65,
    "correto": 0.55, "corretamente": 0.55, "impecavel": 0.85,
    "chegou": 0.30, "entregue": 0.35, "recebi": 0.25, "recebido": 0.25,
    "aprovado": 0.65, "aprovei": 0.65, "indico": 0.75, "indicaria": 0.70,
    "atendimento": 0.20, "atencioso": 0.65, "educado": 0.55,
    "barato": 0.45, "economico": 0.40, "vale": 0.35, "valeu": 0.55,
    "funcionou": 0.55, "funciona": 0.50, "util": 0.45,
    "adequado": 0.45, "recomendavel": 0.70, "sucesso": 0.60,
    "feliz": 0.70, "contente": 0.65, "encantada": 0.80, "encantado": 0.80,
    "surpreendeu": 0.60, "superou": 0.70, "fantastico": 0.88, "incrivel": 0.85,
    "otimamente": 0.80, "belissimo": 0.85, "sensacional": 0.88,
    "pontual": 0.65, "antecipada": 0.60, "antecipado": 0.60, "adiantado": 0.55,
    "bem": 0.30, "confortavel": 0.55, "pratico": 0.50, "facil": 0.45,
}

_NEGATIVE = {
    "pessimo": -0.92, "pessima": -0.92, "pessimos": -0.92,
    "ruim": -0.70, "ruins": -0.70, "pior": -0.75, "piores": -0.75,
    "horrivel": -0.90, "horriveis": -0.90, "terrivel": -0.88,
    "lixo": -0.90, "porcaria": -0.85, "merda": -0.90,
    "decepcao": -0.82, "decepcionado": -0.80, "decepcionada": -0.80,
    "decepcionante": -0.80, "frustrado": -0.75, "frustrante": -0.75,
    "arrependido": -0.78, "arrependi": -0.78, "arrependimento": -0.78,
    "atrasado": -0.65, "atrasada": -0.65, "atraso": -0.70, "atrasou": -0.70,
    "demorou": -0.60, "demora": -0.60, "demorado": -0.60, "demorando": -0.62,
    "quebrado": -0.80, "quebrada": -0.80, "quebrou": -0.80, "rachado": -0.75,
    "danificado": -0.80, "avariado": -0.78, "amassado": -0.65,
    "defeito": -0.80, "defeituoso": -0.82, "estragado": -0.80,
    "errado": -0.70, "errada": -0.70, "erro": -0.60, "trocado": -0.60,
    "falta": -0.55, "faltou": -0.65, "faltando": -0.65, "incompleto": -0.65,
    "cancelar": -0.55, "cancelado": -0.60, "cancelamento": -0.58,
    "devolver": -0.65, "devolucao": -0.65, "devolvi": -0.65,
    "estorno": -0.55, "reembolso": -0.50, "reclamacao": -0.70,
    "enganado": -0.82, "enganosa": -0.82, "enganoso": -0.82, "golpe": -0.90,
    "fraude": -0.92, "mentira": -0.80, "falso": -0.78, "falsificado": -0.85,
    "problema": -0.60, "problemas": -0.62, "dificuldade": -0.50,
    "fraco": -0.55, "fragil": -0.50, "frageis": -0.50,
    "sujo": -0.60, "velho": -0.45, "usado": -0.45,
    "caro": -0.40, "abusivo": -0.65, "absurdo": -0.75,
    "descaso": -0.80, "descaso": -0.80, "abandonado": -0.65,
    "ninguem": -0.35, "nada": -0.40, "nenhum": -0.40, "nenhuma": -0.40,
    "insatisfeito": -0.75, "insatisfeita": -0.75, "inaceitavel": -0.85,
    "pessimamente": -0.85, "horrorosa": -0.85, "horroroso": -0.85,
    "lamentavel": -0.78, "vergonha": -0.80, "vergonhoso": -0.82,
    "perdi": -0.55, "perdido": -0.60, "extraviado": -0.80, "extravio": -0.80,
    "sumiu": -0.70, "roubado": -0.85, "furtado": -0.85,
    "pequeno": -0.30, "menor": -0.30, "diferente": -0.35,
    "insuficiente": -0.55, "ineficiente": -0.65, "despreparado": -0.60,
    "mal": -0.50, "pena": -0.45, "infelizmente": -0.55,
    "impossivel": -0.50, "ignorado": -0.70, "ignorou": -0.70,
}

LEXICON: dict[str, float] = {**_POSITIVE, **_NEGATIVE}

# Multi-word expressions are matched before tokenisation because their meaning
# is not the sum of their parts: "propaganda enganosa" is strongly negative,
# "antes do prazo" strongly positive, and "nao recomendo" inverts a positive.
PHRASES: dict[str, float] = {
    "nao recomendo": -0.85,
    "nao recebi": -0.88,
    "nao chegou": -0.85,
    "nao entregue": -0.85,
    "nao funciona": -0.80,
    "nao gostei": -0.75,
    "nao veio": -0.80,
    "nao comprem": -0.88,
    "nunca mais": -0.80,
    "propaganda enganosa": -0.88,
    "antes do prazo": 0.75,
    "dentro do prazo": 0.60,
    "no prazo": 0.55,
    "super recomendo": 0.90,
    "muito bom": 0.78,
    "muito boa": 0.78,
    "muito ruim": -0.80,
    "otimo produto": 0.85,
    "recomendo a todos": 0.85,
    "chegou rapido": 0.75,
    "chegou antes": 0.72,
    "entrega rapida": 0.75,
    "produto de qualidade": 0.75,
    "atendeu as expectativas": 0.70,
    "acima das expectativas": 0.85,
    "deixou a desejar": -0.70,
    "fora do prazo": -0.70,
    "ate agora": -0.30,
}

NEGATORS = {"nao", "nunca", "jamais", "nem", "sem", "nenhum", "nenhuma"}
INTENSIFIERS = {
    "muito": 1.45, "super": 1.40, "bastante": 1.30, "extremamente": 1.60,
    "demais": 1.35, "totalmente": 1.35, "completamente": 1.35,
    "realmente": 1.20, "bem": 1.15, "mega": 1.40, "ultra": 1.40,
    "absurdamente": 1.55, "tao": 1.20, "tanto": 1.20,
}
DIMINISHERS = {"pouco": 0.55, "meio": 0.65, "levemente": 0.6, "razoavelmente": 0.75}

NEGATION_WINDOW = 3
_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Normalisation constant borrowed from VADER: squashes the raw sum into
# (-1, 1) while keeping short, emphatic reviews from saturating instantly.
_ALPHA = 15.0


@lru_cache(maxsize=200_000)
def normalise(text: str) -> str:
    """Lowercase and strip accents.

    Review text is user-typed and inconsistently accented -- "ótimo", "otimo"
    and "OTIMO" all appear. Folding accents means one lexicon entry covers all
    three, rather than needing every spelling variant.
    """
    lowered = text.lower()
    decomposed = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def score_text(text: str | None) -> float:
    """Return a compound sentiment score in [-1, 1] for one review body."""
    if not isinstance(text, str) or not text.strip():
        return 0.0

    cleaned = normalise(text)

    total = 0.0
    # phrases first; each match is consumed so its tokens cannot double-count
    for phrase, weight in PHRASES.items():
        occurrences = cleaned.count(phrase)
        if occurrences:
            total += weight * occurrences
            cleaned = cleaned.replace(phrase, " ")

    tokens = _TOKEN_RE.findall(cleaned)
    for index, token in enumerate(tokens):
        polarity = LEXICON.get(token)
        if polarity is None:
            continue

        # intensity modifier from the immediately preceding token
        if index > 0:
            previous = tokens[index - 1]
            polarity *= INTENSIFIERS.get(previous, DIMINISHERS.get(previous, 1.0))

        # negation anywhere in the preceding window flips the sign and damps it
        window = tokens[max(0, index - NEGATION_WINDOW) : index]
        if any(word in NEGATORS for word in window):
            polarity *= -0.75

        total += polarity

    return round(total / math.sqrt(total * total + _ALPHA), 4)


def score_frame(frame: pd.DataFrame, text_column: str = "message") -> pd.Series:
    """Vectorised-enough scoring over a column of review bodies."""
    return frame[text_column].map(score_text).astype("float32")


# ---------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------
def score_with_vader(frame: pd.DataFrame, text_column: str = "message") -> pd.Series:
    """Score the same text with English VADER, as a documented baseline."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()

    def _score(text: object) -> float:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        return analyzer.polarity_scores(text)["compound"]

    return frame[text_column].map(_score).astype("float32")


def validate_against_rating(
    frame: pd.DataFrame, score_columns: list[str], rating_column: str = "review_score"
) -> dict:
    """Check each scorer against the star rating the customer also left.

    The star rating is independent of the text, so correlation with it is a
    fair test of whether a scorer is actually reading the review. Spearman is
    used because the rating is ordinal and the relationship need only be
    monotonic.

    `zero_rate` is reported alongside: a scorer that returns 0.0 for most
    reviews is not neutral about them, it simply does not recognise the words.
    """
    scored = frame.loc[frame[rating_column].notna()]
    results = {}

    for column in score_columns:
        series = scored[column]
        results[column] = {
            "spearman_vs_rating": round(
                float(series.corr(scored[rating_column], method="spearman")), 4
            ),
            "pearson_vs_rating": round(
                float(series.corr(scored[rating_column], method="pearson")), 4
            ),
            "zero_rate": round(float((series == 0).mean()), 4),
            "mean_score_1_star": round(
                float(series[scored[rating_column] == 1].mean()), 4
            ),
            "mean_score_5_star": round(
                float(series[scored[rating_column] == 5].mean()), 4
            ),
        }
        # a scorer that cannot separate 1-star from 5-star reviews is not a
        # sentiment feature, whatever its correlation coefficient says
        separation = (
            results[column]["mean_score_5_star"] - results[column]["mean_score_1_star"]
        )
        results[column]["one_to_five_separation"] = round(separation, 4)

    return results
