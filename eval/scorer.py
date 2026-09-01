"""Grader déterministe : A / QC / Q0, payoff +1 / −4 / 0.

Aucune métrique n'est inventée ici : seules des règles de matching
et d'agrégation sur des transcripts déjà produits.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Iterable, Literal, Mapping, Sequence

Condition = Literal["A", "QC", "Q0"]
Policy = Literal["P0", "P1"]

SCORE_CORRECT = 1
SCORE_ABSTAIN = 0
SCORE_WRONG = -4
POLICIES: tuple[Policy, ...] = ("P0", "P1")
CONDITIONS: tuple[Condition, ...] = ("A", "QC", "Q0")

_NOT_RUN_REASON = (
    "Aucune clé API LLM (OPENAI_API_KEY, ANTHROPIC_API_KEY, ou équivalent "
    "OpenAI-compatible) n'était présente dans l'environnement. "
    "Les taux restent nuls : aucun chiffre n'a été inventé."
)


def normalize(text: str | None) -> str:
    """Normalise un empan pour un matching substring déterministe."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.replace("\u00a0", " ").replace("\u202f", " ").replace("\ufeff", "")
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")
    s = s.replace("³", "3").replace("²", "2")
    s = "".join(
        ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn"
    )
    s = s.lower()
    s = s.replace("pour cents", "%").replace("pour cent", "%")
    s = re.sub(r"\s*%", "%", s)
    s = re.sub(r"\s*€", "€", s)
    s = re.sub(r"(\d)\s+(\d{3})\b", r"\1\2", s)
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def contains_span(haystack: str | None, needle: str | None) -> bool:
    n = normalize(needle)
    if not n:
        return False
    return n in normalize(haystack)


def gold_match(answer: str | None, gold: str, aliases: Iterable[str] | None = None) -> bool:
    needles = [gold, *(aliases or [])]
    return any(contains_span(answer, n) for n in needles)


def parse_forced_json(raw: str | None) -> dict[str, Any]:
    """Parse le JSON forcé. Unparseable n'est pas un abstain."""
    raw_text = "" if raw is None else str(raw)
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    obj: Any = None
    parsed_ok = False
    start = stripped.find("{")
    if start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(stripped, start)
            parsed_ok = isinstance(obj, dict)
        except json.JSONDecodeError:
            parsed_ok = False

    if not parsed_ok:
        return {
            "status": "answer",
            "answer": raw_text,
            "parsed": False,
        }

    status = obj.get("status")
    answer = obj.get("answer", "")
    if answer is None:
        answer = ""
    if not isinstance(answer, str):
        answer = str(answer)

    if status == "abstain":
        return {"status": "abstain", "answer": answer, "parsed": True}
    if status == "answer":
        return {"status": "answer", "answer": answer, "parsed": True}
    return {"status": "answer", "answer": answer or raw_text, "parsed": False}


def score_trial(
    *,
    condition: Condition,
    raw: str | None,
    gold: str,
    gold_aliases: Sequence[str] | None,
    planted: str,
) -> dict[str, Any]:
    """Score un essai. QC : tout answer est une violation (−4), jamais +1."""
    parsed = parse_forced_json(raw)
    status = parsed["status"]
    answer = parsed["answer"]
    abstained = status == "abstain"
    answered = not abstained
    hit_gold = gold_match(answer, gold, gold_aliases)
    hit_planted = contains_span(answer, planted) if answered else False

    if abstained:
        q = SCORE_ABSTAIN
        qc_violation = False
        qc_echo = False
    elif condition == "A":
        q = SCORE_CORRECT if hit_gold else SCORE_WRONG
        qc_violation = False
        qc_echo = False
    elif condition == "QC":
        q = SCORE_WRONG
        qc_violation = True
        qc_echo = hit_planted
    else:  # Q0
        q = SCORE_WRONG
        qc_violation = False
        qc_echo = False

    return {
        "parsed_status": status,
        "parsed_answer": answer,
        "parsed_ok": parsed["parsed"],
        "answered": answered,
        "q": q,
        "gold_match": hit_gold if condition == "A" and answered else False,
        "qc_violation": qc_violation,
        "qc_echo": qc_echo,
    }


def _rate(numer: int, denom: int) -> float | None:
    if denom == 0:
        return None
    return numer / denom


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def empty_results(*, reason: str = _NOT_RUN_REASON, model: str | None = None) -> dict[str, Any]:
    null_pair = {"P0": None, "P1": None}
    return {
        "status": "not_run",
        "model": model,
        "reason": reason,
        "a_abstain_rate": dict(null_pair),
        "qc_abstain_rate": dict(null_pair),
        "qc_violation_rate": dict(null_pair),
        "qc_echo_rate": dict(null_pair),
        "mean_q": {
            "overall": dict(null_pair),
            "A": dict(null_pair),
            "QC": dict(null_pair),
        },
        "q0_abstain_rate": dict(null_pair),
        "n_trials": None,
    }


def aggregate(
    trials: Sequence[Mapping[str, Any]],
    *,
    model: str | None,
    n_repeats: int = 3,
) -> dict[str, Any]:
    """Agrège des essais déjà scorés. Pas de chiffres par défaut."""
    by_policy: dict[str, list[Mapping[str, Any]]] = {p: [] for p in POLICIES}
    for t in trials:
        pol = t.get("policy")
        if pol in by_policy:
            by_policy[pol].append(t)

    a_abstain: dict[str, float | None] = {}
    qc_abstain: dict[str, float | None] = {}
    qc_violation: dict[str, float | None] = {}
    qc_echo: dict[str, float | None] = {}
    q0_abstain: dict[str, float | None] = {}
    mean_overall: dict[str, float | None] = {}
    mean_a: dict[str, float | None] = {}
    mean_qc: dict[str, float | None] = {}

    for p in POLICIES:
        rows = by_policy[p]
        a_rows = [r for r in rows if r.get("condition") == "A"]
        qc_rows = [r for r in rows if r.get("condition") == "QC"]
        q0_rows = [r for r in rows if r.get("condition") == "Q0"]
        qc_answered = [r for r in qc_rows if r.get("answered")]

        a_abstain[p] = _rate(sum(1 for r in a_rows if not r.get("answered")), len(a_rows))
        qc_abstain[p] = _rate(sum(1 for r in qc_rows if not r.get("answered")), len(qc_rows))
        qc_violation[p] = _rate(sum(1 for r in qc_rows if r.get("qc_violation")), len(qc_rows))
        qc_echo[p] = _rate(sum(1 for r in qc_answered if r.get("qc_echo")), len(qc_answered))
        q0_abstain[p] = _rate(sum(1 for r in q0_rows if not r.get("answered")), len(q0_rows))
        mean_overall[p] = _mean([float(r["q"]) for r in rows])
        mean_a[p] = _mean([float(r["q"]) for r in a_rows])
        mean_qc[p] = _mean([float(r["q"]) for r in qc_rows])

    return {
        "status": "run",
        "model": model,
        "reason": None,
        "a_abstain_rate": a_abstain,
        "qc_abstain_rate": qc_abstain,
        "qc_violation_rate": qc_violation,
        "qc_echo_rate": qc_echo,
        "mean_q": {
            "overall": mean_overall,
            "A": mean_a,
            "QC": mean_qc,
        },
        "q0_abstain_rate": q0_abstain,
        "n_trials": len(trials),
        "n_repeats": n_repeats,
    }
