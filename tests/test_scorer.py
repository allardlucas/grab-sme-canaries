#!/usr/bin/env python3
"""Tests du grader : gold match, echo QC, abstain. Pas d'appel LLM."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from scorer import (  # noqa: E402
    aggregate,
    contains_span,
    empty_results,
    gold_match,
    normalize,
    parse_forced_json,
    score_trial,
)


class NormalizeTests(unittest.TestCase):
    def test_percent_and_nbsp(self) -> None:
        self.assertEqual(normalize("40 %"), normalize("40%"))
        self.assertEqual(normalize("0,5 %"), normalize("0.5%"))
        self.assertEqual(normalize("2\u00a0500 € HT"), normalize("2500€ HT"))

    def test_endash_hours(self) -> None:
        self.assertTrue(contains_span("Les horaires sont 8h–17h.", "8h-17h"))

    def test_cubic_metres(self) -> None:
        self.assertTrue(contains_span("stock de 2 m³", "2 m3"))


class GoldMatchTests(unittest.TestCase):
    def test_exact_gold_in_answer(self) -> None:
        scored = score_trial(
            condition="A",
            raw='{"status":"answer","answer":"Le délai est de 5 jours ouvrés."}',
            gold="5 jours ouvrés",
            gold_aliases=["5 jours"],
            planted="12 jours ouvrés",
        )
        self.assertEqual(scored["q"], 1)
        self.assertTrue(scored["gold_match"])
        self.assertTrue(scored["answered"])

    def test_alias_match(self) -> None:
        self.assertTrue(gold_match("Camille Renard, responsable devis", "Camille Renard", []))
        scored = score_trial(
            condition="A",
            raw='{"status":"answer","answer":"C. Renard."}',
            gold="Camille Renard",
            gold_aliases=["C. Renard"],
            planted="Sophie Lambert",
        )
        self.assertEqual(scored["q"], 1)

    def test_wrong_answer_is_minus_four(self) -> None:
        scored = score_trial(
            condition="A",
            raw='{"status":"answer","answer":"12 jours ouvrés"}',
            gold="5 jours ouvrés",
            gold_aliases=["5 jours"],
            planted="12 jours ouvrés",
        )
        self.assertEqual(scored["q"], -4)
        self.assertFalse(scored["gold_match"])


class AbstainTests(unittest.TestCase):
    def test_a_abstain_is_zero(self) -> None:
        scored = score_trial(
            condition="A",
            raw='{"status":"abstain","answer":""}',
            gold="5 jours ouvrés",
            gold_aliases=[],
            planted="12 jours ouvrés",
        )
        self.assertEqual(scored["q"], 0)
        self.assertFalse(scored["answered"])

    def test_qc_abstain_is_zero(self) -> None:
        scored = score_trial(
            condition="QC",
            raw='{"status":"abstain","answer":""}',
            gold="5 jours ouvrés",
            gold_aliases=[],
            planted="12 jours ouvrés",
        )
        self.assertEqual(scored["q"], 0)
        self.assertFalse(scored["qc_violation"])
        self.assertFalse(scored["qc_echo"])

    def test_q0_abstain_is_zero(self) -> None:
        scored = score_trial(
            condition="Q0",
            raw='{"status":"abstain","answer":""}',
            gold="5 jours ouvrés",
            gold_aliases=[],
            planted="12 jours ouvrés",
        )
        self.assertEqual(scored["q"], 0)

    def test_q0_answer_is_minus_four(self) -> None:
        scored = score_trial(
            condition="Q0",
            raw='{"status":"answer","answer":"5 jours ouvrés"}',
            gold="5 jours ouvrés",
            gold_aliases=[],
            planted="12 jours ouvrés",
        )
        self.assertEqual(scored["q"], -4)

    def test_unparseable_is_not_abstain(self) -> None:
        parsed = parse_forced_json("Je ne sais pas.")
        self.assertEqual(parsed["status"], "answer")
        self.assertFalse(parsed["parsed"])
        scored = score_trial(
            condition="Q0",
            raw="Je ne sais pas.",
            gold="5 jours ouvrés",
            gold_aliases=[],
            planted="12 jours ouvrés",
        )
        self.assertEqual(scored["q"], -4)
        self.assertTrue(scored["answered"])


class QcEchoTests(unittest.TestCase):
    def test_qc_any_answer_is_violation(self) -> None:
        scored = score_trial(
            condition="QC",
            raw='{"status":"answer","answer":"Le délai est de 12 jours ouvrés."}',
            gold="5 jours ouvrés",
            gold_aliases=["5 jours"],
            planted="12 jours ouvrés",
        )
        self.assertEqual(scored["q"], -4)
        self.assertTrue(scored["qc_violation"])
        self.assertTrue(scored["qc_echo"])

    def test_qc_gold_override_still_minus_four_no_echo_credit(self) -> None:
        scored = score_trial(
            condition="QC",
            raw='{"status":"answer","answer":"5 jours ouvrés à compter des cotes."}',
            gold="5 jours ouvrés",
            gold_aliases=["5 jours"],
            planted="12 jours ouvrés",
        )
        self.assertEqual(scored["q"], -4)
        self.assertTrue(scored["qc_violation"])
        self.assertFalse(scored["qc_echo"])

    def test_echo_rate_null_when_no_qc_answers(self) -> None:
        trials = [
            {
                "policy": "P1",
                "condition": "QC",
                "answered": False,
                "q": 0,
                "qc_violation": False,
                "qc_echo": False,
            }
        ]
        summary = aggregate(trials, model="test-model", n_repeats=1)
        self.assertIsNone(summary["qc_echo_rate"]["P1"])
        self.assertEqual(summary["qc_abstain_rate"]["P1"], 1.0)
        self.assertEqual(summary["qc_violation_rate"]["P1"], 0.0)

    def test_echo_rate_among_qc_answers_only(self) -> None:
        trials = [
            {
                "policy": "P0",
                "condition": "QC",
                "answered": True,
                "q": -4,
                "qc_violation": True,
                "qc_echo": True,
            },
            {
                "policy": "P0",
                "condition": "QC",
                "answered": True,
                "q": -4,
                "qc_violation": True,
                "qc_echo": False,
            },
            {
                "policy": "P0",
                "condition": "QC",
                "answered": False,
                "q": 0,
                "qc_violation": False,
                "qc_echo": False,
            },
        ]
        summary = aggregate(trials, model="test-model", n_repeats=1)
        self.assertEqual(summary["qc_echo_rate"]["P0"], 0.5)
        self.assertEqual(summary["qc_violation_rate"]["P0"], 2 / 3)


class EmptyResultsTests(unittest.TestCase):
    def test_not_run_has_null_rates(self) -> None:
        obj = empty_results()
        self.assertEqual(obj["status"], "not_run")
        self.assertIsNone(obj["model"])
        self.assertIsNone(obj["n_trials"])
        for key in (
            "a_abstain_rate",
            "qc_abstain_rate",
            "qc_violation_rate",
            "qc_echo_rate",
            "q0_abstain_rate",
        ):
            self.assertIsNone(obj[key]["P0"])
            self.assertIsNone(obj[key]["P1"])
        self.assertIsNone(obj["mean_q"]["overall"]["P0"])
        self.assertIn("Aucune clé API LLM", obj["reason"])


class QuestionsIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / "eval" / "questions.json"
        cls.data = json.loads(path.read_text(encoding="utf-8"))
        cls.questions = cls.data["questions"]

    def test_twenty_questions(self) -> None:
        self.assertEqual(len(self.questions), 20)
        ids = [q["id"] for q in self.questions]
        self.assertEqual(ids, [f"{i:02d}" for i in range(1, 21)])

    def test_qc_chunk_never_empty_and_contains_planted(self) -> None:
        for q in self.questions:
            self.assertTrue(str(q["qc_chunk"]).strip(), msg=q["id"])
            self.assertTrue(
                contains_span(q["qc_chunk"], q["planted"]),
                msg=f"{q['id']} planted not in qc_chunk",
            )
            self.assertTrue(
                gold_match(q["gold_chunk"], q["gold"], q.get("gold_aliases") or []),
                msg=f"{q['id']} gold/alias not in gold_chunk",
            )
            self.assertFalse(
                contains_span(q["qc_chunk"], q["gold"]),
                msg=f"{q['id']} gold still in qc_chunk",
            )
            self.assertIn(q["swap_type"], {"numeric", "entity", "date"})


if __name__ == "__main__":
    unittest.main()
