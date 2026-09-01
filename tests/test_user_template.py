#!/usr/bin/env python3
"""USER_TEMPLATE is a str.format string; JSON example braces must be escaped."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import USER_TEMPLATE  # noqa: E402


class UserTemplateTests(unittest.TestCase):
    def test_format_injects_context_and_keeps_json_example(self) -> None:
        rendered = USER_TEMPLATE.format(context="c", question="q")
        self.assertIn("c", rendered)
        self.assertIn("q", rendered)
        self.assertIn('{"status":"answer"|"abstain","answer":"..."}', rendered)
        self.assertNotIn("{{", rendered)
        self.assertNotIn("}}", rendered)


if __name__ == "__main__":
    unittest.main()
