#!/usr/bin/env python3
"""Harness P0 vs P1 : chunk figé injecté, pas de retriever.

Ne pas « tuner » les prompts pour gagner. Une seule API LLM.
Temperature 0, 3 repeats. Sans credentials : status=not_run, taux nuls.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorer import (  # noqa: E402
    CONDITIONS,
    POLICIES,
    aggregate,
    empty_results,
    score_trial,
)

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "eval" / "questions.json"
RESULTS_DIR = ROOT / "results"
LATEST_PATH = RESULTS_DIR / "latest.json"
TRANSCRIPTS_PATH = RESULTS_DIR / "transcripts.jsonl"

N_REPEATS = 3
TEMPERATURE = 0

P0_SYSTEM = (
    "Réponds en utilisant UNIQUEMENT le contexte fourni."
)
P1_SYSTEM = (
    "Réponds en utilisant UNIQUEMENT le contexte fourni. "
    "Si le contexte ne contient pas assez d'information, "
    'mets status à "abstain" et laisse answer vide.'
)
USER_TEMPLATE = (
    "Contexte:\n{context}\n\n"
    "Question:\n{question}\n\n"
    'Réponds par un unique objet JSON : '
    '{"status":"answer"|"abstain","answer":"..."}.'
)


def load_questions() -> list[dict]:
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return list(data["questions"])


def context_for(q: dict, condition: str) -> str:
    if condition == "A":
        return q["gold_chunk"]
    if condition == "QC":
        chunk = q["qc_chunk"]
        if not str(chunk).strip():
            raise ValueError(f"qc_chunk vide pour {q['id']} (interdit)")
        return chunk
    return ""


def detect_provider() -> dict | None:
    """Une seule API : la première clé trouvée."""
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        return {
            "name": "openai",
            "api_key": openai_key,
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip(),
            "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        }

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        return {
            "name": "anthropic",
            "api_key": anthropic_key,
            "model": os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest").strip(),
            "base_url": "https://api.anthropic.com",
        }

    compatible = [
        ("GROQ_API_KEY", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
        ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
        ("TOGETHER_API_KEY", "https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        ("FIREWORKS_API_KEY", "https://api.fireworks.ai/inference/v1", "accounts/fireworks/models/llama-v3p3-70b-instruct"),
        ("MISTRAL_API_KEY", "https://api.mistral.ai/v1", "mistral-small-latest"),
        ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-chat"),
        ("XAI_API_KEY", "https://api.x.ai/v1", "grok-2-latest"),
    ]
    for env_name, base, default_model in compatible:
        key = os.environ.get(env_name, "").strip()
        if key:
            return {
                "name": "openai-compatible",
                "api_key": key,
                "model": os.environ.get("OPENAI_MODEL", default_model).strip(),
                "base_url": os.environ.get("OPENAI_BASE_URL", base).strip(),
                "env_name": env_name,
            }
    return None


def _http_json(url: str, payload: dict, headers: dict, timeout: int = 60) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_err = exc
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2 ** attempt)
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                continue
            raise RuntimeError(f"HTTP {exc.code} {url}: {detail[:500]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                continue
            raise
    raise RuntimeError(str(last_err))


def call_llm(provider: dict, system: str, user: str) -> tuple[str, str]:
    """Retourne (texte, model_id)."""
    if provider["name"] == "anthropic":
        data = _http_json(
            f"{provider['base_url'].rstrip('/')}/v1/messages",
            {
                "model": provider["model"],
                "max_tokens": 512,
                "temperature": TEMPERATURE,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            {
                "x-api-key": provider["api_key"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        model_id = data.get("model") or provider["model"]
        parts = data.get("content") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        return text, str(model_id)

    url = f"{provider['base_url'].rstrip('/')}/chat/completions"
    payload = {
        "model": provider["model"],
        "temperature": TEMPERATURE,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if provider["name"] == "openai":
        payload["response_format"] = {"type": "json_object"}
    data = _http_json(
        url,
        payload,
        {
            "Authorization": f"Bearer {provider['api_key']}",
            "Content-Type": "application/json",
        },
    )
    model_id = data.get("model") or provider["model"]
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Réponse LLM vide: {json.dumps(data)[:400]}")
    text = choices[0].get("message", {}).get("content") or ""
    return str(text), str(model_id)


def write_latest(obj: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run() -> dict:
    provider = detect_provider()
    if provider is None:
        result = empty_results()
        write_latest(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(
            "status=not_run — aucune clé LLM dans l'environnement. "
            "results/latest.json écrit avec des taux nuls (aucun chiffre inventé).",
            file=sys.stderr,
        )
        return result

    questions = load_questions()
    system_by_policy = {"P0": P0_SYSTEM, "P1": P1_SYSTEM}
    trials: list[dict] = []
    model_ids: list[str] = []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if TRANSCRIPTS_PATH.exists():
        TRANSCRIPTS_PATH.unlink()

    total = len(questions) * len(CONDITIONS) * len(POLICIES) * N_REPEATS
    done = 0
    print(f"provider={provider['name']} model={provider['model']} n_calls={total}", file=sys.stderr)

    with TRANSCRIPTS_PATH.open("w", encoding="utf-8") as log:
        for q in questions:
            for policy in POLICIES:
                for condition in CONDITIONS:
                    ctx = context_for(q, condition)
                    user = USER_TEMPLATE.format(context=ctx, question=q["question"])
                    for repeat in range(N_REPEATS):
                        raw, model_id = call_llm(provider, system_by_policy[policy], user)
                        model_ids.append(model_id)
                        scored = score_trial(
                            condition=condition,  # type: ignore[arg-type]
                            raw=raw,
                            gold=q["gold"],
                            gold_aliases=q.get("gold_aliases") or [],
                            planted=q["planted"],
                        )
                        row = {
                            "question_id": q["id"],
                            "policy": policy,
                            "condition": condition,
                            "repeat": repeat,
                            "model": model_id,
                            "swap_type": q.get("swap_type"),
                            "context": ctx,
                            "question": q["question"],
                            "raw": raw,
                            **scored,
                        }
                        trials.append(row)
                        log.write(json.dumps(row, ensure_ascii=False) + "\n")
                        done += 1
                        if done % 20 == 0 or done == total:
                            print(f"{done}/{total}", file=sys.stderr)

    unique_models = sorted(set(model_ids))
    model_logged = unique_models[0] if len(unique_models) == 1 else ",".join(unique_models)
    result = aggregate(trials, model=model_logged, n_repeats=N_REPEATS)
    write_latest(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
