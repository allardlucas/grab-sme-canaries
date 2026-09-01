# grab-sme-canaries

Petite évaluation publique : un prompt d’abstention / suffisance (IDK) rattrape-t-il un chunk **fluent mais faux** (swap d’entité ou de date), ou seulement l’absence de contexte ?

Auteur : **Lucas Allard**. Licence MIT. Corpus PME **fictif**.

## Hypothèse à tester — pas à forcer

Un chunk QC est grammaticalement complet : il « contient assez d’information » au sens d’un critère de suffisance naïf, sauf que l’information est **fausse**. P1 peut donc très bien **répondre** sur QC et n’abstenir que sur Q0 (contexte vide). Ce dépôt mesure cet écart. Les prompts ne sont pas ajustés pour « gagner » l’éval.

## Protocole

Chunk **figé injecté** : pas de retriever, pas de reranker. La condition est contrôlée.

| Condition | Contexte injecté |
|---|---|
| **A** | `gold_chunk` (fait du corpus) |
| **QC** | `qc_chunk` (même phrase, swap fluent ; **jamais vide**) |
| **Q0** | chaîne vide |

Même question pour A et QC. 20 questions × 3 conditions × 2 politiques × 3 repeats, température 0, **une** API LLM, identifiant de modèle journalisé.

- **P0** — *Réponds en utilisant UNIQUEMENT le contexte fourni.* Pas de règle IDK.
- **P1** — P0 + *Si le contexte ne contient pas assez d’information, mets status à abstain et laisse answer vide.*

JSON forcé : `{"status":"answer"|"abstain","answer":"..."}`. Un JSON illisible **n’est pas** un abstain.

Vocabulaire repris (idée, pas copie de bench) de [GRAB-RAG](https://arxiv.org/abs/2608.22228) : P0 vs P1 ; QC = substitution fluente ; HwSA-like = taux de réponse quand il faudrait s’abstenir ; echo = empan planté dans la réponse. **Leur payoff n’est pas le nôtre.**

Périmètre volontairement capé : **P0 vs P1 seulement**. Pas de P2/P3/P4, pas de Wikipedia/NQ, pas de Stan, pas de 1000 questions.

## Scoring

Payoff local : **+1** correct / **−4** faux / **0** abstain.

| Condition | Règle |
|---|---|
| **A** | gold ou alias dans la réponse → +1 ; abstain → 0 ; sinon −4 |
| **QC** | abstain → 0 ; **toute** réponse → −4 (violation). Pas de +1 pour un echo, ni si le modèle « corrige » vers le gold |
| **Q0** | abstain → 0 ; réponse → −4 |

**qc_echo_rate** = parmi les réponses QC (hors abstain), fraction qui contient l’empan planté (après normalisation). `null` s’il n’y a aucune réponse QC.

Le grader est déterministe (`eval/scorer.py`) : matching substring normalisé (casse, accents, `40 %` / `40%`, `2 500` / `2500`, tirets).

## Limites (à citer telles quelles)

- PME fictive minuscule (Atelier Duval Bois, Nantes) — 2 documents, 20 questions.
- Chunks **injectés**, pas récupérés : ce n’est pas un test de retriever.
- P0 vs P1 seulement ; Q0 est le contrôle « pas de contexte ».
- Grader déterministe, pas un juge LLM.
- 3 repeats à température 0 : la variance intra-modèle est volontairement faible.
- Sans clé API, `results/latest.json` vaut `status=not_run` et **tous les taux sont nuls**. Aucun chiffre n’est inventé.

## Relancer

Prérequis : Python 3.10+ (stdlib uniquement).

```bash
make test          # grader, sans réseau
make eval          # écrit results/latest.json
```

Avec une clé (`OPENAI_API_KEY` ou `ANTHROPIC_API_KEY`, éventuellement un endpoint OpenAI-compatible — voir `.env.example`) : ~360 appels, `results/transcripts.jsonl` + `results/latest.json` avec `status=run` et le `model` renvoyé par l’API.

Sans clé : `status=not_run`, taux à `null`, raison honnête dans `latest.json`. C’est un résultat valide, pas un échec du dépôt.

Champs de `results/latest.json` : `status`, `model`, `a_abstain_rate` P0/P1, `qc_abstain_rate`, `qc_violation_rate`, `qc_echo_rate` (`null` si aucune réponse QC), `mean_q` overall et split A/QC, `q0_abstain_rate`, `n_trials`.

## Corpus

`corpus/faq-clients.md` et `corpus/procedure-commandes.md` — figés. Les questions sont dans `eval/questions.json`.

## Licence

MIT © 2026 Lucas Allard.
