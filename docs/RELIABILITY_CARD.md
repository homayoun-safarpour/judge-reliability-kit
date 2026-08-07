# Reliability card — judge-reliability-kit

| Field | Value |
| --- | --- |
| **Job** | Diagnose why an LLM judge panel has low agreement |
| **Primary metrics** | Fleiss / Cohen kappa (chance-corrected); intra-judge vs inter-judge agreement |
| **Named verdicts** | `item_ambiguous`, `rubric_underspecified`, `broken_judge` (see `decompose`) |
| **Fixtures** | `examples/judge_panel_ratings.json`, `examples/ambiguous_item_fixture.json` |
| **Central test** | `tests/test_decompose.py::test_the_two_causes_produce_the_same_kappa_but_different_verdicts` |
| **Runtime deps for core claim** | numpy only; **no LLM calls** |
| **Claim** | Same panel kappa can hide opposite fixes; replication separates them |
| **Not claimed** | Human–model preference leaderboard; RewardBench/JudgeBench scores; provider model quality |

## Field alignment (not affiliation)

Same problem family as meta-eval work around RewardBench / JudgeBench / evalstats:
**measure the judge before trusting the score.** This kit is a small offline decomposer for panel disagreement, not a full preference benchmark.
