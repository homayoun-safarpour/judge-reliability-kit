# judge-reliability-kit

**A low kappa tells you your LLM judge panel is broken. It does not tell you whether to fix the rubric or the task. This tells you which.**

[![CI](https://github.com/homayoun-safarpour/judge-reliability-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/homayoun-safarpour/judge-reliability-kit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## The problem

You built a panel of LLM judges. Each one looks competent when you read its outputs. Then you measure inter-judge agreement and get a Fleiss kappa of 0.20, barely above chance.

Now what?

There are two completely different reasons a panel disagrees, and they need **opposite fixes**:

| cause | what is actually happening | the fix | the fix that will NOT work |
| --- | --- | --- | --- |
| **Item ambiguity** | The item genuinely has no single right label. Even one judge, asked twice, answers differently. | Change the **task**: add an abstain label, route to a human, exclude from scoring. | Rewriting the rubric. It is already fine. |
| **Rubric underspecification** | Each judge is perfectly self-consistent, and each has settled on a *different* stable reading of your criterion. | Change the **rubric**: find the disputed clause, add a boundary example on each side. | Adding judges, or averaging harder. Three stable disagreements average to noise. |

**A single kappa cannot distinguish them.** Both produce the same bad number. Pick the wrong fix and you will spend a month rewriting a rubric that was never the problem.

## The insight

The two causes become separable the moment you add **replication**: have each judge score each item more than once, independently.

```
low  intra-judge agreement                    ->  the item moves the judge     ->  ITEM AMBIGUITY
high intra-judge, low inter-judge agreement   ->  stable judges, stably apart  ->  RUBRIC UNDERSPECIFIED
low  intra-judge agreement everywhere         ->  the judge cannot repeat itself -> BROKEN JUDGE
```

That is the whole idea. This package implements it, plus the agreement statistics and bias probes you need around it.

## Install

```bash
pip install judge-reliability-kit
```

or from source:

```bash
git clone https://github.com/homayoun-safarpour/judge-reliability-kit
cd judge-reliability-kit
pip install -e ".[dev]"
```

## Quickstart

Interview pack: [docs/INTERVIEW.md](docs/INTERVIEW.md).

Claim boundaries: [docs/RELIABILITY_CARD.md](docs/RELIABILITY_CARD.md).

```python
from judgekit import decompose, markdown_report

# ratings[item_id][judge_id] = list of REPEATED labels from that judge
ratings = {
    "REC001": {
        "gpt-4o":  ["include", "include", "include", "include"],
        "claude":  ["exclude", "exclude", "exclude", "exclude"],
        "llama":   ["include", "include", "include", "include"],
    },
    "REC002": {
        "gpt-4o":  ["include", "exclude", "include", "exclude"],
        "claude":  ["exclude", "include", "exclude", "include"],
        "llama":   ["include", "exclude", "exclude", "include"],
    },
}

result = decompose(ratings)
print(result.summary())

for item in result.items:
    print(item.item_id, item.verdict, "->", item.actionable)
```

REC001: every judge repeats itself perfectly and they still disagree → **rubric underspecified**.
REC002: every judge contradicts itself → **item ambiguous**.

Same kappa. Different diagnosis. Different fix.

## Command line

```bash
judgekit report ratings.json -o reliability.md
```

Exits non-zero when the panel is below a trustworthy kappa, so you can gate a pipeline on it:

```yaml
- name: Judge panel must be trustworthy before we ship scores
  run: judgekit report data/panel_ratings.json -o reliability.md
```

## What is in the box

### Agreement statistics (`judgekit.agreement`)

| function | use it when |
| --- | --- |
| `fleiss_kappa` | fixed panel, every judge rates every item |
| `cohen_kappa` | exactly two judges |
| `krippendorff_alpha` | judges skipped items, or you are sampling (handles `None`) |
| `pairwise_kappas` | **always**. The panel mean hides whether two judges are locked together while a third is orthogonal |
| `percent_agreement` | only as a foil, to show how misleading it is |
| `interpret_kappa` | turns a number into what it *licenses you to claim* |

The imbalance trap is asserted as a test: two judges who both say "exclude" 96% of the time agree 96% of the time and share **nothing**. `percent_agreement` reads 0.96 while `cohen_kappa` goes negative.

### Cause decomposition (`judgekit.decompose`)

`decompose(ratings)` returns per-item verdicts (`clean`, `item_ambiguous`, `rubric_underspecified`, `judge_unstable`), each carrying a concrete action, plus panel-level intra- and inter-judge agreement.

### Bias probes (`judgekit.probes`)

Model-agnostic controlled perturbations. Pass your own `score_fn`; the probe handles the perturbation and the statistics.

| probe | changes exactly one thing that should not matter |
| --- | --- |
| `position_bias` | the order the two answers are shown in |
| `verbosity_bias` | pads the answer with content-free text |
| `self_enhancement_bias` | who authored the answer being scored |

### Reporting (`judgekit.report`)

`markdown_report(...)` renders a report that leads with the diagnosis and the action, not the statistic.

## Worked example

```bash
python examples/worked_example.py
python examples/ambiguous_item_demo.py
```

The ambiguous demo prints `verdict=item_ambiguous` for hedge-heavy item wording
(see `examples/ambiguous_item_output.txt`).

A 36-record binary-label panel with three LLM judges. Real output:

```
WHAT A BARE KAPPA TELLS YOU
Fleiss kappa = 0.242  (fair - a shared construct may exist but the rubric is not carrying it)
Actionable next step: none. You know it is bad. You do not know why.

WHAT REPLICATION TELLS YOU
inter-judge Fleiss kappa (single pass): 0.242
  same panel after majority vote:       0.245   (inflation +0.002)
mean intra-judge self-consistency:      0.867

disagreement attributed to:
  item ambiguity            10  (28%)
  rubric underspecified     12  (33%)
  judge instability          0  (0%)
  clean                     14  (39%)

dominant cause: rubric_underspecified

  REC014  [item_ambiguous]
    self-consistency 0.44 | consensus 0.33
    -> Change the TASK: allow abstain, route to human, or exclude from scoring.

  REC024  [rubric_underspecified]
    self-consistency 1.00 | consensus 0.00
    -> Change the RUBRIC: pin the disputed clause, add a boundary example.
```

One number became a work plan: 12 records need a rubric fix, 10 need to leave the scored set, and 14 are fine. See [`examples/report.md`](examples/report.md) for the full generated report.

### Majority-vote inflation

`decompose` reports the panel kappa twice (once on a single pass, once after majority-voting the replicates) because the gap between them is itself a finding:

```python
result.single_pass_kappa        # what you would have measured without replication
result.modal_kappa              # what you measure after majority vote
result.majority_vote_inflation  # modal - single_pass
```

A large positive inflation is a **warning, not a win**. It means your headline agreement is being manufactured by aggregation over judges whose individual verdicts are not reproducible. Report both numbers or you are reporting a statistic about your voting rule, not about your judges.

## Why this exists

Production LLM evaluation panels often look competent judge-by-judge and still disagree as a group. A low kappa alone does not say whether to rewrite the rubric or replicate ambiguous items, and teams keep picking the wrong fix. This kit separates those failure modes with a named, tested decomposition (`tests/test_decompose.py::test_the_two_causes_produce_the_same_kappa_but_different_verdicts`).

Regulatory context: the EU AI Act's high-risk conformity assessment obligations make documented evaluation reliability (not just evaluation results) a compliance artefact. "We used an LLM judge" is not an answer to "how do you know your evaluation is sound".

## Design commitments

- **No LLM dependency.** The package never calls a model. You bring ratings or a `score_fn`; it does statistics. This keeps it testable, deterministic, offline, and free.
- **Numpy only.** One runtime dependency.
- **Every claim is a test.** `tests/test_decompose.py::test_the_two_causes_produce_the_same_kappa_but_different_verdicts` asserts the central claim of the package directly.
- **Errors that teach.** Passing single-pass ratings raises a message explaining *why* replication is required rather than a `KeyError`.

## Field alignment

Meta-eval culture (RewardBench / JudgeBench / evalstats-style work) treats the **judge as the system under test**. This kit is the small offline cousin: decompose a bad panel kappa into a named fix. Claim boundaries and fixtures: [docs/RELIABILITY_CARD.md](docs/RELIABILITY_CARD.md).

## Fail-closed demo

Ambiguous items should surface as `item_ambiguous`, not a quiet average:

```bash
python examples/ambiguous_item_demo.py
# or: judgekit report examples/ambiguous_item_fixture.json -o examples/ambiguous_item_output.txt
```

Gate a pipeline when panel kappa is untrustworthy (`judgekit report` exits non-zero). See the CI snippet under Command line.

## Contributing

Issues and PRs welcome. Run `ruff check src tests` and `pytest` before opening one.

## Citation

```bibtex
@software{safarpour_judge_reliability_kit_2026,
  author  = {Homayoun Safarpour},
  title   = {judge-reliability-kit: diagnosing why an LLM judge panel disagrees},
  year    = {2026},
  url     = {https://github.com/homayoun-safarpour/judge-reliability-kit},
  license = {MIT}
}
```

Author: Homayoun Safarpour · [LinkedIn](https://www.linkedin.com/in/homayoun-safarpour/)

## License

MIT. See [LICENSE](LICENSE).
