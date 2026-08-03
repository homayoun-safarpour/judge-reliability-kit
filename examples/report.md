# Screening panel - worked example

## Headline

- **Inter-judge Fleiss kappa:** 0.242 - fair - a shared construct may exist but the rubric is not carrying it
- **Mean intra-judge self-consistency:** 0.867
- **Dominant cause of disagreement:** `rubric_underspecified`

Most of your disagreement is **rubric underspecification**. The items are fine; your judges each hold a different stable reading of the criterion. Find the disputed clause, rewrite it with a boundary example on each side, and re-run. Adding more judges will not help.

## Where the disagreement comes from

| cause | items | share | what it means |
| --- | ---: | ---: | --- |
| item ambiguity | 10 | 28% | judges cannot reproduce themselves |
| rubric underspecified | 12 | 33% | judges are each stable, and stably different |
| judge instability | 0 | 0% | a judge is not deterministic enough to score with |
| clean | 14 | 39% | stable and agreeing |

## Per-judge self-consistency

| judge | self-consistency |
| --- | ---: |
| `claude-judge` | 0.856 |
| `gpt-4o-judge` | 0.856 |
| `llama-70b-judge` | 0.889 |

## Contested items (22 total, showing up to 20)

| item | verdict | self-consistency | consensus | action |
| --- | --- | ---: | ---: | --- |
| `REC014` | item_ambiguous | 0.44 | 0.33 | Change the TASK: allow abstain, route to human, or exclude from scoring. |
| `REC015` | item_ambiguous | 0.61 | 1.00 | Change the TASK: allow abstain, route to human, or exclude from scoring. |
| `REC016` | item_ambiguous | 0.39 | 1.00 | Change the TASK: allow abstain, route to human, or exclude from scoring. |
| `REC017` | item_ambiguous | 0.44 | 0.33 | Change the TASK: allow abstain, route to human, or exclude from scoring. |
| `REC018` | item_ambiguous | 0.61 | 0.33 | Change the TASK: allow abstain, route to human, or exclude from scoring. |
| `REC019` | item_ambiguous | 0.67 | 0.33 | Change the TASK: allow abstain, route to human, or exclude from scoring. |
| `REC020` | item_ambiguous | 0.39 | 1.00 | Change the TASK: allow abstain, route to human, or exclude from scoring. |
| `REC021` | item_ambiguous | 0.61 | 1.00 | Change the TASK: allow abstain, route to human, or exclude from scoring. |
| `REC022` | item_ambiguous | 0.61 | 0.33 | Change the TASK: allow abstain, route to human, or exclude from scoring. |
| `REC023` | item_ambiguous | 0.44 | 0.33 | Change the TASK: allow abstain, route to human, or exclude from scoring. |
| `REC024` | rubric_underspecified | 1.00 | 0.00 | Change the RUBRIC: pin the disputed clause, add a boundary example. |
| `REC025` | rubric_underspecified | 1.00 | 0.00 | Change the RUBRIC: pin the disputed clause, add a boundary example. |
| `REC026` | rubric_underspecified | 1.00 | 0.00 | Change the RUBRIC: pin the disputed clause, add a boundary example. |
| `REC027` | rubric_underspecified | 1.00 | 0.00 | Change the RUBRIC: pin the disputed clause, add a boundary example. |
| `REC028` | rubric_underspecified | 1.00 | 0.00 | Change the RUBRIC: pin the disputed clause, add a boundary example. |
| `REC029` | rubric_underspecified | 1.00 | 0.00 | Change the RUBRIC: pin the disputed clause, add a boundary example. |
| `REC030` | rubric_underspecified | 1.00 | 0.00 | Change the RUBRIC: pin the disputed clause, add a boundary example. |
| `REC031` | rubric_underspecified | 1.00 | 0.00 | Change the RUBRIC: pin the disputed clause, add a boundary example. |
| `REC032` | rubric_underspecified | 1.00 | 0.00 | Change the RUBRIC: pin the disputed clause, add a boundary example. |
| `REC033` | rubric_underspecified | 1.00 | 0.00 | Change the RUBRIC: pin the disputed clause, add a boundary example. |

## Method note

Ambiguity and underspecification are separated by **replication**: each judge scores each item more than once, independently. Without replicates the two causes are mathematically indistinguishable, which is why a single kappa cannot tell you which fix to apply.
