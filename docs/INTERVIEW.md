# Interview talking points : judge-reliability-kit

Five CLI-backed points for a technical screen (no resume recap).

- **`python examples/ambiguous_item_demo.py`** : one intentionally ambiguous item; decomposition returns `item_ambiguous` so you fix the task, not rewrite a fine rubric.
- **`python examples/worked_example.py`** : realistic low-kappa panel; prints how replication separates item ambiguity from rubric underspecification from broken judges.
- **`judgekit report examples/judge_panel_ratings.json -o examples/report.md`** : markdown report with Fleiss kappa and per-item verdicts you can attach to a labeling review.
- **`judgekit report ratings.json --self-consistency-threshold 0.8 --consensus-threshold 0.6`** : tune gates when panels use abstain labels or fewer replicates than the defaults.
- **Non-zero exit when `inter_judge_kappa < 0.6`** : `judgekit report` returns exit `1` on an untrusted panel so CI can gate without parsing prose.

## Three questions

1. **Why is headline kappa not enough?**  
   Item ambiguity and rubric underspecification both crush kappa. Replication lets you see whether each judge disagrees with itself or only with peers.

2. **When do you rewrite the rubric vs change the task?**  
   High self-consistency with low cross-judge consensus points to rubric boundary examples. Low self-consistency on one item points to ambiguity or a broken judge on that item.

3. **How does this relate to judge-drift-sentinel?**  
   This kit diagnoses panel design on static ratings. Sentinel freezes human anchors and detects judge drift across runs. Export panel JSON through `drift-sentinel import-judgekit` when both matter.

## Related instruments

- [judge-field-guide](https://github.com/homayoun-safarpour/judge-field-guide) - CI-tested registry of the LLM-judge tool ecosystem (not a judge implementation; use it to discover peers and keep links from rotting).
- [judge-drift-sentinel](https://github.com/homayoun-safarpour/judge-drift-sentinel) - judge vs system drift on frozen anchors; multi-run history fixture at `examples/drifting/`.

## One limitation

The bundled examples are small caricatures. Production panels need enough replicates per item for stable self-consistency estimates; the thresholds are starting points, not calibrated for your label space.
