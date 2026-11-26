# Deviations

## Phase 1

- The selected mini-SWE-agent submission publishes per-instance resolution booleans. The pilot uses these official labels in place of a resolved-ID list. The data card records the difference between the metadata percentage and the count-based rate.
- The proposal calls for per-step token counts as a compute proxy. The inspected SWE-agent trace exposes aggregate tokens in `info.model_stats`, while the mini-SWE-agent example exposes cost and API-call totals there. The final choice of compute proxy depends on the Phase 2 coverage check.

## Phase 2

- The pilot replaces one shared resolved task with a shared unresolved task to meet the per-framework class-balance requirement. The failure-rate comparison applies to the expanded cohort because the pilot was selected for schema coverage and label balance.
- The expanded OpenHands Devstral export serializes tool calls in assistant text and places the matching named execution result in the next reply. The parser handles this format alongside the structured call IDs used in the other OpenHands submissions.
- No normalized step had an attributable API token count in the available exports. The schema keeps those values nullable, and the pipeline retains observed character cost and available run-level token totals.

## Phases 3 and 4

- Stricter leave-one-framework-out design: 3 held-out frameworks × 5 outer task folds yield 15 holdout splits. In each split the held-out framework is evaluated only on unseen tasks. The other frameworks supply training and calibration on disjoint tasks. Each held-out run is evaluated once across the five task folds.
- PrefixSpan candidates and their training statistics come from each fold's training partition. The primary Fisher and CMH assessment uses that fold's unseen tasks. The same tasks repeat across submissions, so runs are dependent within task. The task-clustered score check addresses that dependence. Report the four patterns passing all three Bonferroni checks first, the thirteen passing Fisher and CMH second, and BH counts as secondary context.

## Phases 5 and 6

- The prespecified outcome-token four-gram analysis produced no communities of at least 20 failed runs. The reported archetype is from a sensitivity view with collapsed base-action trigrams at the same 0.60 Jaccard cutoff. Its benchmark is accompanied by the default-view benchmark.
- MinHash-LSH retrieves candidate edges. Exact shingle-set Jaccard verifies the cutoff, so MinHash estimation noise does not serve as a second cutoff. The benchmark uses the full available failed-run pool as its largest sample.
- Prediction references consider the default outcome four-gram view and the base-action trigram view. Every medoid, vocabulary and top-K pattern list is fit on the corresponding training partition. Pattern ranking uses absolute support difference irrespective of inferential significance. A missing archetype contributes a constant similarity column.
- Per-step API token counts are unavailable, so the steps-only baseline primarily uses observed character count and a visible submission flag. No LLM judge runs were made because no key was supplied.

## Additions before Phase 7

- The fixed-cohort evaluation follows the same runs with at least 40 recorded steps at all six checkpoints. It evaluates the checkpoint models on that fixed population. The deployment view retains all runs reaching each checkpoint and reports its changing failure prevalence.
- The descriptive archetype sweep used exact base-action Jaccard graphs. The selected setting is **k=3, Jaccard ≥0.40**, the highest threshold yielding at least two qualifying communities, at least 5% coverage, and no community holding more than 85% of covered runs. It yielded seven communities covering 265 of 2,169 failed runs. Each prediction fold repeats the selection on its training failures only. If that criterion yields no setting, it uses the highest training threshold with a qualifying community, or a constant feature when none exists.
- The scaling unit changed from full failed trajectories to full 20-step windows at stride 5 from every run. The corpus has 23,456 such windows, so the 40,000-window target is unavailable. The exact benchmark is compiled C++ and counts qualifying window pairs even when several windows have identical shingle sets.

## Phases 7 and 8

- Early-stop replay uses characters across all submissions because observed step-level token costs are unavailable. A run ending exactly at a checkpoint has no remaining action to stop. Wilson intervals are nominal run-level intervals under within-task dependence. The report gives both all-test-success and continuing-success denominators.
- Transfer reuses the stricter task-disjoint framework holdouts. Model fitting and calibration use the two non-held frameworks on disjoint tasks, and the policy is evaluated on the held framework.

## Phase 9

- The final script generates a fixed-cohort early-detection curve and a same-task pair diagnostic alongside the requested figures. The LSH figure emphasizes candidate-pair workload and distinguishes the intended asymptotic benefit from the measured native exact runtime.
- The quick target uses an isolated, seeded common-task sample, smaller training grids and a small bootstrap. It may contain a single class in a particular held-out framework fold and is a pipeline smoke test. Its outputs do not contribute to the preprint or the reported full-cohort results.
- Public raw files are cached by submission and verified during reproduction. The local project has no Git commit to record, so the metrics file records the pinned upstream SWE-bench experiments commit and the run configuration.
