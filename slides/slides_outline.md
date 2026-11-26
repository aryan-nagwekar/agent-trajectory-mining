# Doomed in the First Steps?

Twelve-minute talk · Aryan Nagwekar, Niyam Momaya, and Navneet Deswal

## 1. What the first steps tell us

Aryan · 30 seconds

Fixed-cohort LightGBM reaches 0.653 ± 0.021 AUROC at step 5.

[Figure](../figures/01_pipeline.png)

Speaker notes

- At 5, 20, and 40 steps, fixed-cohort AUROC is 0.653 ± 0.021, 0.675 ± 0.017, and 0.723 ± 0.033.
- Same-task AUROC is 0.709 at step 20, so difficulty alone cannot explain the signal.
- Multi-step stopping at α=0.10 saves 9.83%, catches 16.69% of failures, and falsely stops 4.90% [4.16%, 5.76%].
- Held-framework AUROCs are 0.577, 0.622, and 0.535 at step 20. OpenHands reaches 24.38% false stops and retained pattern overlap is zero.

## 2. Why monitor execution logs?

Aryan · 40 seconds

Execution logs give us a monitoring surface even when model activations are unavailable.

[Figure](../figures/01_pipeline.png)

Speaker notes

- Describe the cost of continuing a run that will eventually fail evaluation.
- Explain that the outcome label arrives after the whole trajectory.
- An early-stop detector trades saved work against falsely aborting a run that would resolve.

## 3. Three research questions

Aryan · 40 seconds

We study whole-run patterns, early prediction and stopping, and cross-framework transfer.

[Figure](../figures/01_pipeline.png)

Speaker notes

- Explain the difference between descriptive patterns and prefix prediction.
- Show how grouped tasks separate training, calibration, and testing.
- Preview the stricter framework holdout, which also reserves unseen tasks.

## 4. From logs to action tokens

Aryan · 45 seconds

4,965 labeled runs from 10 submissions form the corpus.

[Figure](../figures/02_dataset.png)

Speaker notes

- Show a real normalized step: `ls -l` → `list`.
- SWE-agent has native actions, OpenHands pairs call and response identifiers, and mini-SWE-agent embeds commands and codes in messages.
- The human validation sample is pending.

## 5. Which patterns survive?

Niyam · 50 seconds

4 robust patterns lead the evidence, followed by 13 primary patterns.

[Figure](../figures/04_patterns.png)

Speaker notes

- PrefixSpan grows frequent prefixes on training tasks and is assessed on distinct tasks.
- Primary means Bonferroni Fisher and same-direction CMH; robust also passes the task-clustered score test.
- Submissions repeat tasks, so the cluster check addresses dependence. BH is secondary.

## 6. Difficulty changes the length result

Niyam · 45 seconds

The length–failure correlation falls from 0.183 overall to 0.044 after centering within bins.

[Figure](../figures/03_length_confound.png)

Speaker notes

- The pooled length relationship combines different kinds of tasks.
- Bin-centered correlation shows a much smaller association.
- This is a descriptive check; it does not assign a causal share.

## 7. Archetypes cover a subset of failures

Niyam · 50 seconds

Base-action trigrams at cutoff 0.40 yield 7 communities covering 265 failed runs.

[Figure](../figures/05_archetypes.png)

Speaker notes

- Walk through the sizes and framework composition.
- Point to the representative action strips and explain why names remain pending.
- Archetype similarities are recomputed within each training fold.

## 8. LSH cuts candidate work

Niyam · 55 seconds

At 23,456 windows, LSH checks 7,640,309 candidate pairs versus 275,080,240 all pairs.

[Figure](../figures/06_lsh_pairs.png)

Speaker notes

- Recall is 98.57%. Pair-count slopes are 1.97 for LSH and 2.00 for all pairs.
- Compiled exact search takes 7.26 s and LSH takes 7.65 s at the largest size.
- The intended LSH advantage is asymptotic under sparse collisions. The observed near-quadratic slopes do not establish that advantage here.

## 9. How early is the signal?

Navneet · 60 seconds

Fixed-cohort LightGBM AUROC rises from 0.653 ± 0.021 at step 5 to 0.675 ± 0.017 at 20 and 0.723 ± 0.033 at 40.

[Figure](../figures/07_fixed_cohort.png)

Speaker notes

- The same 2,018 runs appear at every checkpoint, with a 52.87% failure rate.
- Same-task pair AUROC at step 20 is 0.709; show figure 07b for that diagnostic.
- Highest fixed-cohort AUROC at t=10 is LightGBM (0.671 ± 0.026) and at t=20 is Bag LR (0.678 ± 0.024).
- Failure prevalence changes with eligibility in deployment, so the fixed cohort is the how-early comparison.

## 10. How much can stopping save?

Navneet · 60 seconds

At α=0.10 the multi-checkpoint rule saves 9.83% and catches 16.69% of failures.

[Figure](../figures/08_compute_saved.png)

Speaker notes

- False stops were 4.90% among successful runs with Wilson interval [4.16%, 5.76%].
- The Bonferroni rule used less than its false-stop budget. Its savings are a conservative floor for feasible threshold relaxation on this replay cohort.
- Gross-savings best singles: step 5 (7.94% saved, 5.03% eligible false stops) at α=0.05 and step 5 (17.66% saved, 10.03% eligible false stops) at α=0.10.
- With the observed eligible-success rate within target, the best singles are step 15 (6.12% saved, 4.90% eligible false stops) and step 10 (10.98% saved, 9.96% eligible false stops), respectively.

## 11. Complex features add little

Navneet · 40 seconds

Mined pattern indicators and archetype similarities show no consistent incremental predictive gain.

[Figure](../figures/09_ablation.png)

Speaker notes

- Distinguish the full-trace inferential pattern tier from training-ranked prediction features.
- Only a small robust pattern subset survives difficulty adjustment and the task-clustered check.
- Simple temporal actions are informative even when these more elaborate feature groups add little.

## 12. Transfer weakens discrimination and calibration

Navneet · 65 seconds

Held-framework AUROC at step 20 is 0.577 OpenHands, 0.622 SWE-agent, and 0.535 mini-SWE-agent.

[Figure](../figures/10_transfer.png)

Speaker notes

- OpenHands falsely stops 24.38% at α=0.10 in the step-20 single-policy replay.
- All retained framework-specific significant-pattern lists have zero pairwise overlap.
- Different frameworks use different tools and model mixtures. Task-disjoint transfer tests a stricter setting than within-corpus accuracy.

## 13. Read one real replay

Navneet · 60 seconds

The demo shows the saved model's live prefix score and the first policy crossing.

[Figure](../figures/11_replay.png)

Speaker notes

- Choose one of the suggested held-out runs and move the observed-step slider.
- Explain the blue score markers and orange calibration thresholds.
- Show the original recorded result after the last observed step.

## 14. What the replay can establish

Navneet · 35 seconds

The study measures association and retrospective character savings under an observed logging process.

[Figure](../figures/07b_same_task_pairs.png)

Speaker notes

- Character cost proxies tokens and infrastructure costs.
- Same-task comparisons show discrimination beyond task difficulty but cannot identify a causal share.
- Manual vocabulary agreement and live stopping validation remain open.

## 15. Takeaways

Navneet · 45 seconds

Logs carry early signal in this corpus, but framework transfer weakens both discrimination and calibration.

[Figure](../figures/07_fixed_cohort.png)

Speaker notes

- Recap fixed-cohort AUROC at steps 5, 20, and 40: 0.653 ± 0.021, 0.675 ± 0.017, 0.723 ± 0.033.
- Same-task AUROC is 0.709 at step 20.
- The α=0.10 policy saves 9.83%, catches 16.69%, and falsely stops 4.90% [4.16%, 5.76%].
- Transfer AUROCs at step 20 are 0.577, 0.622, and 0.535. OpenHands false stops reach 24.38% and pattern overlap is zero.
