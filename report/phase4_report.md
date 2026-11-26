# Doomed in the First Steps?

Phase 4 checkpoint

## Executive summary

The audit covers **10 submissions**, **4,965 runs**, and **196,501 steps**. The maximum base-action share was 28.53%, the maximum action-outcome token share was 28.48%, and the maximum unknown test-outcome share was 26.16%. All audit gates passed.

**4 robust patterns** passed the Bonferroni-corrected Fisher, difficulty-adjusted CMH, and task-clustered checks in the same fold. They are part of **13 primary patterns** that passed Fisher and CMH in the same direction. Fisher alone retained 72 of 4,250 candidates. BH counts are secondary. [Robust pattern list](../results/patterns_robust.csv) · [Primary pattern list](../results/patterns_primary.csv)

The correlation between total steps and failure was **0.183** overall and **0.044** after centering within difficulty bins.

## Submission coverage

| Submission | Framework | Base model | Runs | Official resolve | Normalized resolve |
| --- | --- | --- | --- | --- | --- |
| 20241029_OpenHands-CodeAct-2.1-sonnet-20241022 | OpenHands | claude-3-5-sonnet-20241022 | 493 | 53.00% | 53.75% |
| 20250520_openhands_devstral_small | OpenHands | mistralai/Devstral-Small-2505 | 492 | 46.80% | 47.56% |
| 20250524_openhands_claude_4_sonnet | OpenHands | claude-4-sonnet-20250514 | 499 | 70.40% | 70.54% |
| 20250807_openhands_gpt5 | OpenHands | openai/gpt-5-2025-08-07 | 499 | 71.80% | 71.94% |
| 20240620_sweagent_claude3.5sonnet | SWE-agent | claude-3-5-sonnet-20241022 | 486 | 33.60% | 34.57% |
| 20250511_sweagent_lm_32b | SWE-agent | Qwen 2.5 | 496 | 40.20% | 40.52% |
| 20250522_sweagent_claude-4-sonnet-20250514 | SWE-agent | claude-4-sonnet-20250514 | 500 | 66.60% | 66.60% |
| 20250726_mini-v1.0.0_claude-sonnet-4-20250514 | mini-SWE-agent | claude-4-sonnet-20250514 | 500 | 64.80% | 64.80% |
| 20250726_mini-v1.0.0_gemini-2.5-pro | mini-SWE-agent | gemini-2.5-pro | 500 | 53.60% | 53.60% |
| 20250726_mini-v1.0.0_o3-2025-04-16 | mini-SWE-agent | o3-20250416 | 500 | 58.40% | 58.40% |

| Framework | Submissions | Runs | Coverage flag |
| --- | --- | --- | --- |
| OpenHands | 4 | 1,983 | Pass |
| SWE-agent | 3 | 1,482 | Pass |
| mini-SWE-agent | 3 | 1,500 | Pass |

The coverage flags use fewer than 2 submissions or fewer than 1,000 runs. Official resolution uses the full Verified roster. Normalized resolution uses the eligible parsed runs. [Submission audit](../results/submission_audit.csv)

## Action vocabulary audit

Shares use all normalized steps from the corresponding framework.

| Action | OpenHands | SWE-agent | mini-SWE-agent |
| --- | --- | --- | --- |
| `create` | 7.06% | 9.21% | 9.54% |
| `edit` | 14.25% | 17.98% | 13.30% |
| `error_format` | 0.00% | 0.00% | 4.17% |
| `install` | 0.49% | 0.24% | 0.17% |
| `list` | 2.59% | 0.85% | 3.09% |
| `other_bash` | 0.66% | 0.20% | 0.51% |
| `revert` | 0.13% | 0.61% | 0.74% |
| `run_script` | 19.59% | 18.65% | 14.67% |
| `run_test` | 7.97% | 5.84% | 2.91% |
| `search` | 19.84% | 13.77% | 20.42% |
| `submit` | 0.00% | 2.52% | 3.51% |
| `think` | 3.22% | 0.67% | 0.00% |
| `vcs` | 1.65% | 0.93% | 0.70% |
| `view` | 22.56% | 28.53% | 26.28% |

### Outcome-augmented tokens

| Token | OpenHands | SWE-agent | mini-SWE-agent |
| --- | --- | --- | --- |
| `create` | 6.32% | 9.11% | 0.01% |
| `create:error` | 0.38% | 0.10% | 0.41% |
| `create:ok` | 0.36% | 0.00% | 9.12% |
| `edit` | 12.61% | 15.78% | 0.14% |
| `edit:error` | 1.11% | 2.20% | 1.90% |
| `edit:ok` | 0.53% | 0.00% | 11.27% |
| `error_format:error` | 0.00% | 0.00% | 4.17% |
| `install` | 0.03% | 0.19% | 0.02% |
| `install:error` | 0.09% | 0.06% | 0.04% |
| `install:ok` | 0.38% | 0.00% | 0.11% |
| `list` | 0.00% | 0.85% | 0.00% |
| `list:error` | 0.04% | 0.00% | 0.12% |
| `list:ok` | 2.55% | 0.00% | 2.96% |
| `other_bash` | 0.12% | 0.20% | 0.00% |
| `other_bash:error` | 0.36% | 0.00% | 0.24% |
| `other_bash:ok` | 0.18% | 0.00% | 0.27% |
| `revert` | 0.02% | 0.61% | 0.00% |
| `revert:error` | 0.01% | 0.00% | 0.09% |
| `revert:ok` | 0.10% | 0.00% | 0.64% |
| `run_script` | 0.13% | 12.88% | 0.03% |
| `run_script:error` | 6.12% | 5.64% | 4.12% |
| `run_script:ok` | 13.34% | 0.13% | 10.52% |
| `run_test` | 0.02% | 1.53% | 0.28% |
| `run_test:error` | 0.00% | 0.19% | 0.00% |
| `run_test:fail` | 3.31% | 1.40% | 1.66% |
| `run_test:pass` | 4.64% | 2.72% | 0.98% |
| `search` | 0.05% | 13.75% | 0.10% |
| `search:error` | 2.31% | 0.03% | 1.49% |
| `search:ok` | 17.49% | 0.00% | 18.82% |
| `submit` | 0.00% | 2.51% | 3.50% |
| `submit:error` | 0.00% | 0.00% | 0.01% |
| `submit:ok` | 0.00% | 0.00% | 0.00% |
| `think` | 3.22% | 0.67% | 0.00% |
| `vcs` | 0.00% | 0.93% | 0.00% |
| `vcs:error` | 0.05% | 0.00% | 0.01% |
| `vcs:ok` | 1.59% | 0.00% | 0.68% |
| `view` | 17.58% | 28.48% | 0.14% |
| `view:error` | 0.72% | 0.05% | 0.52% |
| `view:ok` | 4.26% | 0.00% | 25.62% |

### Outcomes on `run_test` steps

| Framework | Test steps | Pass | Fail | None | Error | OK |
| --- | --- | --- | --- | --- | --- | --- |
| OpenHands | 6,971 | 58.27% | 41.51% | 0.22% | 0.00% | 0.00% |
| SWE-agent | 3,960 | 46.69% | 23.91% | 26.16% | 3.23% | 0.00% |
| mini-SWE-agent | 1,198 | 33.56% | 56.93% | 9.52% | 0.00% | 0.00% |

The audit checks for any token above 40% and for `run_test:none` above 30%. The SWE-agent `error` share records explicit error text without a benchmark-test failure marker. The denominator includes every test step.

[Action distribution](../results/action_distribution.csv) · [Token distribution](../results/token_distribution.csv) · [Test outcomes](../results/run_test_outcomes.csv) · [Manual validation scorer](../scripts/score_validation.py)

## Task splits and difficulty

GroupKFold keeps each Verified task together across submissions. A seeded 20% share of the outer training tasks is reserved for calibration. The leave-one-framework-out manifest combines the outer task folds with framework holdout. Each held-out framework is tested only on the held-out tasks for that fold, and the remaining frameworks supply training and calibration on disjoint tasks. Rotating the task folds evaluates every run in each held-out framework exactly once.

### Grouped folds

| Test framework | Fold | Train tasks / runs | Calibration tasks / runs | Test tasks / runs |
| --- | --- | --- | --- | --- |
| All frameworks | 1 | 320 / 3,178 | 80 / 793 | 100 / 994 |
| All frameworks | 2 | 320 / 3,179 | 80 / 793 | 100 / 993 |
| All frameworks | 3 | 320 / 3,183 | 80 / 789 | 100 / 993 |
| All frameworks | 4 | 320 / 3,179 | 80 / 794 | 100 / 992 |
| All frameworks | 5 | 320 / 3,176 | 80 / 796 | 100 / 993 |

### Leave-one-framework-out folds

| Test framework | Fold | Train tasks / runs | Calibration tasks / runs | Test tasks / runs |
| --- | --- | --- | --- | --- |
| OpenHands | 1 | 320 / 1,909 | 80 / 477 | 100 / 398 |
| OpenHands | 2 | 320 / 1,910 | 80 / 477 | 100 / 398 |
| OpenHands | 3 | 320 / 1,911 | 80 / 475 | 100 / 397 |
| OpenHands | 4 | 320 / 1,908 | 80 / 477 | 100 / 395 |
| OpenHands | 5 | 320 / 1,906 | 80 / 478 | 100 / 395 |
| SWE-agent | 1 | 320 / 2,229 | 80 / 556 | 100 / 296 |
| SWE-agent | 2 | 320 / 2,229 | 80 / 556 | 100 / 295 |
| SWE-agent | 3 | 320 / 2,232 | 80 / 554 | 100 / 296 |
| SWE-agent | 4 | 320 / 2,231 | 80 / 557 | 100 / 297 |
| SWE-agent | 5 | 320 / 2,230 | 80 / 558 | 100 / 298 |
| mini-SWE-agent | 1 | 320 / 2,218 | 80 / 553 | 100 / 300 |
| mini-SWE-agent | 2 | 320 / 2,219 | 80 / 553 | 100 / 300 |
| mini-SWE-agent | 3 | 320 / 2,223 | 80 / 549 | 100 / 300 |
| mini-SWE-agent | 4 | 320 / 2,219 | 80 / 554 | 100 / 300 |
| mini-SWE-agent | 5 | 320 / 2,216 | 80 / 556 | 100 / 300 |

The split and leakage tests passed, including once-only held-out coverage and a held-out-data perturbation that leaves candidate discovery unchanged. [Test results](../results/phase34_tests.txt)

### Difficulty bins

Hard tasks have solve rates at most one-third. Medium tasks are above one-third through two-thirds. Easy tasks are above two-thirds. Rates use all observed submissions for descriptive stratification and are kept outside the split manifest and model inputs.

| Difficulty | Tasks | Runs | Resolved |
| --- | --- | --- | --- |
| easy | 245 | 2,436 | 2,149 |
| hard | 159 | 1,577 | 141 |
| medium | 96 | 952 | 506 |

[Fold sizes](../results/fold_sizes.csv) · [Difficulty summary](../results/difficulty_summary.csv)

## Held-out sequential patterns

PrefixSpan discovers candidates separately in the training failures and successes using 5% support, a maximum length of 5 collapsed tokens, and up to 3 intervening tokens between consecutive matches. Each sequence contributes at most one presence count. Candidate discovery and its training support estimates exclude calibration and test tasks. Fisher's two-sided test and difficulty-binned Cochran–Mantel–Haenszel (CMH) use the outer test tasks. Bonferroni allocates equal shares of α = 0.05 to each outer fold and adjusts within each fold's candidate set. Pooled Benjamini–Hochberg (BH) is secondary. A survivor passes Fisher and CMH at the same correction level with matching directions in the same fold. Repeated tokens in a gapped pattern can be separated by intervening actions. A suffix such as `×3+` denotes an adjacent streak in the collapsed sequence.

| Test and adjustment | Significant fold-pattern tests | Distinct patterns | Patterns in multiple folds |
| --- | --- | --- | --- |
| Robust, including task-cluster check | 5 | 4 | 1 |
| Primary, Fisher and CMH in same direction | 15 | 13 | 2 |
| Fisher, Bonferroni | 78 | 72 | 6 |
| CMH, Bonferroni | 106 | 95 | 8 |
| Fisher, BH | 1,188 | 758 | 284 |
| CMH, BH | 1,714 | 1,060 | 423 |
| Both, same direction, BH | 677 | 480 | 141 |

| Fold | Candidates | Fisher Bonferroni | Both Bonferroni, same direction |
| --- | --- | --- | --- |
| 1 | 3,513 | 23 | 4 |
| 2 | 3,730 | 2 | 1 |
| 3 | 3,605 | 0 | 0 |
| 4 | 3,697 | 6 | 6 |
| 5 | 3,579 | 47 | 4 |

The same tasks repeat across submissions, so runs within a task are dependent. The sensitivity check uses a difficulty-adjusted score test with task-clustered variance. The difficulty bins were derived from this same corpus. Patterns summarize full trajectories and therefore can contain events near termination. These associations alone do not establish an early signal.

[All pattern statistics](../results/patterns_all.csv) · [By-framework evidence](../results/patterns_by_framework.csv) · [PrefixSpan paper](https://doi.org/10.1109/ICDE.2001.914830) · [CMH implementation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.StratifiedTable.test_null_odds.html)

Ranked rows prioritize robust patterns, then primary Fisher and CMH survivors, then Fisher-only and exploratory rows, with absolute support difference ranking within each tier. Each direction shows distinct sequences and the fold represented by the row. Odds ratios are oriented toward failure. Intervals are unadjusted Wald intervals with a half-count correction when an informative table has a zero cell.

### Top 10 failure-associated patterns

| Pattern | Fold | Fail support | Success support | Δ pp | Growth | OR [95% CI] | Fisher Bonf / BH | CMH OR | CMH Bonf / BH | Task-cluster Bonf | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `edit → view×3+` | 2 | 9.98% | 1.89% | +8.09 | 5.28 | 5.75 [2.92, 11.33] | 0.000399 / 4.85e-05 | 9.57 | 6.27e-05 / 5.36e-06 | 0.00182 | Robust |
| `edit → edit → edit` | 5 | 30.30% | 14.26% | +16.04 | 2.12 | 2.61 [1.91, 3.57] | 2.45e-05 / 2.03e-05 | 3.62 | 3.69e-06 / 5.33e-07 | 0.18 | Primary |
| `edit → edit → edit → edit` | 5 | 20.05% | 7.76% | +12.28 | 2.58 | 2.98 [2.02, 4.40] | 0.000293 / 4.85e-05 | 3.16 | 0.00804 / 0.000154 | 1 | Primary |
| `edit → edit → edit → edit → edit` | 1 | 16.88% | 5.58% | +11.30 | 3.03 | 3.44 [2.20, 5.36] | 0.000211 / 4.36e-05 | 4.85 | 0.000206 / 1.06e-05 | 0.0868 | Primary |
| `edit → view → search` | 4 | 11.06% | 2.96% | +8.10 | 3.73 | 4.07 [2.29, 7.26] | 0.00587 / 0.000184 | 6.30 | 0.000185 / 9.55e-06 | 0.799 | Primary |
| `view×2 → edit → edit → edit` | 5 | 9.79% | 2.71% | +7.09 | 3.62 | 3.90 [2.14, 7.12] | 0.0409 / 0.000576 | 5.65 | 0.00556 / 0.000125 | 1 | Primary |
| `edit → view → search → view` | 4 | 9.29% | 2.22% | +7.07 | 4.18 | 4.51 [2.34, 8.67] | 0.0167 / 0.000333 | 10.96 | 1.74e-06 / 4.27e-07 | 0.207 | Primary |
| `search → view → edit → view` | 4 | 9.07% | 2.04% | +7.03 | 4.45 | 4.80 [2.44, 9.45] | 0.0183 / 0.000345 | 10.07 | 7.11e-05 / 5.36e-06 | 0.21 | Primary |
| `view → edit → view → run_script` | 4 | 8.41% | 1.48% | +6.93 | 5.67 | 6.10 [2.82, 13.22] | 0.00324 / 0.000136 | 9.53 | 2.45e-05 / 3e-06 | 1 | Primary |
| `view → search → edit → view` | 4 | 7.96% | 1.67% | +6.30 | 4.78 | 5.11 [2.43, 10.72] | 0.0352 / 0.000515 | 8.95 | 0.000113 / 6.72e-06 | 0.533 | Primary |

### Top 5 success-associated patterns

| Pattern | Fold | Fail support | Success support | Δ pp | Growth | OR [95% CI] | Fisher Bonf / BH | CMH OR | CMH Bonf / BH | Task-cluster Bonf | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `run_script:ok` | 4 | 46.90% | 62.78% | -15.88 | 0.75 | 0.52 [0.41, 0.68] | 0.0103 / 0.000249 | 0.27 | 9.12e-07 / 3.4e-07 | 8.65e-06 | Robust |
| `run_test:pass` | 1 | 19.62% | 34.81% | -15.19 | 0.56 | 0.46 [0.34, 0.61] | 0.00153 / 0.000105 | 0.29 | 3.05e-05 / 3.49e-06 | 0.0016 | Robust |
| `run_test:pass×2` | 1 | 7.81% | 18.46% | -10.66 | 0.42 | 0.37 [0.25, 0.56] | 0.011 / 0.000262 | 0.20 | 0.00288 / 7.82e-05 | 0.0184 | Robust |
| `search:ok` | 4 | 59.07% | 71.48% | -12.41 | 0.83 | 0.58 [0.44, 0.75] | 0.987 / 0.00407 | 0.27 | 1.04e-06 / 3.4e-07 | 1.75e-05 | CMH only |
| `run_test:fail` | 1 | 28.48% | 39.42% | -10.94 | 0.72 | 0.61 [0.47, 0.80] | 1 / 0.0116 | 0.41 | 0.0575 / 0.000534 | 0.00615 | Exploratory |

## Length and failure

Pearson correlation between raw step count and the binary failure flag is point-biserial. The pooled within-bin estimate centers both variables by their respective bin means before correlating them.

| Population | Tasks | Runs | Failed | Pearson r |
| --- | --- | --- | --- | --- |
| Overall | 500 | 4,965 | 2,169 | +0.183 |
| Hard | 159 | 1,577 | 1,436 | -0.045 |
| Medium | 96 | 952 | 446 | -0.034 |
| Easy | 245 | 2,436 | 287 | +0.166 |
| Within bin centered | 500 | 4,965 | 2,169 | +0.044 |

[Length correlations](../results/length_confound.csv) · [Full ranked CSV](../results/patterns_top.csv) · [Machine-readable Phase 4 summary](../results/phase4.json)
