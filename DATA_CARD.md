# SWE-bench Verified data card

## Sources

The pilot uses the same Claude 4 Sonnet release across three single-attempt SWE-bench Verified submissions. Each sample consists of the same eight tasks. The pilot was seeded from the sorted intersection of trajectory IDs with eligible evaluation labels. The balance check prompted a shared swap from `django__django-12125` to `pydata__xarray-6992` so every framework has at least three runs in each class. The pilot is a schema and join check. It does not estimate benchmark performance or cover framework version differences.

| Framework | Resolved pilot runs | Unresolved pilot runs | Sample joins |
|---|---:|---:|---:|
| [SWE-agent][swe] | 4 | 4 | 8/8 |
| [OpenHands][oh] | 5 | 3 | 8/8 |
| [mini-SWE-agent][mini] | 3 | 5 | 8/8 |

Sample joins are checked by `make inspect`. The submission manifest pins the experiments commit and the public object keys for each raw trace.

## Expanded cohort

The expanded run count, step count and exclusions come from [Phase 2 checks](results/normalization_checks.csv). Resolve rates use the official count of resolved instances over the full Verified roster. Observed failure rates use the eligible, successfully parsed runs in each submission. The rate gap compares observed failure to one minus official resolution.

| Submission | Framework | Base model | Official resolved | Normalized runs | Observed failure | Rate gap |
|---|---|---|---:|---:|---:|---:|
| [20250522_sweagent_claude-4-sonnet-20250514](https://github.com/SWE-bench/experiments/tree/40f164d5b8f1d249bf95a6df8b74b577fd8e519d/evaluation/verified/20250522_sweagent_claude-4-sonnet-20250514) | SWE-agent | claude-4-sonnet-20250514 | 333/500 (66.6%) | 500 | 33.40% | 0.00 pp |
| [20240620_sweagent_claude3.5sonnet](https://github.com/SWE-bench/experiments/tree/40f164d5b8f1d249bf95a6df8b74b577fd8e519d/evaluation/verified/20240620_sweagent_claude3.5sonnet) | SWE-agent | claude-3-5-sonnet-20241022 | 168/500 (33.6%) | 486 | 65.43% | 0.97 pp |
| [20250511_sweagent_lm_32b](https://github.com/SWE-bench/experiments/tree/40f164d5b8f1d249bf95a6df8b74b577fd8e519d/evaluation/verified/20250511_sweagent_lm_32b) | SWE-agent | Qwen 2.5 | 201/500 (40.2%) | 496 | 59.48% | 0.32 pp |
| [20250524_openhands_claude_4_sonnet](https://github.com/SWE-bench/experiments/tree/40f164d5b8f1d249bf95a6df8b74b577fd8e519d/evaluation/verified/20250524_openhands_claude_4_sonnet) | OpenHands | claude-4-sonnet-20250514 | 352/500 (70.4%) | 499 | 29.46% | 0.14 pp |
| [20241029_OpenHands-CodeAct-2.1-sonnet-20241022](https://github.com/SWE-bench/experiments/tree/40f164d5b8f1d249bf95a6df8b74b577fd8e519d/evaluation/verified/20241029_OpenHands-CodeAct-2.1-sonnet-20241022) | OpenHands | claude-3-5-sonnet-20241022 | 265/500 (53.0%) | 493 | 46.25% | 0.75 pp |
| [20250520_openhands_devstral_small](https://github.com/SWE-bench/experiments/tree/40f164d5b8f1d249bf95a6df8b74b577fd8e519d/evaluation/verified/20250520_openhands_devstral_small) | OpenHands | mistralai/Devstral-Small-2505 | 234/500 (46.8%) | 492 | 52.44% | 0.76 pp |
| [20250807_openhands_gpt5](https://github.com/SWE-bench/experiments/tree/40f164d5b8f1d249bf95a6df8b74b577fd8e519d/evaluation/verified/20250807_openhands_gpt5) | OpenHands | openai/gpt-5-2025-08-07 | 359/500 (71.8%) | 499 | 28.06% | 0.14 pp |
| [20250726_mini-v1.0.0_claude-sonnet-4-20250514](https://github.com/SWE-bench/experiments/tree/40f164d5b8f1d249bf95a6df8b74b577fd8e519d/evaluation/verified/20250726_mini-v1.0.0_claude-sonnet-4-20250514) | mini-SWE-agent | claude-4-sonnet-20250514 | 324/500 (64.8%) | 500 | 35.20% | 0.00 pp |
| [20250726_mini-v1.0.0_gemini-2.5-pro](https://github.com/SWE-bench/experiments/tree/40f164d5b8f1d249bf95a6df8b74b577fd8e519d/evaluation/verified/20250726_mini-v1.0.0_gemini-2.5-pro) | mini-SWE-agent | gemini-2.5-pro | 268/500 (53.6%) | 500 | 46.40% | 0.00 pp |
| [20250726_mini-v1.0.0_o3-2025-04-16](https://github.com/SWE-bench/experiments/tree/40f164d5b8f1d249bf95a6df8b74b577fd8e519d/evaluation/verified/20250726_mini-v1.0.0_o3-2025-04-16) | mini-SWE-agent | o3-20250416 | 292/500 (58.4%) | 500 | 41.60% | 0.00 pp |

The [exclusion log](results/normalization_skips.csv) records known evaluation exclusions, absent traces and parse failures by instance. The Verified roster controls inclusion even when upstream directories contain extra JSON objects.

## Cost and validation

Character cost counts the model output and paired observations available in each export. SWE-agent uses its response field when present. OpenHands counts assistant text once per message, plus structured tool arguments and replies. mini-SWE-agent counts assistant text and the shell observation, including warnings. Recorded SWE-agent API token totals are retained at run level. Missing step or run token values remain null.

The validation sample is stratified by framework and is for human annotation. An `ok` outcome records observed execution success. It does not imply that the overall task was resolved.

## Labels and trace identity

- SWE-agent and OpenHands use the official `resolved` array. Within the Verified task roster, resolution gives `failed=0` and an unresolved evaluated run gives `failed=1`. Tasks explicitly flagged in `no_generation` or `no_logs` are excluded from the eligible sample. OpenHands flags `psf__requests-1142` as having no evaluation logs even though the trajectory inventory includes that task.
- mini-SWE-agent uses the official `per_instance_details.json` map. `resolved: true` gives `failed=0` and `resolved: false` gives `failed=1`. In the Claude 4 Sonnet submission, metadata reports 64.93% while the per-task file contains 324 resolved among 500 records. The table reports the count-based rate. No alternate denominator has been assumed.
- For SWE-agent, the filename is checked against the embedded task ID when `environment` names an instance. The legacy export uses `swe_main` as an environment label. For mini-SWE-agent, it agrees with `instance_id`. In OpenHands, the exported message array has no top-level ID field, so the published object key supplies the task ID.

## Access

The upstream repository lists the submissions, and their metadata points to publicly readable S3 objects. A specific redistribution license for these raw traces has not been established. Raw files are kept in the ignored local cache. The project includes short excerpts for schema inspection.

[swe]: https://github.com/SWE-bench/experiments/tree/40f164d5b8f1d249bf95a6df8b74b577fd8e519d/evaluation/verified/20250522_sweagent_claude-4-sonnet-20250514
[oh]: https://github.com/SWE-bench/experiments/tree/40f164d5b8f1d249bf95a6df8b74b577fd8e519d/evaluation/verified/20250524_openhands_claude_4_sonnet
[mini]: https://github.com/SWE-bench/experiments/tree/40f164d5b8f1d249bf95a6df8b74b577fd8e519d/evaluation/verified/20250726_mini-v1.0.0_claude-sonnet-4-20250514
