# Website / Manuscript Alignment

Verified on 2026-10-13. The website follows the current manuscript; the earlier TRO draft is a supplementary media source, not an alternative numeric authority.

## Authoritative source

- TeX: `C:\Users\46462\Desktop\TIE(3)\PASDP\PSADP_v29.tex`
- Compiled reference: `C:\Users\46462\Desktop\TIE(3)\PASDP\PSADP_v29.pdf`
- TeX SHA-256: `2345124ad9dbcd4a7b55ac7ebccea6e9322819d6f78ea85572704c7b9fbc4702`
- Earlier TRO directory: `C:\Users\46462\Desktop\TRO0502\PADPv0503`
- TRO `PADPv0503.tex` SHA-256: `f690075833f18cc72b0a9eb60233500c4d4d7c88ac258cd60932e982d79f493b`

The supplied TeX and compiled PDF agree on the reviewed result tables. Source manuscripts, logs, configurations, checkpoints, and videos were not edited. The page reproduces manuscript-reported results; this work does not independently audit the underlying experimental evidence or resolve claims in the manuscript itself.

## Table mapping

Each HTML table has a `data-paper-table` attribute matching the TeX label. The public JSON snapshot contains the same values, including prediction targets and inference budgets.

| Paper table | TeX lines | HTML / JSON key | Important qualification |
| --- | --- | --- | --- |
| I | 523–553 | `main_simulation_comparison` | H = 40, NFE = 1; other settings may differ |
| II | 562–587 | `method_configurations` | Method-dependent settings; not fully matched closed-loop comparisons |
| III | 595–630 | `single_eval_operating_points` | Separate T_a = 1 / 8 groups; SDP uses separately trained checkpoints |
| IV | 644–693 | `inference_budget_sweep` | PASDP H = 40 / T_a = 1 versus DP H = 16 / T_a = 8 |
| V | 698–727 | `component_ablation` | No buffer reuse costs 41 NFE per decision |
| VI | 747–771 | `beta_h_selection` | Square development sweep, seed 10000; not ten-task average |
| VII | 806–835 | `real_world_task1` | 20 trials per setting; reach rate and conditional mean stage time |
| VIII | 844–866 | `real_world_task2` | Continuous operation to 80 successful grasps; normalize to ten |

The website contains all 51 table rows, including the development-sweep latency row. Successes and latencies are copied from the manuscript independently; no configuration is inferred from shared numeric values. Table abbreviations use the same ten-task order as the paper.

## Numeric and wording corrections

- Removed the old common-H DP 75.9% / 261 ms row; current Table I uses DP-x0 73.8% / 18 ms, with Pick Place 4% and Nut Assembly 48%.
- Current Table I's SDP-x0 row retains 26.7% but uses 18 ms, not the website's old 11 ms.
- The iterative DP-epsilon 75.2% / 83 ms operating point remains separate in Table II and the abstract.
- Replaced unsupported generic 7x action-generation and 4x+ execution headlines with the reported 18 ms / 11 ms metrics and explicit 1x / 2x / 4x execution-rate settings. The example clip filenames/durations are not quantitative trial averages.
- Real-world latency is image acquisition to command issuance, including preprocessing and inference and excluding subsequent robot execution.
- Replaced the old Task 2 "best uninterrupted sequence" wording with the current continuous-operation / normalized-ten-success definition.
- Parameter selection uses zero-based w_7 = max(exp(-7 beta), 0.02). Selected beta = 0.2 / H = 40 / 94.8% meets w_7 > 0.2; highest observed 96.6% at beta = 0.25 fails that criterion.
- Development timing of 16 ms is batch-one on a separate RTX 4090 workstation, not directly comparable with the main-platform 18 ms. No horizon-independent computation claim is made.
- Simulation protocol: 200 demonstrations / task, 250 epochs, batch size 64, training seed 42, 30 checkpoint-selection validation seeds 100000–100029, and 100 final test seeds 20000–20099. Single-seed estimates are not confidence intervals.

## Figures and legacy material

| Website asset | Original source | Decision |
| --- | --- | --- |
| `images/figures/pasdp-method.png` | Current `PSADP_Method.pdf` | Use the current method explanation / Fig. 2 |
| `images/figures/real-world-tasks.png` | TRO `Experiments_real.pdf` | Use; byte-identical to current Fig. 4 source |
| `images/figures/real-world-setup.png` | TRO `Env.png` | Use as compatible platform photograph |
| HTML component-ablation bars | Current Table V | Exact manuscript values, NFE shown alongside each bar |
| HTML parameter heatmap | Current Table VI / TRO parameter table | All 28 success-rate cells agree; use current caption, selection, and timing qualifications |

Source hashes:

- Current method PDF: `38048aaa45ce5c2d9cc70e568f9c5897e62629bddd912213c05724d6990c6dd1`
- Both task PDFs: `11adc0717fd09b85d7f0904bd63312fe3fbebdddb36b4755f18d87d966f4c5535`
- TRO platform PNG: `ce5556e051c34ce8cd49c31b452ee8e576bd4070b3216e66989b6e5564428841`

Not imported into the current result tables: older method overviews with embedded 261 ms / legacy speed claims, the older normalized latency-vs-horizon curve and O(1)-in-H assertions, or appendix ablations as if they followed the current benchmark protocol. No values are taken from unrelated third-party papers in the TRO folder.

## Separately archived preliminary development studies

The author confirmed that the original TRO appendix ablations were completed before the parameter selection in current Table VI. They are now included below Table VI in a default-collapsed `details` block, not merged into Tables I–VIII or their provenance dataset.

| Original TRO label | Source lines | Website label | Rows |
| --- | --- | --- | --- |
| `obs_steps_ablation` | 1536–1555 | `legacy_obs_steps_ablation` | 4 |
| `batch_size_ablation` | 1564–1582 | `legacy_batch_size_ablation` | 5 |
| `data_size_ablation` | 1591–1611 | `legacy_data_size_ablation` | 5 |

Their separate snapshot is `assets/data/legacy-development-results.json`, using the original TRO TeX hash above. Historical method name PADP, observation/execution settings, all 14 data rows, and the DP 100-training-timestep / 20-inference-step qualification are retained. The original batch caption's `weight=exp(0.2)` notation is quoted rather than silently reinterpreted.

The appendix tables do not individually identify a task, evaluation seed list, or evaluation episode count. General training settings elsewhere in the old draft are not sufficient to infer the per-study rollout protocol. Accordingly, 96.5% is not presented as a current ten-task macro-average or equated with Table VI's separately trained Square result of 94.8%. The chronological relationship is author-provided, not inferred from shared numbers. The development studies describe tested choices rather than a universal optimum.

## Maintenance and deployment

```powershell
python tools/check_paper_alignment.py --paper "C:\Users\46462\Desktop\TIE(3)\PASDP\PSADP_v29.tex"
python tools/check_paper_alignment.py --paper "C:\Users\46462\Desktop\TIE(3)\PASDP\PSADP_v29.tex" --legacy-paper "C:\Users\46462\Desktop\TRO0502\PADPv0503\PADPv0503.tex"
git diff --check
```

The validator reads the supplied TeX directly and checks all eight current HTML tables, the JSON snapshot, ablation-chart values and NFE, source hash, local media links, hidden Paper/BibTeX, and all 20 muted autoplay players. It separately checks the archived appendix against its snapshot and default-collapsed state; providing `--legacy-paper` additionally checks the 14 legacy rows and original source hash. It intentionally fails after a manuscript source change until the site/provenance are reviewed. Manual review is still required for prose and numbers embedded in imported figures.

Paper, BibTeX, and Acknowledgements stay commented out; no final publication or funding information was invented. Competition recognition is separate project material based on user-provided sources, not a paper experiment.

The existing GitHub Actions workflow publishes `assets/` and `images/`, including the new figures and public JSON. It does not publish `code/`, this internal record, or the validator. Changes are local until the user commits and pushes.

Browser QA limitation: the in-app preview failed to bind its browser session. Static data/link checks passed, and PDF/PNG figure content was visually reviewed; desktop/mobile browser rendering has not been verified in this session.

Additional checks passed: all 28 legacy parameter-sweep cells agree with Table VI; Table I / V macro-averages match their ten task values; read-only regression mutations of DP 73.8% to legacy 75.9% or Table II's 75.2% are detected as mismatches. The local homepage, stylesheet, public provenance JSON, and all three added figures return HTTP 200. The three figures add approximately 2.23 MiB; video assets are unchanged.
