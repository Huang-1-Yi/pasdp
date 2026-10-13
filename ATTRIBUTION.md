# Attribution

This implementation uses original, dependency-free HTML, CSS, and SVG created for this project.

Its section ordering, content width, typography scale, link color, separators, and media-grid proportions were intentionally aligned with the UMI academic project page at <https://umi-gripper.github.io/> as a visual baseline. No source files or media from that site are redistributed here.

## Competition broadcast excerpt

`videos/协作机器人作业任务挑战赛.mp4` is an unedited excerpt from the public live broadcast of the Manufacturing Category — Collaborative Robot Operation Task Challenge at the 2025 Intelligent Robot Competition.

- Original live-stream source: Robot National Evaluation Center, <https://weixin.qq.com/sph/AQb0aK2kGR>
- Related CCTV.com coverage: <https://tv.cctv.com/2025/08/25/VIDE9NT9Bngj5JyfdxjVaKrc250825.shtml>

The broadcast footage, on-screen graphics, and broadcaster marks remain the property of their respective rights holders. The excerpt is included here for project documentation and attribution.

The competition certificate image documents the First Prize awarded to the project team. Other photographs and demonstration videos in this repository are project media. Before adding new media, verify ownership or redistribution terms and record any required attribution here.

## Manuscript figures and result visualizations

- `images/figures/pasdp-method.png` is a raster rendering of the project's `PSADP_Method.pdf` used in the current PASDP manuscript, Fig. 2.
- `images/figures/real-world-tasks.png` is a raster rendering of the project's earlier TRO `Experiments_real.pdf`. Its source PDF is byte-identical to Fig. 4's source in the current manuscript.
- `images/figures/real-world-setup.png` is the project's `Env.png` platform photograph from the earlier TRO manuscript. It illustrates the same Franka Research 3 / two-D405 conveyor platform described in the current manuscript.
- The component-ablation bars and parameter heatmap are original HTML/CSS visualizations of current manuscript Tables V and VI. The 28 parameter-sweep success rates also match the earlier TRO parameter table; the current selection criterion and zero-based position label `w_7` are retained.

The current manuscript is the numeric authority for the eight current result tables. Older performance claims and normalized latency curves are not reproduced; historical appendix studies are not merged into those current results. Source locations, hashes, and verification scope are recorded in `PAPER_ALIGNMENT.md`.

At the author's request, the earlier TRO appendix's observation-step, batch-size, and demonstration-count results are additionally reproduced as separate historical development tables in a default-collapsed block. They are not included in the current manuscript's eight result tables. Their provenance and missing per-study evaluation details are explicitly distinguished in `PAPER_ALIGNMENT.md` and `assets/data/legacy-development-results.json`.
