# Position-Aware Streaming Diffusion Policy

Official project page and implementation for **Position-Aware Streaming Diffusion Policy (PASDP)**, a streaming diffusion policy for real-time robotic manipulation.

## Project Website

**[Visit the PASDP project page](https://huang-1-yi.github.io/pasdp/)**

The website presents the method, simulation results, real-world experiments, and video demonstrations of accelerated robot execution, cup stacking, and jelly pouch grasping.

## Highlights

- **77.0%** macro-average success across ten Robomimic tasks with **18 ms** policy inference
- Approximately **11 ms** real-world perception-to-command latency (image acquisition through command issuance)
- Sequential picking-and-stacking at **2× and 4×** execution-rate settings, completing all four stages in **100%** of 20 trials per setting
- Separate common-H / NFE comparisons and method-dependent configurations, without treating them as fully matched closed-loop settings
- Real-world conveyor-manipulation demonstrations
- Qualitative DP-versus-PASDP comparison experiments for cup stacking and jelly grasping
- **First Prize (1st place)** in the Manufacturing Category — Collaborative Robot Operation Task Challenge at the 2025 Intelligent Robot Competition

## Reported Results and Provenance

The page's eight result tables follow the current `PSADP_v29.tex` manuscript. Component ablations, the Square-task parameter sweep, evaluation protocols, and latency definitions are included. Figures from the earlier TRO manuscript are reused only where compatible with the current manuscript; its legacy baseline numbers and speed claims are not carried over.

These are manuscript-reported results, not an independent audit of raw experiment logs. See [the alignment record](PAPER_ALIGNMENT.md) and [the machine-readable table snapshot](assets/data/paper-provenance.json).

The earlier TRO appendix's observation-history, batch-size, and demonstration-budget studies are available in a default-collapsed **Earlier Development Ablations** block below Table VI. These author-confirmed preliminary studies precede that parameter selection and retain a [separate historical snapshot](assets/data/legacy-development-results.json); they are not current benchmark results.

A read-only consistency check is available:

```powershell
python tools/check_paper_alignment.py --paper "C:\Users\46462\Desktop\TIE(3)\PASDP\PSADP_v29.tex"
```

The checker compares all table rows directly with the supplied manuscript, validates the ablation chart, checks the source hash, and checks local media references and muted autoplay settings. It does not establish that the reported experiments used the claimed configurations. Paper and BibTeX links remain hidden until final publication details are available.

## Competition Recognition

We used PASDP in the **Manufacturing Category — Collaborative Robot Operation Task Challenge** at the **2025 Intelligent Robot Competition** in China. Our team received **First Prize and ranked first**.

An unedited excerpt from the official live stream is included at [`videos/协作机器人作业任务挑战赛.mp4`](videos/协作机器人作业任务挑战赛.mp4).

The competition was also reported and broadcast by CCTV.com, CCTV, Lecture Hall of Robotics, and other Chinese media outlets.

- [Official live-stream source — Robot National Evaluation Center](https://weixin.qq.com/sph/AQb0aK2kGR)
- [CCTV coverage](https://tv.cctv.com/2025/08/25/VIDE9NT9Bngj5JyfdxjVaKrc250825.shtml)

## Repository Structure

```text
.
├── index.html              # Main project page
├── assets/css/site.css     # Page styles and responsive layout
├── images/                 # Figures, icons, and video posters
├── videos/                 # Web-ready demonstration videos
├── code/                   # Algorithm source, configs, and vendored dependencies
└── .nojekyll               # Serves the site directly with GitHub Pages
```

The website is a dependency-free static site built with HTML and CSS. No build step is required.

## Local Preview

Run a local web server from the repository root:

```powershell
python -m http.server 8000
```

Then open <http://localhost:8000/> in a browser.

## Deployment with GitHub Pages

The site is published from `main` by [`.github/workflows/pages.yml`](.github/workflows/pages.yml). The workflow packages only the website assets, leaving the large `code/` tree in GitHub. After committing the changes, deploy the latest version with:

```powershell
git push origin main
```

For this combined code-and-website repository, open **Settings → Pages → Build and deployment**, and choose **GitHub Actions** as the source. This is also required when switching from the previous root-branch deployment. After pushing, check the **Deploy project page** workflow in Actions.

For a fork, the public URL normally follows this pattern:

```text
https://<github-username>.github.io/<repository-name>/
```

## Video Assets

The videos use browser-compatible MP4/H.264 encoding with poster images. All 20 players autoplay muted and loop; browsers may begin streaming video data immediately despite the metadata preload hint.

Additional preparation and naming guidance is available in [`videos/README.md`](videos/README.md).

## Codebase

The implementation is included in this repository:

- [Source code](code/)
- [Installation, training, and evaluation guide](code/README.md)
- [Import provenance](code/SOURCE_IMPORT.md)

Run algorithm commands from `code/` so dataset paths and local third-party dependencies resolve correctly. Datasets, training outputs, checkpoints, and local environments are excluded from Git.

The source snapshot was imported from the [original Gitee repository](https://gitee.com/huang-1-yi/dp_pro6000) without its Git history. Vendored dependencies retain their upstream licenses.

## License and Attribution

Only project-owned or appropriately licensed media should be added to this repository. See [`ATTRIBUTION.md`](ATTRIBUTION.md) for attribution information.
