# PASDP Project Page — AI Maintenance Guide

> Audience: AI coding agents and future maintainers.  
> Source-code integration verified: 2026-10-13.
> Website results aligned with `PSADP_v29.tex`: 2026-10-13.
> Public-facing documentation remains in `README.md`; do not replace it with this file.

## 1. Project Purpose

This repository contains the official project page for **Position-Aware Streaming Diffusion Policy (PASDP)**, a streaming diffusion policy for real-time robotic manipulation.

- Public site: `https://huang-1-yi.github.io/pasdp/`
- Code directory linked by the page: `https://github.com/Huang-1-Yi/pasdp/tree/main/code`
- Original source repository: `https://gitee.com/huang-1-yi/dp_pro6000`
- Contact: `20241301010@stu.cqu.edu.cn`
- GitHub Pages source: GitHub Actions, `.github/workflows/pages.yml`, from `main`

The site is a dependency-free static website. It uses plain HTML and CSS and has no build step, package manager, JavaScript application, or server-side component. Algorithm source and its own dependencies live separately in `code/`.

## 2. Source of Truth

- `index.html`: all page content, tables, links, video markup, and section order
- `assets/css/site.css`: all layout, typography, colors, cards, tables, and responsive rules
- `videos/`: final browser-ready videos that may be committed
- `images/video-posters/`: video poster images
- `images/`: certificate image, favicon, and other visual assets
- `videos/README.md`: video-format and conversion guidance
- `.gitignore`: excludes original footage and local temporary files
- `.nojekyll`: tells GitHub Pages to serve the static files directly
- `ATTRIBUTION.md`: third-party source and media attribution
- `PAPER_ALIGNMENT.md`: authoritative manuscript, exact source hashes, table mapping, and legacy-asset decisions
- `assets/data/paper-provenance.json`: manuscript-reported snapshot of all eight tables; no local private paths
- `assets/data/legacy-development-results.json`: separately labeled historical TRO appendix results; never mix with the current eight tables
- `tools/check_paper_alignment.py`: read-only table, chart, source-hash, media-link, and autoplay validator
- `README.md`: short English README for human visitors
- `code/README.md`: English installation, training, and evaluation entry point
- `code/README_UPSTREAM.md`: preserved original source README
- `code/SOURCE_IMPORT.md`: source commit, integration changes, and verification scope
- `code/third_party/`: vendored runtime dependencies, including meshes, textures, SDKs, wheels, and upstream licenses

Run algorithm commands with `code/` as the working directory. Keep datasets and generated training outputs out of Git. Do not apply blanket media or `*.obj` exclusions to `code/third_party/`: these files include runtime textures and robot meshes. Preserve the site at the repository root. Importing source does not verify that its defaults reproduce every published experiment; do not update result tables from configuration defaults.

Do not introduce a framework or build tool unless the user explicitly requests one. Preserve relative URLs because the site is served from the `/pasdp/` project path rather than a domain root.

## 3. Current Page Structure

The current section order in `index.html` is:

1. Paper-style title header
   - Title: Position-Aware Streaming Diffusion Policy
   - Subtitle: for Real-Time Robotic Manipulation
   - Authors, affiliations, and corresponding-author contact are populated.
2. Performance highlights
   - Reported metrics: **18 ms simulation policy inference** and **11 ms real-world perception-to-command latency**
   - Three videos: 1×, 2×, and 4× motion
   - The clips illustrate execution-rate settings, not latency ratios or quantitative trial averages.
3. Abstract
4. Paper section — intentionally hidden in an HTML comment
5. Code and Tutorial
6. Team — default-collapsed; click the summary to show the repeated author / affiliation information. The title-header author list remains visible.
7. BibTeX section — intentionally hidden in an HTML comment
8. Method
   - Position-Dependent Corruption
   - Position-Weighted Objective
   - Sliding-Buffer Streaming Inference
9. Experiments
   - Simulation Experiments: 10 Robomimic tasks, Table I, protocol, and collapsible Tables II–IV
   - Real-World Experiments: platform and task figures, Tables VII–VIII, six detailed comparison videos
   - Component Ablations: chart and collapsible full Table V
   - Parameter Sensitivity: development-task heatmap / Table VI and selection criterion
   - Earlier Development Ablations: default-collapsed TRO appendix tables for observation steps, batch size, and demonstration count
10. Competition recognition
11. Acknowledgements — hidden until authoritative text is supplied
12. Contact

Do not expose the Paper or BibTeX sections until the user supplies final paper and citation information.

## 4. Real-World Experiment Mapping

This mapping is intentional and must remain clear in the page:

| Displayed experiment | Physical content in the filenames | Videos shown |
| --- | --- | --- |
| Task 1: Sequential Conveyor Picking | Cup stacking (`叠杯子`) | DP, PASDP 1×, PASDP 4× |
| Task 2: Continuous Pick-and-Place | Jelly grasping (`抓果冻`) | DP, PASDP 1×, PASDP 4× |

The videos are part of the **Real-World Experiments** section. They are not a separate top-level “Comparison Experiments” section. A small “Comparison Experiments” heading introduces the detailed videos after the real-world result tables.

Current comparison assets:

- Task 1 / DP: `videos/对比实验-DP叠杯子.mp4`
- Task 1 / PASDP 1×: `videos/对比实验-PASDP叠杯子-1x.mp4`
- Task 1 / PASDP 4×: `videos/对比实验-PASDP叠杯子-4x.mp4`
- Task 2 / DP: `videos/对比实验-DP抓果冻.mp4`
- Task 2 / PASDP 1×: `videos/对比实验-PASDP抓果冻-1x.mp4`
- Task 2 / PASDP 4×: `videos/对比实验-PASDP抓果冻-4x.mp4`

Do not relabel Task 1 as “Cup Stacking” or Task 2 as “Jelly Grasping” in the visible experiment headings. Those phrases describe the footage, while the formal experiment names above are the intended labels.

## 5. Video Inventory and Behavior

### Videos currently referenced by the page

- Performance: `videos/1x70s.mp4`, `videos/2x50s.mp4`, `videos/4x25s.mp4`
- Detailed real-world comparisons: the six files listed in Section 4
- Competition: `videos/协作机器人作业任务挑战赛.mp4`
- Simulation: ten task-named MP4 clips, from `videos/coffee.mp4` through `videos/three_piece_assembly.mp4`

The page currently contains 20 video players. Every player must retain:

```html
autoplay loop muted playsinline preload="metadata"
```

Keep `controls` on the performance, real-world, and competition players; the compact simulation cards currently omit controls. `muted` is necessary for browser autoplay policies. Every video should also have:

- a browser-compatible MP4/H.264 source;
- a poster image under `images/video-posters/`;
- a useful `aria-label`;
- fallback text inside the `<video>` element.

### Legacy files not currently referenced

These generic demonstrations remain in the repository but are no longer used by `index.html`:

- `videos/杯子堆叠.mp4`
- `videos/果冻抓取.mp4`
- `images/video-posters/cup-stacking.jpg`
- `images/video-posters/jelly-grasping.jpg`

Do not restore them to the page unless the user asks. Do not delete them without explicit approval.

### Compression policy

- Preserve original footage; never overwrite it during conversion.
- Original recordings belong outside the repository or in `videos/raw/` or `videos/originals/`, both ignored by Git.
- Only final web-ready versions belong directly in `videos/`.
- Files below 50 MB should normally be left unchanged unless their bitrate or browser compatibility is clearly unsuitable.
- When conversion is necessary, prefer H.264 MP4, `yuv420p`, `faststart`, no audio for silent clips, and a maximum width around 1280 px.
- Keep the existing filename if replacing an updated clip so `index.html` does not need a cache-breaking path change unless browser caching becomes a real issue.

## 6. Competition Content

The page states that the PASDP-based entry won **First Prize and ranked first** in China’s **2025 Intelligent Robot Competition**, Manufacturing Category — Collaborative Robot Operation Task Challenge.

Required sources and assets:

- Unedited broadcast excerpt: `videos/协作机器人作业任务挑战赛.mp4`
- Award image: `images/制造业类-协作机器人作业任务挑战赛-一等奖-唯一.png`
- Official live-stream source: `https://weixin.qq.com/sph/AQb0aK2kGR`
- CCTV report: `https://tv.cctv.com/2025/08/25/VIDE9NT9Bngj5JyfdxjVaKrc250825.shtml`

Keep source attribution near the competition material and preserve `ATTRIBUTION.md`.

## 7. Visual Design System

The visual direction is a restrained academic project page rather than a marketing-heavy product site.

- Main content width: `60em`
- Body typeface: Source Sans Pro fallback stack using Helvetica and Arial
- Primary accent / title color: dark red `#8c1515`
- Link accent: green `#49bf9d`
- Body and heading colors: neutral grays
- Background: white
- Cards: light borders, rounded corners, subtle shadows
- Video frames: fixed `26 / 15` aspect ratio with `object-fit: contain`
- Desktop video grids: three columns for speed and comparison videos
- Mobile breakpoint: `760px`; major grids collapse to one column
- Real-world platform introduction: the platform/protocol paragraph appears immediately below the section heading, followed by a centered photograph with maximum width 680px. The photograph scales down to fit mobile screens; no side-by-side layout. The separate task-overview figure remains full width.
- Table VI is content-width rather than page-width. Its latency label spans the two leading columns so it cannot inflate the beta column; keep the beta and w_7 columns compact.
- The Earlier Development Ablations details block is content-width, centered, and capped at 46em (or available screen width). Its three tables remain centered. Do not apply this narrower layout to the separate Extended simulation comparisons block.

Maintain the paper-like hierarchy, generous whitespace, and restrained palette. Reuse existing classes before adding new ones.

## 8. Accessibility and HTML Conventions

- Keep `lang="en"`, UTF-8 encoding, viewport metadata, Open Graph text, and the favicon reference.
- Use semantic sections and sequential heading levels where practical.
- Keep `aria-labelledby` or descriptive `aria-label` values on important sections, video groups, metrics, and tables.
- Preserve table captions, row/column headers, and keyboard-scrollable `.table-wrap` containers.
- Add meaningful `alt` text to images and `aria-label` text to videos.
- External links that open a new tab must retain `rel="noopener noreferrer"`.
- Preserve Unicode filenames exactly; do not transliterate existing paths unless all references and Git history implications are handled deliberately.

## 9. Manuscript Alignment and Hidden Content

All website experiment numbers must match `C:\Users\46462\Desktop\TIE(3)\PASDP\PSADP_v29.tex` (see source SHA-256 in `PAPER_ALIGNMENT.md`). Never restore older TRO results simply because a diagram or old table contains them. Do not infer configurations from equal results.

- Table I: common H = 40 / NFE = 1; DP-x₀ 73.8% / 18 ms, SDP-x₀ 26.7% / 18 ms. Other settings may differ.
- Table II: method-dependent settings; DP-ε 75.2% / 83 ms, SDP-ε 75.8% / 74 ms; not a fully matched closed-loop comparison.
- Table III: keep Tₐ = 1 and Tₐ = 8 groups distinct. SDP-ε 49.0% / 24 ms is the former; 0.3% / 21 ms is the latter. These are separately trained checkpoints.
- Table IV: do not confuse PASDP H = 40 / Tₐ = 1 with DP H = 16 / Tₐ = 8.
- Table V: no buffer reuse costs 41 NFE per decision, not one.
- Table VI: Square development task, seed 10000, not the ten-task benchmark. Selected 94.8% differs from the highest 96.6%. The separate-workstation 16 ms timing is not the main-platform 18 ms timing.
- Tables VII–VIII: use perception-to-command latency, not pure policy inference. Task 1 has 20 independent trials per setting; Task 2 accumulates 80 successful grasps and normalizes time and arrivals to ten.
- The top motion demonstrations are illustrative clips and are not substituted for table trial averages.
- The author confirms the earlier TRO appendix development ablations preceded Table VI parameter selection. Keep them in the separate, default-collapsed `earlier-development-ablations` details element. Preserve the historical name PADP and all 14 source rows. Do not label the archived 96.5% as the current benchmark, infer a task or rollout protocol, or equate old execution-mode labels with NFE. Their table sections do not identify per-study task/seeds/episode counts.

Paper, BibTeX, and Acknowledgements remain commented out. Do not invent publication, funding, or citation details. No visible method / ablation placeholders remain. Do not claim the manuscript-alignment check independently validates the raw experiment evidence.

## 10. GitHub Pages and Deployment

The public site is expected to deploy from `main` using `.github/workflows/pages.yml`. The workflow exports only `index.html`, `.nojekyll`, `assets/`, `images/`, `videos/`, `README.md`, and `ATTRIBUTION.md`. Do not package `code/` into the published site: the full combined checkout exceeds the Pages 1 GB site limit. Settings → Pages must use GitHub Actions instead of root-branch publishing. A normal update is:

```powershell
git add .
git commit -m "Describe the site update"
git push origin main
```

Pushing is an external action. An AI agent should modify and validate files locally, then only commit or push when the user asks or clearly authorizes it.

GitHub Pages deployment can take a short time after a push. If an updated video keeps the same filename, a stale browser or CDN cache may temporarily show the previous version; verify with a hard refresh before changing the source path.

## 11. Required AI Workflow

Before editing:

1. Run `git status --short` and preserve unrelated user changes.
2. Inspect the exact section being changed; do not rewrite the whole page for a small request.
3. Check whether an existing class or media asset already solves the need.

After editing:

1. Run `git diff --check`.
2. Confirm every local path referenced by `src`, `poster`, `href`, and image attributes exists where applicable.
3. Confirm all 20 current video players still include `autoplay`, `loop`, `muted`, and `playsinline`; retain controls where already present unless the user deliberately changes playback behavior.
4. Confirm the six comparison videos remain mapped to the two formal task names described in Section 4.
5. Confirm the Paper and BibTeX blocks remain commented out.
6. Preview with `python -m http.server 8000` when layout or media behavior changes.
7. Report what changed and whether the changes were only local or were pushed.
8. Run `python tools/check_paper_alignment.py --paper "C:\Users\46462\Desktop\TIE(3)\PASDP\PSADP_v29.tex"`. All eight tables, 51 rows, chart values, and source hashes must pass. Manually recheck prose, caption definitions, and any numbers embedded in imported figures after a manuscript revision.
9. For archived appendix verification, append `--legacy-paper "C:\Users\46462\Desktop\TRO0502\PADPv0503\PADPv0503.tex"`. This additionally checks all three historical tables / 14 rows against the original TeX and its hash, and ensures the block is collapsed by default.

Avoid destructive Git commands, deleting original media, replacing user work, or recompressing every video by default.

## 12. Current Design Intent in One Sentence

Present PASDP as a concise academic project page that leads with measurable speed improvements, supports the claims with autoplaying muted demonstrations and structured experiment tables, maps detailed comparison footage to the correct formal tasks, and closes with verified competition recognition and contact information.
