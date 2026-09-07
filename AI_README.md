# PASDP Project Page — AI Maintenance Guide

> Audience: AI coding agents and future maintainers.  
> Last verified: 2026-09-07.  
> Public-facing documentation remains in `README.md`; do not replace it with this file.

## 1. Project Purpose

This repository contains the official project page for **Position-Aware Streaming Diffusion Policy (PASDP)**, a streaming diffusion policy for real-time robotic manipulation.

- Public site: `https://huang-1-yi.github.io/pasdp/`
- Code repository linked by the page: `https://gitee.com/huang-1-yi/dp_pro6000.git`
- Contact: `20241301010@stu.cqu.edu.cn`
- GitHub Pages source: repository root of the `main` branch

The site is a dependency-free static website. It uses plain HTML and CSS and has no build step, package manager, JavaScript application, or server-side component.

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
- `README.md`: short English README for human visitors

Do not introduce a framework or build tool unless the user explicitly requests one. Preserve relative URLs because the site is served from the `/pasdp/` project path rather than a domain root.

## 3. Current Page Structure

The current section order in `index.html` is:

1. Paper-style title header
   - Title: Position-Aware Streaming Diffusion Policy
   - Subtitle: for Real-Time Robotic Manipulation
   - Author and affiliation placeholders remain intentionally visible.
2. Performance highlights
   - Claims: **7× faster action generation** and **4×+ faster execution**
   - Three videos: 1×, 2×, and 4× motion
3. Abstract
4. Paper section — intentionally hidden in an HTML comment
5. Code and Tutorial
6. Team
7. BibTeX section — intentionally hidden in an HTML comment
8. Method
   - Position-Dependent Corruption
   - Position-Weighted Objective
   - Sliding-Buffer Streaming Inference
9. Experiments
   - Simulation Experiments: 10 Robomimic tasks and summary tables
   - Real-World Experiments: two result tables and six detailed comparison videos
   - Parameter Sensitivity placeholder
10. Competition recognition
11. Acknowledgements
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

The page currently contains 10 video players. Every player must retain:

```html
controls autoplay loop muted playsinline preload="metadata"
```

`muted` is necessary for browser autoplay policies. Every video should also have:

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

Maintain the paper-like hierarchy, generous whitespace, and restrained palette. Reuse existing classes before adding new ones.

## 8. Accessibility and HTML Conventions

- Keep `lang="en"`, UTF-8 encoding, viewport metadata, Open Graph text, and the favicon reference.
- Use semantic sections and sequential heading levels where practical.
- Keep `aria-labelledby` or descriptive `aria-label` values on important sections, video groups, metrics, and tables.
- Preserve table captions, row/column headers, and keyboard-scrollable `.table-wrap` containers.
- Add meaningful `alt` text to images and `aria-label` text to videos.
- External links that open a new tab must retain `rel="noopener noreferrer"`.
- Preserve Unicode filenames exactly; do not transliterate existing paths unless all references and Git history implications are handled deliberately.

## 9. Known Placeholders

The following content is intentionally unfinished:

- Authors
- Affiliations
- Tutorial
- Method figure
- Parameter-sensitivity / ablation figure
- Paper link and preview, currently hidden
- BibTeX entry, currently hidden

Do not invent these details. Leave the placeholders or hidden sections intact until the user provides authoritative content.

## 10. GitHub Pages and Deployment

The public site is expected to deploy from the `main` branch and repository root. A normal update is:

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
3. Confirm all 10 current video players still include `controls`, `autoplay`, `loop`, `muted`, and `playsinline` unless the user deliberately changes playback behavior.
4. Confirm the six comparison videos remain mapped to the two formal task names described in Section 4.
5. Confirm the Paper and BibTeX blocks remain commented out.
6. Preview with `python -m http.server 8000` when layout or media behavior changes.
7. Report what changed and whether the changes were only local or were pushed.

Avoid destructive Git commands, deleting original media, replacing user work, or recompressing every video by default.

## 12. Current Design Intent in One Sentence

Present PASDP as a concise academic project page that leads with measurable speed improvements, supports the claims with autoplaying muted demonstrations and structured experiment tables, maps detailed comparison footage to the correct formal tasks, and closes with verified competition recognition and contact information.
