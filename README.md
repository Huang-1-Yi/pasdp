# Position-Aware Streaming Diffusion Policy

Official project page for **Position-Aware Streaming Diffusion Policy (PASDP)**, a streaming diffusion policy for real-time robotic manipulation.

## Project Website

**[Visit the PASDP project page](https://huang-1-yi.github.io/pasdp/)**

The website presents the method, simulation results, real-world experiments, and video demonstrations of accelerated robot execution, cup stacking, and jelly pouch grasping.

## Highlights

- Up to **7× faster action generation**
- More than **4× faster robot execution**
- Approximately **11 ms** policy-inference latency in the reported real-world experiments
- Competitive performance across 10 Robomimic simulation tasks
- Real-world conveyor-manipulation demonstrations

## Repository Structure

```text
.
├── index.html              # Main project page
├── assets/css/site.css     # Page styles and responsive layout
├── images/                 # Figures, icons, and video posters
├── videos/                 # Web-ready demonstration videos
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

The site is published from the `main` branch and the repository root. After committing the changes, deploy the latest version with:

```powershell
git push origin main
```

If GitHub Pages has not been enabled for a fork of this repository, open **Settings → Pages**, select **Deploy from a branch**, and choose the `main` branch with the `/ (root)` directory.

For a fork, the public URL normally follows this pattern:

```text
https://<github-username>.github.io/<repository-name>/
```

## Video Assets

The videos use browser-compatible MP4/H.264 encoding. The page loads poster images and video metadata first, so visitors do not download every full video when the page opens.

Additional preparation and naming guidance is available in [`videos/README.md`](videos/README.md).

## Codebase

The associated implementation is available at:

- [PASDP codebase on Gitee](https://gitee.com/huang-1-yi/dp_pro6000.git)

## License and Attribution

Only project-owned or appropriately licensed media should be added to this repository. See [`ATTRIBUTION.md`](ATTRIBUTION.md) for attribution information.
