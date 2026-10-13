# Source Import Record

- Import date: 2026-10-13 (Asia/Shanghai).
- Source repository: https://gitee.com/huang-1-yi/dp_pro6000
- Source branch: `master`.
- Source commit: `a55e81143c002f326778a04cd3806542b68253f8`.
- Source commit date: 2026-10-13 17:51:59 +08:00.
- Source commit subject: `feat: 新增 AUBO v4 限位防抖与自动采集路径`.
- Snapshot: 4,275 tracked files, approximately 924.6 MiB uncompressed.
- Destination: `code/` within the PASDP website repository.

## Preservation

The source was exported from this exact Git commit, without `.git` or its history. Algorithm files, experiment configurations, scripts, vendored dependencies, installation wheels, SDKs, robot model resources, and upstream licenses were preserved. No source experiment settings were changed.

Two source files were renamed to retain their content while adding integration documentation:

| Original source path | Imported path |
| --- | --- |
| `README.md` | `README_UPSTREAM.md` |
| `.gitignore` | `.gitignore.upstream` |

All 4,275 original file contents and Git modes match the source commit, including 20 executable files. Automatic line-ending conversion is disabled inside `code/` to preserve upstream asset bytes and shell-script line endings across platforms.

New files: `README.md`, `SOURCE_IMPORT.md`, `.gitignore`, and `.gitattributes`. The new ignore rules cover local environments and generated data/results while retaining the original source assets. The original ignore rules remain available for reference and are not active.

## Website integration

The repository-root website, CSS, images, and videos retain their paths. Its code and guide links point to this GitHub directory. The root README and AI maintenance guide describe the combined structure.

The complete source and website assets together exceed the 1 GB published-site limit. `.github/workflows/pages.yml` exports only committed website paths to the Pages artifact; `code/` is available through GitHub, not copied into the deployed website. Set **Settings → Pages → Build and deployment → Source → GitHub Actions** when publishing this integration.

## Verification scope

Verify every imported source file against its source Git blob hash, accounting for the two renames. Also check local website asset references, Pages packaging paths, ignored artifact samples, preserved third-party fixtures, and `git diff --check`.

The working-tree and staged-source hash/mode comparisons passed. HTML asset and guide links passed; all 20 website videos retain autoplay, loop, muted, and inline playback. The Pages workflow parsed successfully. Whitespace checks apply to the integration edits and new documentation; inherited whitespace in the unchanged source snapshot is intentionally retained.

The import does not establish that the current development snapshot reproduces all paper tables. No training, robot motion, checkpoint loading, or simulation evaluation was run as part of this integration. Existing server jobs and source repositories were not changed.
