# Video assets

Place only final, web-ready video files in this directory. Keep original recordings outside the repository or under `videos/raw/`, which is ignored by Git.

Recommended filenames use lowercase ASCII characters, numbers, and hyphens, for example:

- `teaser-01.mp4`
- `teaser-02.mp4`
- `task-1.mp4`
- `task-2.mp4`

Recommended delivery format:

- MP4 container
- H.264 video
- `yuv420p` pixel format
- no audio track when the clip is silent
- `faststart` enabled
- 1280 px maximum width for ordinary project-page clips
- preferably below 20–25 MiB per file

Example conversion:

```powershell
ffmpeg -i input.mov -an -c:v libx264 -crf 24 -preset slow -pix_fmt yuv420p -movflags +faststart -vf "scale='min(1280,iw)':-2" videos/teaser-01.mp4
```

After a file is present, replace the corresponding placeholder in `index.html` with markup such as:

```html
<video autoplay loop muted playsinline preload="metadata">
  <source src="videos/teaser-01.mp4" type="video/mp4">
</video>
```

Use `controls` instead of autoplay for longer explanatory videos. Poster images, when available, should be placed under `images/video-posters/` and referenced with the `poster` attribute.
