# Animation Frames

This folder should contain your WebP animation frames for the scroll-triggered animation.

## Expected File Naming

Your animation frames should be named using this pattern:
- `frame-000.webp`
- `frame-001.webp`
- `frame-002.webp`
- ...
- `frame-099.webp` (for 100 frames total)

## Configuration

The animation configuration is set in `script.js`:
- **Frame Path**: `animation/`
- **Frame Pattern**: `frame-{frame}.webp`
- **Total Frames**: 100 (update this in `script.js` if you have a different number)

## How It Works

1. As the user scrolls down the page, the animation advances through frames
2. As the user scrolls up, the animation reverses through frames
3. The animation is tied to scroll position, not time-based

## Adding Your Frames

1. Export your animation sequence as individual WebP files
2. Name them according to the pattern above (with 3-digit zero-padding)
3. Place all frames in this `animation/` folder
4. Update `totalFrames` in `script.js` to match your actual frame count

