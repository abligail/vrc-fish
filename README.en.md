<div align="center">
  <img src="banner.jpg" alt="VRChat FISHǃ Auto Fishing Assistant Banner" width="100%" />
</div>

<div align="center">
  <h1>VRChat FISHǃ Auto Fishing Assistant (Assistive)</h1>
  <p>Windows · C++ · OpenCV 4.6.0 · Template Matching · Physics + MPC (Optional ML)</p>
  <p>
    <b>English</b> · <a href="README.md">中文</a>
  </p>
  <p>
    <img alt="Platform" src="https://img.shields.io/badge/Platform-Windows-0078D6?style=flat-square" />
    <img alt="Language" src="https://img.shields.io/badge/Language-C%2B%2B-00599C?style=flat-square" />
    <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.6.0-5C3EE8?style=flat-square" />
    <img alt="License" src="https://img.shields.io/badge/License-GPL--3.0-blue?style=flat-square" />
  </p>
</div>

> Note: This project was originally built as an **assistive tool** for a friend with **motor impairments**, and is shared mainly for **learning/research** (see `LICENSE` for licensing terms).  
> Fishing should be relaxing—please keep it casual and follow VRChat and related service rules/terms.

<details>
  <summary><b>Table of Contents</b></summary>

- [Acknowledgements](#acknowledgements)
- [Overview](#overview)
- [Demo Video](#demo-video)
- [Features](#features)
- [Target World](#target-world)
- [Download](#download)
- [Quick Start](#quick-start)
- [Tuning \& Adaptation](#tuning--adaptation)
- [Build](#build)
- [Configuration](#configuration)
- [Logging \& Debugging](#logging--debugging)
- [Experimental Scripts (Optional)](#experimental-scripts-optional)
- [Project Layout](#project-layout)
- [Support \& Contributing](#support--contributing)
- [Disclaimer](#disclaimer)
- [License](#license)

</details>

---

## Acknowledgements

- Thanks to my good friend **aflotia** (VRChat player) for tuning better parameters and contributing template images

---

## Overview

A small Windows tool that assists the VRChat fishing minigame by detecting key UI elements (OpenCV template matching / simple brightness-based detection) and controlling the minigame via <kbd>Left Mouse Button</kbd> press/release.

This repository currently focuses on the fishing logic, parameters, and experimental scripts for the **VRChat world FISHǃ**.

## Demo Video

3× speed:

<video src="https://github.com/user-attachments/assets/06b362be-6b1d-4f4d-ab5d-a4811a0f8a64" controls></video>

## Features

- Fully automated loop: cast → wait for bite → hook → control minigame → cleanup → next round
- Detection:
  - `matchTemplate`: bite prompt, minigame track, fish icon(s), slider template (fallback)
  - Brightness-based slider boundary detection on the track column (primary)
- Control: a lightweight physics model + MPC (Model Predictive Control) to decide hold/release
- Optional ML workflow: record data / inference mode + analysis & fitting scripts

## Target World

This project is mainly tuned for **FISHǃ**:
- World URL: https://vrchat.com/home/world/wrld_ae001ea3-ed05-42f0-adf2-3d47efd10a77
- World ID: `wrld_ae001ea3-ed05-42f0-adf2-3d47efd10a77`

Templates, thresholds, and ROIs are calibrated for the current UI of this world. If the world/UI updates, re-capture templates under `Resource-VRChat/` and adjust `config.ini`.

## Download

Head to the [Releases](https://github.com/abligail/vrc-fish/releases) page to download the latest pre-built executable.

## Quick Start

1. In VRChat display/graphics settings, set the resolution to `1280×960` (matches the default templates in this repo).
2. Enter **FISHǃ** in VRChat, make sure you are ready to fish and the fishing UI is visible (do not cover it with other windows).
3. After casting, choose a position/view so that the bite indicator (the dot at the bottom of the exclamation mark) and the full minigame slider track are visible on screen (not cropped/occluded).
4. Adjust `config.ini` if needed (window matching, resolution, thresholds, cleanup parameters), then run `vrc-fish.exe`. If `is_pause=1`, press <kbd>Tab</kbd> to pause/resume.

Notes:
- The project enables `RequireAdministrator` by default, so the app may require admin privileges.
- For stable template matching, the app can force VRChat client-area resolution via `force_resolution`.
- The app reads `config.ini` from the current working directory and loads templates from `resource_dir` (default `Resource-VRChat/`). Run it from the repo root, or copy `config.ini` and `Resource-VRChat/` next to the executable.

## Tuning & Adaptation

The bundled templates and default parameters are based on my own fishing spot (the **end of the wooden pier at Coconut Bay**, the starting island, with an avatar height of **1.1 m**). The current parameters can already reliably catch the vast majority of fish. Further optimizations are welcome!

> **Tip**: It is recommended to first go to the same spot described above (end of the wooden pier at Coconut Bay, avatar height ~1.1 m) to verify that the program works correctly in your environment before trying other locations.

If detection is not working well in your setup, adjust in the following priority:

### 1. Adjust Track Template Scale Range (Most Common)

Different positions, avatar heights, fishing rods, and resolution/UI scaling will cause the slider track to appear at different sizes on screen. The program uses multi-scale template matching to handle this. The relevant parameters are:

```ini
track_scale_min=0.8
track_scale_max=2
track_scale_step=0.2
```

**How to adjust**: Run the program with `debug=1` and watch the console log for the `scale` value printed when the track is successfully matched (e.g. `scale=1.4`). Once you know the approximate scale range for your setup, narrow the search range around that value and reduce the step size for better precision. For example, if you observe scale values around `1.2~1.6`, set:

```ini
track_scale_min=1.0
track_scale_max=1.8
track_scale_step=0.1
```

- If the matched scale is close to `track_scale_min` or `track_scale_max`, the search range may be too narrow—expand it in the corresponding direction.
- A smaller `track_scale_step` gives more accurate matching but increases computation. Usually `0.1~0.2` is sufficient.

### 2. Keep the Slider Track as Vertical as Possible

The program assumes the slider track is roughly vertical on screen. If your position/view angle causes noticeable tilt, there are two approaches:

- **Recommended**: Adjust your in-game position or camera angle so the fishing track appears as vertical as possible on screen.
- **Compensate via config**: Enable angle search parameters to let the program detect small rotations automatically:
  ```ini
  track_angle_min=-5.0
  track_angle_max=5.0
  track_angle_step=1.0
  ```
  You can check the `angle` value in the log output to confirm the actual tilt, then narrow the range and reduce the step size for better accuracy. Note that angle search multiplies computation cost, so only enable it when needed.

### 3. Other Adjustments

- **Lots of "miss" in the logs**: If tuning scale/angle doesn't help enough, the most effective fix is to take your own screenshots at your fishing spot, crop the UI elements, and replace the images in `Resource-VRChat/`. You can also tweak matching thresholds (`bite_threshold`, `fish_icon_threshold`, etc.) in `config.ini`.
- **Different PC / display environments**: Detection accuracy is sensitive to screen resolution and rendering settings. You may need to adjust thresholds and control parameters for your specific setup.

## Build

1. Open `vrc-fish.sln` with Visual Studio
2. Select `x64` + `Release` (or `Debug`)
3. Build `vrc-fish.exe`

Dependencies:
- OpenCV 4.6.0: headers/libs are organized under `include/` + `lib/`, and the runtime `opencv_*460.dll` should be placed next to the executable (the repo root already includes these DLLs).
- For VS Code, see `.vscode/tasks.json` (`Build Release`) and adjust paths for your local VS installation.
- Make sure the runtime can locate `config.ini` and `Resource-VRChat/` (set the working directory to the repo root, or copy required files to your output folder).

## Configuration

Config file: `config.ini` (the file contains inline comments, mainly in Chinese).

Key highlights:
- Window matching: `window_class`, `window_title_contains`
- Resolution: `force_resolution`, `target_width`, `target_height`
- After cast: `cast_mouse_move_dx`, `cast_mouse_move_dy` (moves the mouse slightly; restored at end of the round)
- Thresholds: `bite_threshold`, `minigame_threshold`, `fish_icon_threshold`, `slider_threshold`
- Template matching: track lock uses `track_scale_*` / `track_scale_min`/`track_scale_max`/`track_scale_step` / `track_angle_*` (range-based scale+angle scan supported)
- Cleanup loop: `cleanup_*`, `cleanup_reel_key`
- ML: `ml_mode` (0=auto, 1=record, 2=infer), `ml_record_csv`, `ml_weights_file`
- Debug/log: `debug`, `debug_pic`, `debug_dir`, `vr_log_file`

Templates:
- Default directory: `Resource-VRChat/` (override via `tpl_*` keys)
- Fish icons: auto-loads `fish_icon_alt*.png` (e.g. `fish_icon_alt3.png`) without requiring config entries

## Logging & Debugging

- `debug=1`: prints scores, state transitions, and control info to the console
- `debug_pic=1`: saves key-frame screenshots under `debug_dir`
- `vr_log_file`: appends runtime logs to a file (see examples under `data/logs/`)

## Experimental Scripts (Optional)

Scripts live in `scripts/`:

- `fit_physics.py`: fit slider physics parameters from debug logs
- `analyze_log.py` / `analyze_oscillation.py`: analyze overshoot/oscillation behavior
- `train_bc.py`: behavior cloning training (requires `numpy`), outputs weights for `ml_mode=2`

## Project Layout

- `vrc-fish.cpp`: main app (capture, detection, state machine, control, cleanup)
- `config.ini`: configuration
- `Resource-VRChat/`: UI templates for FISHǃ
- `data/`: logs, sample data, ML weights
- `scripts/`: analysis / fitting / training utilities

## Support & Contributing

If you find this project helpful, a Star would be much appreciated!

I have limited time to maintain this regularly, so contributions from anyone interested are very welcome :)

## Disclaimer

- This is NOT an official VRChat project and is not affiliated with VRChat.
- Please follow VRChat and related service terms/rules. Use at your own risk.
- This repository is shared for learning/research purposes with no warranty. Please do not use it to harm others’ experience, violate service terms, or for other improper purposes.
- The above guidance is informational and is not an additional restriction on the GPL-3.0 license terms.

## License

The source code of this project is licensed under GPL-3.0. See `LICENSE`. Third-party components/assets in this repository may be under different licenses or rights notices; see `THIRD_PARTY_NOTICES.md`.
