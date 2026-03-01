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

- [Overview](#overview)
- [Demo Video](#demo-video)
- [Features](#features)
- [Target World](#target-world)
- [Download](#download)
- [Quick Start](#quick-start)
- [Tuning & Adaptation](#tuning--adaptation)
- [Build](#build)
- [Configuration](#configuration)
- [Logging \& Debugging](#logging--debugging)
- [Experimental Scripts (Optional)](#experimental-scripts-optional)
- [Project Layout](#project-layout)
- [Support & Contributing](#support--contributing)
- [Disclaimer](#disclaimer)
- [License](#license)

</details>

---

## Overview

A small Windows tool that assists the VRChat fishing minigame by detecting key UI elements (OpenCV template matching / simple brightness-based detection) and controlling the minigame via <kbd>Left Mouse Button</kbd> press/release.

This repository currently focuses on the fishing logic, parameters, and experimental scripts for the **VRChat world FISHǃ**.

## Demo Video

3× speed:

<video src="https://github.com/user-attachments/assets/9490c80f-6a6e-45b1-b329-977a9c3e77ec" controls></video>

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

The bundled templates and default parameters are based on my own fishing spot (the **end of the wooden pier at Coconut Bay**, the starting island, with an avatar height of **1.1 m**). Different positions, avatars, and fishing rods may require adjustments:

- **Lots of "miss" in the logs**: The most effective fix is to take your own screenshots at your fishing spot, crop the UI elements, and replace the images in `Resource-VRChat/`. You can also tweak the matching thresholds (`bite_threshold`, `fish_icon_threshold`, etc.) and scale parameters (`fish_scale_*`, `track_scale_*`) in `config.ini`.
- **Fish above Green rarity**: Tracking of fast-moving fish (Blue, Purple, and other rare fish) is still being improved. The repo currently only includes icon templates for Green, Purple, and White fish—other colors are more likely to lose tracking. Contributions of additional fish icon templates are welcome.
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
- Thresholds: `bite_threshold`, `minigame_threshold`, `fish_icon_threshold`, `slider_threshold`
- Multi-scale template matching: `fish_scale_*` (fish icons), `track_scale_*` (minigame track)
- Cleanup loop: `cleanup_*`, `cleanup_reel_key`
- ML: `ml_mode` (0=auto, 1=record, 2=infer), `ml_record_csv`, `ml_weights_file`
- Debug/log: `debug`, `debug_pic`, `debug_dir`, `vr_log_file`

Templates:
- Default directory: `Resource-VRChat/` (override via `tpl_*` keys)

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
