# StreamDiffusion for TouchDesigner on Apple Silicon

A TouchDesigner image-to-image patch using SDXS-512 and CoreML. The portable TouchDesigner component is `StreamDiffusion.tox`; helper scripts live in `streamdiffusionTD/`.

This repository contains original integration code and a small `.tox` component. It intentionally does **not** contain model weights, converted CoreML packages, Python environments, or unrelated project/MCP-server content.

## Requirements

- macOS on Apple Silicon (tested on an M4)
- TouchDesigner 2025.33230 or newer
- Python 3.12 on Apple Silicon for the external worker (the provided `setup.sh` creates `.coreml.venv` using `uv`)
- About 3.2 GB free for the SDXS Hugging Face snapshot and converted CoreML packages, in addition to the Python environment

## Install and run

1. Clone this repository and install `uv` (`brew install uv` if Homebrew is available).
2. Obtain the three CoreML packages listed below and place them under `coreml_models/` **before** setup if available. Otherwise, run `./setup.sh` once to create `.coreml.venv`, install the pinned packages, and download the SDXS text encoder/tokenizer snapshot to `models/`; it will exit with an explanatory message because the CoreML packages are not included. Add the packages, then run `./setup.sh` again to verify the required files.
3. The required package names are:
   - `unet_sdxs_512.mlpackage`
   - `taesd_encoder_512.mlpackage`
   - `taesd_decoder.mlpackage`
4. Open `StreamDiffusion.tox` in TouchDesigner or drag it into a project. Connect the input TOP to connector 1; connector 2 is reserved for future ControlNet use. The outputs are image (TOP) and inference metrics (CHOP).
5. On the component's `engine` COMP, set **Enabled** and use **Start** if needed. Enter the text prompt in **Prompt**. The Worker page also exposes feedback, input sample rate, fit/crop framing, and gentle output sharpening.

The component resolves `Modelroot` to the containing `.toe` folder. Keep the worker scripts at `streamdiffusionTD/`, the CoreML package directories at `coreml_models/`, and the Hugging Face cache in its normal `models/hub/` layout.

## Model files and licenses

The GitHub repo excludes the model artifacts because they are large and have their own licenses. Download the SDXS model `IDKiro/sdxs-512-0.9` through Hugging Face (the setup script uses `hf download`). The converted CoreML packages were produced from SDXS-512 and TAESD. Follow the model authors' licenses and terms for the original and converted weights; this repo's Apache-2.0 license applies to the original integration code only, not model weights, TouchDesigner, or upstream StreamDiffusion files.

If you already have authorized copies of the CoreML packages, copy the complete `.mlpackage` directories as-is. Each package should contain its `Manifest.json`, `Data/com.apple.CoreML/model.mlmodel`, and weights data. These generated package directories are git-ignored. The conversion commands are specific to the upstream SDXS export and are not included in this repo; do not assume arbitrary Stable Diffusion checkpoints are interchangeable with these packages. The checked-in `.tox` is an export-only sample; a future user should set its **Modelroot** to their cloned repository directory (which contains `streamdiffusionTD/`, `coreml_models/`, `.coreml.venv/`, and `models/`) before enabling the engine.

## Parameters

- **Prompt**: positive text prompt used for generation
- **Feedback**: temporal consistency; higher values smooth changes but can produce trails
- **Input Sample Every**: skip source readbacks; increase to lower CPU/GPU-transfer overhead
- **Input Framing**: `Fit` preserves the whole input with padding; `Center crop` fills the square model input
- **Output Sharpening**: restrained post-sharpen amount; high values amplify artifacts

## Worker details

TouchDesigner runs the network and copies frames to a background helper process. The helper binds only to `127.0.0.1:13317`; the TouchDesigner MCP server port is not used by this component. Do not load CoreML/PyTorch inside TouchDesigner Python.

## Source and attribution

The runtime worker code and `.tox` are the project-specific integration. The upstream StreamDiffusion implementation is at [dotsimulate/StreamDiffusion](https://github.com/dotsimulate/StreamDiffusion) and is Apache-2.0 licensed; its license is retained as `UPSTREAM-LICENSE-Apache-2.0`. SDXS and TAESD weights are separate works with separate licensing.
