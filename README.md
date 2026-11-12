# ComfyUI-Viggle-Remake-Extended

An experimental ComfyUI custom node that adds wider input bounds to Viggle Remake's `RemakeConditioning` node. It registers under the separate class type `RemakeConditioningExtended`; it does not overwrite the official node. The repository also includes an optional self-contained Colab inference cell.

## Install in Colab

Clone this public repository directly into ComfyUI's custom-node folder:

```python
%cd /content/ComfyUI/custom_nodes
!git clone https://github.com/Yuvrajxms09/ComfyUI-Viggle-Remake-Extended.git
```

For an existing clone, update it with:

```python
%cd /content/ComfyUI/custom_nodes/ComfyUI-Viggle-Remake-Extended
!git pull --ff-only origin main
```

Change `/content/ComfyUI` if your ComfyUI checkout is elsewhere. Restart ComfyUI after cloning or updating so the changed node is registered. There are no extra pip dependencies for the node; it uses ComfyUI's own Python packages.

Keep the official [Viggle Remake ComfyUI node pack](https://huggingface.co/Viggle/Viggle-Remake) installed as well. The full workflow also uses its `RemakeKeyframeUntil` and `RemakeNoise` nodes.

## Use

Use an API-format workflow containing `RemakeConditioningExtended`. To adapt an official API graph, replace the conditioning node's `class_type` value `RemakeConditioning` with `RemakeConditioningExtended`. Set the frame count, short edge, reference images, and pins to match the source and task. This repository does not contain a Viggle workflow, model weights, LoRAs, or media.

To load the optional end-to-end runner into an editable Colab cell, run this magic in an empty cell:

```python
%load https://raw.githubusercontent.com/Yuvrajxms09/ComfyUI-Viggle-Remake-Extended/main/examples/colab_inference_cell.py
```

The runner expects the Gemini instruction files and official Viggle `PROMPTING.md` and API workflow to be available in the paths it checks. It reads `GOOGLE_API_KEY` and `MAGICHOUR_API_KEY` from Colab Secrets or environment variables.

## What changes

Compared with upstream Viggle Remake Conditioning, this copy registers a separate node class and expands the input bounds. Its conditioning path follows upstream; frame counts are validated against the accepted H3 grid, while unaligned API canvas values are rounded to the nearest 32-pixel size:

- Frame counts: **5–362**, on the H3 `17n+5` grid. The official Viggle sampler and node remain capped at 345; counts below 90 are also outside the documented reference range. A 360-frame source (15 seconds) can be padded by its caller to 362 model frames and trimmed back to 360 output frames. This path completed three 15-second inference runs on one ComfyUI 0.39.0 server; see [15-second validation](docs/15s-duration-validation.md). The 362-frame model input is 15.083 seconds, two frames beyond H3's documented maximum, so this remains experimental.
- Short edge: **480 through the installed ComfyUI `MAX_RESOLUTION`**, in steps of 32. H3 canvas dimensions are multiples of 32. The Colab resolution buckets treat their labels as maximum short edges and round down to stay within the cap: `480p` → 480, `720p` → 704, `1080p` → 1056. Raw unaligned API values are still rounded to the nearest valid size by the node.

At 24 fps, a two-second source has 48 frames. The longest legal unpadded grid prefix is 39 frames (1.625 seconds), leaving 9 source frames unused. The next legal count is 56 frames (2.333 seconds), which needs 8 padded frames. The node does not add that padding itself.

## Canvas and aspect ratio

The node has no general aspect-ratio input. It retains the source's aspect ratio only when the source short edge already equals the selected value and both dimensions are multiples of 32. Otherwise it resizes and center-crops to a 16:9-class landscape canvas or its portrait transpose. The first pinned reference is resized directly to the canvas when dimensions differ, so a mismatched reference can be stretched. Prepare the source and pinned image for the intended canvas if preserving their exact aspect ratio matters.

For a 16:9-class landscape canvas, the `1080p` maximum-edge bucket targets 1856×1056, the largest aligned short edge no greater than 1080. The next higher aligned canvas is 1920×1088 and exceeds that cap. Both dimensions are multiples of 32, as required by the H3 pipeline documentation. The node may center-crop a source whose dimensions do not match its selected canvas.

An earlier exact-1080 render failed in `SamplerCustomAdvanced` with a latent reshape error. At VAE stride 16, the 1080-pixel side produced 67 latent cells; the H3 transformer could not divide that odd dimension into 2-cell spatial patches. A raw API value of 1080 still rounds to nearest, 1088 (68 latent cells); the production `1080p` maximum-edge bucket instead sends 1056 (66 latent cells), which also avoids the recorded odd dimension while staying under the cap. See [resolution alignment and the recorded failure](docs/resolution-alignment.md). The exact case has not been rerun with either aligned size, so this is not an end-to-end success claim.

## Compatibility and status

This node is based on Viggle Remake's ComfyUI implementation at upstream revision [`21665fa7`](https://huggingface.co/Viggle/Viggle-Remake/commit/21665fa7a1e038e0a5312fb15f37b18d0f06f117). The copied implementation preserves the upstream conditioning behavior and changes its input bounds. The original custom-node pack remains unchanged.

The short-duration and higher-resolution settings are node-level capabilities, not evidence that the official sampler was trained or validated for every setting. The Colab `1080p` maximum-edge preset requests 1056 pixels, not exactly 1080. The 362-frame endpoint has completed three end-to-end inference runs in one setup, but it is outside H3's published 15-second frame limit and is not an official support claim. This repository contains the custom node only; a workflow or caller must select its class type explicitly. The node itself does not pad source videos or trim rendered outputs.

## License

The node code carries the upstream `SPDX-License-Identifier: Apache-2.0` notice and is distributed under Apache-2.0. See [`NOTICE`](NOTICE) for attribution. Viggle and MiniMax model weights, LoRAs, workflows, and other assets have their own terms and are not included here.
