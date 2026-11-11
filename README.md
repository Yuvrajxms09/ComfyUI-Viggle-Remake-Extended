# ComfyUI-Viggle-Remake-Extended

An experimental ComfyUI custom node that adds wider input bounds to Viggle Remake's `RemakeConditioning` node. It registers under the separate class type `RemakeConditioningExtended`; it does not overwrite the official node.

## Install in Colab

Clone this public repository directly into ComfyUI's custom-node folder:

```python
%cd /content/ComfyUI/custom_nodes
!git clone https://github.com/Yuvrajxms09/ComfyUI-Viggle-Remake-Extended.git
```

Change `/content/ComfyUI` if your ComfyUI checkout is elsewhere. Restart ComfyUI after cloning so the new node is registered. There are no extra pip dependencies; the module uses ComfyUI's own Python packages.

Keep the official [Viggle Remake ComfyUI node pack](https://huggingface.co/Viggle/Viggle-Remake) installed as well. The full workflow also uses its `RemakeKeyframeUntil` and `RemakeNoise` nodes.

## Use

Use an API-format workflow containing `RemakeConditioningExtended`. To adapt an official API graph, replace the conditioning node's `class_type` value `RemakeConditioning` with `RemakeConditioningExtended`. Set the frame count, short edge, reference images, and pins to match the source and task. This repository contains the custom node only; it does not contain a Viggle workflow, model weights, LoRAs, or media.

## What changes

Compared with upstream Viggle Remake Conditioning, this copy changes only the input bounds:

- Frame counts: **5–345**, on the H3 `17n+5` grid. The official reference sampler documents 90–345 frames; anything below 90 is experimental and has not been validated by an end-to-end inference.
- Short edge: **480 through the installed ComfyUI `MAX_RESOLUTION`**, in steps of 8. A value of 1080 is accepted when the ComfyUI maximum allows it.

At 24 fps, a two-second source has 48 frames. The longest legal unpadded grid prefix is 39 frames (1.625 seconds), leaving 9 source frames unused. The next legal count is 56 frames (2.333 seconds), which needs 8 padded frames. The node does not add that padding itself.

## Canvas and aspect ratio

The node has no general aspect-ratio input. It retains the source's aspect ratio only when the source short edge already equals the selected value and both dimensions are multiples of 32. Otherwise it resizes and center-crops to a 16:9-class landscape canvas or its portrait transpose. The first pinned reference is resized directly to the canvas when dimensions differ, so a mismatched reference can be stretched. Prepare the source and pinned image for the intended canvas if preserving their exact aspect ratio matters.

At a 1080 short edge, landscape canvas math targets 1920×1080. ComfyUI's MiniMax H3 VAE uses 16-pixel spatial compression, so 1080 is expected to align to 1072 during encoding. This is inferred from ComfyUI source code; this repository has not measured a generated output at that setting.

## Compatibility and status

This node is based on Viggle Remake's ComfyUI implementation at upstream revision [`21665fa7`](https://huggingface.co/Viggle/Viggle-Remake/commit/21665fa7a1e038e0a5312fb15f37b18d0f06f117). The copied implementation preserves the upstream conditioning behavior and changes its input bounds. The original custom-node pack remains unchanged.

The short-duration and 1080 settings are node-level capabilities, not evidence that the official sampler was trained or validated for those settings. The extended node has only been statically checked; no video inference results are claimed. The caller must also use a workflow with the extended node class type—the standard Viggle runner does not automatically switch to it.

## License

The node code carries the upstream `SPDX-License-Identifier: Apache-2.0` notice and is distributed under Apache-2.0. See [`NOTICE`](NOTICE) for attribution. Viggle and MiniMax model weights, LoRAs, workflows, and other assets have their own terms and are not included here.
