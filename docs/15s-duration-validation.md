# 15-second inference validation

**Run date:** 2026-11-12

**Result:** three completed 15-second inference runs on ComfyUI 0.39.0 using `RemakeConditioningExtended`.

## Tested cases

| Source | Edit request | Input | Model input | Saved output |
|---|---|---:|---:|---:|
| ET red-carpet interview | Remove the woman | 360 frames · 854×480 · 24 fps · 15.000 s | 362 frames · 15.083 s | 360 frames · 832×480 · 24 fps · 15.000 s |
| ET red-carpet interview | Red sweater on the woman; black coat on the presenter | 360 frames · 854×480 · 24 fps · 15.000 s | 362 frames · 15.083 s | 360 frames · 832×480 · 24 fps · 15.000 s |
| Couch interview | Remove the woman | 360 frames · 854×480 · 24 fps · 15.000 s | 362 frames · 15.083 s | 360 frames · 832×480 · 24 fps · 15.000 s |

The existing edited first-frame references and six-section prompts were reused; no new image editing or prompt generation was part of these duration tests. Each run used the edit LoRA at strength 1.0, a frame-0 pin, the 480p bucket, Euler sampling, and six steps. The exact API graph, prompt, and references remain in the private experiment archive.

## Duration handling

At 24 fps, a 15-second source has 360 frames. The H3 temporal grid is `17n+5`, so 360 is not a valid model frame count. The caller cloned the final video frame twice and sent 362 frames to `RemakeConditioningExtended`; after generation it trimmed the result back to the original 360-frame source interval. The saved outputs were each confirmed as 360 frames and exactly 15.000 seconds.

The custom node only accepts the expanded frame-count range. Padding and output trimming are caller responsibilities. The extended request is 362 frames (15.083 seconds), two frames beyond H3's published 15-second maximum. The [official H3 Diffusers documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3) requires the `17n+5` frame grid and says the resulting duration must remain within 5–15 seconds. These runs show that this exact workflow completed on one server configuration; they do not extend the official H3 limit or establish behavior on other versions, weights, or prompts. The three tests verified duration plumbing, not visual edit quality.

## Tested model configuration

- ComfyUI: 0.39.0
- Conditioning: `RemakeConditioningExtended`, 362 frames
- UNet: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- Text encoder: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- Video VAE: `minimax_h3_video_vae_fp16.safetensors`
- Audio VAE: `minimax_h3_audio_vae_fp32.safetensors`
- LoRA: `viggle_remake_edit_comfyui.safetensors`, strength 1.0
- Sampling: Euler, six steps
- Pin: edited first frame at frame 0

This repository contains the custom node and runner, not the media or private run artifacts.
