# SPDX-License-Identifier: Apache-2.0
"""Experimental copy of Viggle Remake Conditioning with wider input ranges.

The conditioning implementation follows Viggle Remake's ComfyUI node. Only its
input bounds differ: legal frame counts span 5–362 on the 17n+5 grid, and
short-edge resolution follows ComfyUI's global integer limit instead of stopping
at 736. A caller can use 362 model frames for a 15-second source, then trim the
saved output to 15 seconds; this model length is experimental.
"""

import torch

import comfy.audio
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.utils
import node_helpers
import nodes


FPS, AUDIO_SR, AUDIO_LATENT_FPS = 24, 32000, 40


def lanczos(image, width, height):
    return comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)


def cover(image, width, height):
    h, w = image.shape[1], image.shape[2]
    scale = max(width / w, height / h)
    scaled_w, scaled_h = max(width, round(w * scale)), max(height, round(h * scale))
    left, top = (scaled_w - width) // 2, (scaled_h - height) // 2
    return lanczos(image, scaled_w, scaled_h)[:, top:top + height, left:left + width]


def stamps(frame_count):
    indices, cursor = [], 0.0
    while round(cursor) < frame_count:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += FPS / 2
    times = [index / 2 for index in range(len(indices))]
    times += times[-1:] * (len(times) % 2)
    return [(times[i] + times[i + 1]) / 2 for i in range(0, len(times), 2)]


class RemakeConditioningExtended:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "source": ("IMAGE", {"tooltip": "The source clip's frames at 24 fps."}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "The six-section prompt, used verbatim."}),
                "ref_1": ("IMAGE", {"tooltip": "<Picture 1>: the edited frame-0 reference."}),
                "pins": ("STRING", {"default": "", "tooltip": "One output frame per reference, e.g. '0'."}),
                "frames": ("INT", {
                    "default": 124, "min": 5, "max": 362, "step": 17,
                    "tooltip": "24 fps; valid lengths are 17n+5 from 5 to 362. The 362-frame input is experimental; trim its output to 15 seconds for a 15-second source.",
                }),
                "short_edge": ("INT", {
                    "default": 480, "min": 480, "max": (nodes.MAX_RESOLUTION // 32) * 32, "step": 32,
                    "tooltip": "Maximum short edge in pixels; unaligned API values are rounded down to a multiple of 32.",
                }),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "The source soundtrack; silence if not connected."}),
                "ref_2": ("IMAGE",),
                "ref_3": ("IMAGE",),
                "ref_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "run"
    CATEGORY = "Viggle Remake/Experimental"

    def run(self, clip, vae, audio_vae, source, prompt, ref_1, pins, frames, short_edge,
            audio=None, ref_2=None, ref_3=None, ref_4=None):
        frame_count, se = frames, short_edge
        if frame_count < 5 or frame_count > 362 or (frame_count - 5) % 17:
            raise ValueError(f"frames must be 17n+5 in 5..362, got {frame_count}")
        if se < 480 or se > nodes.MAX_RESOLUTION:
            raise ValueError(f"short_edge must be in 480..{nodes.MAX_RESOLUTION}, got {se}")
        requested_short_edge = se
        se = max(480, (requested_short_edge // 32) * 32)
        if se != requested_short_edge:
            print(f"RemakeConditioningExtended: rounding short_edge down from {requested_short_edge} to {se} (multiple of 32).")

        refs = [image[:1, ..., :3] for image in (ref_1, ref_2, ref_3, ref_4) if image is not None]
        pins = [int(value) for value in pins.replace(",", " ").split()]
        if len(pins) not in (0, len(refs)):
            raise ValueError(f"give one pin per reference image or none: {len(pins)} pins, {len(refs)} refs")
        if any(pin < 0 or pin >= frame_count for pin in pins):
            raise ValueError(f"pins must be in [0, {frame_count})")

        source = source[..., :3]
        height, width = source.shape[1], source.shape[2]
        if min(height, width) != se or height % 32 or width % 32:
            long_edge = se * 16 // 9 // 32 * 32
            width, height = (long_edge, se) if width >= height else (se, long_edge)
            source = comfy.utils.common_upscale(source.movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)
        source = source[:frame_count]
        if source.shape[0] != frame_count:
            raise ValueError(f"source has {source.shape[0]} frames; {frame_count} are required")

        segments = ["<Video 1>: ", *["<%.1f seconds>" % time for time in stamps(frame_count)],
                    *["<Picture %d>: " % (index + 1) for index in range(len(refs))], prompt]
        tokens = {"qwen3vl_32b": [[token for segment in segments for token in clip.tokenize(segment)["qwen3vl_32b"][0]]]}
        positive = clip.encode_from_tokens_scheduled(tokens)

        video_latent = vae.encode(source)
        blocks = [{"kind": "video", "latent_t": video_latent.shape[2], "latent_h": height // 16,
                   "latent_w": width // 16, "ref_audio_t": 0, "latent": video_latent, "audio_latent": None}]
        for reference in refs:
            ref_h, ref_w = reference.shape[1], reference.shape[2]
            target_h = max(32, round(ref_h * se / min(ref_h, ref_w) / 32) * 32)
            target_w = max(32, round(ref_w * se / min(ref_h, ref_w) / 32) * 32)
            reference = reference if (ref_h, ref_w) == (target_h, target_w) else lanczos(reference, target_w, target_h)
            blocks.append({"kind": "image", "latent_h": target_h // 16, "latent_w": target_w // 16,
                           "latent": vae.encode(reference)})

        keyframes = []
        for index, (reference, pin) in enumerate(zip(refs, pins)):
            if (reference.shape[1], reference.shape[2]) != (height, width):
                reference = lanczos(reference, width, height) if index == 0 else cover(reference, width, height)
            keyframes.append({"resolved_frame_index": pin, "latent": vae.encode(reference)})
        values = {"minimax_refs": blocks, "minimax_keyframes": keyframes} if keyframes else {"minimax_refs": blocks}
        positive = node_helpers.conditioning_set_values(positive, values)

        sample_count = round((frame_count + 1) / FPS * AUDIO_SR)
        waveform = torch.zeros(2, sample_count)
        if audio is not None:
            waveform, sample_rate = audio["waveform"][0].float(), audio["sample_rate"]
            waveform = waveform.repeat(2, 1) if waveform.shape[0] == 1 else waveform[:2]
            waveform = waveform[:, :round(frame_count / FPS * sample_rate)]
            if sample_rate != AUDIO_SR:
                waveform = comfy.audio.resample(waveform, sample_rate, AUDIO_SR)
        waveform = torch.nn.functional.pad(waveform, (0, max(0, sample_count - waveform.shape[1])))[:, :sample_count]
        waveform = torch.nn.functional.pad(waveform, (0, -sample_count % 800))
        audio_latent = audio_vae.encode(waveform[None].movedim(1, -1))[..., :round(frame_count / FPS * AUDIO_LATENT_FPS)]

        video = torch.zeros(1, 24, (frame_count - 5) // 17 * 5 + 2, height // 16, width // 16)
        latent = {
            "samples": comfy.nested_tensor.NestedTensor((video, audio_latent)),
            "noise_mask": comfy.nested_tensor.NestedTensor((torch.ones_like(video), torch.zeros_like(audio_latent))),
        }
        return (positive, latent)


NODE_CLASS_MAPPINGS = {"RemakeConditioningExtended": RemakeConditioningExtended}
NODE_DISPLAY_NAME_MAPPINGS = {"RemakeConditioningExtended": "Viggle Remake Conditioning (Extended)"}
